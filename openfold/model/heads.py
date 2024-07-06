# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn

from openfold.model.primitives import Linear, LayerNorm
from openfold.utils.loss import (
    compute_plddt,
    compute_tm,
    compute_predicted_aligned_error,
)
from openfold.utils.precision_utils import is_fp16_enabled


class AuxiliaryHeads(nn.Module):
    def __init__(self, config):
        super(AuxiliaryHeads, self).__init__()

        self.plddt = PerResidueLDDTCaPredictor(
            **config["lddt"],
        )

        self.distogram = DistogramHead(
            **config["distogram"],
        )

        self.experimentally_resolved = ExperimentallyResolvedHead(
            **config["experimentally_resolved"],
        )

        self.affinity_2d = Affinity2DPredictor(
            **config["affinity_2d"],
        )

        self.affinity_1d = Affinity1DPredictor(
            **config["affinity_1d"],
        )

        self.affinity_cls = AffinityClsTokenPredictor(
            **config["affinity_cls"],
        )

        self.binding_site = BindingSitePredictor(
            **config["binding_site"],
        )

        self.inter_contact = InterContactHead(
            **config["inter_contact"],
        )

        self.config = config

    def forward(self, outputs):
        aux_out = {}
        lddt_logits = self.plddt(outputs["sm"]["single"])
        aux_out["lddt_logits"] = lddt_logits

        # Required for relaxation later on
        aux_out["plddt"] = compute_plddt(lddt_logits)

        distogram_logits = self.distogram(outputs["pair"])
        aux_out["distogram_logits"] = distogram_logits

        experimentally_resolved_logits = self.experimentally_resolved(
            outputs["single"]
        )
        aux_out["experimentally_resolved_logits"] = experimentally_resolved_logits

        aux_out["inter_contact_logits"] = self.inter_contact(outputs["single"], outputs["pair"],
                                                             outputs["start_ligand_ind"])

        aux_out["affinity_2d_logits"] = self.affinity_2d(outputs["pair"], outputs["start_ligand_ind"],
                                                         aux_out["inter_contact_logits"])

        aux_out["affinity_1d_logits"] = self.affinity_1d(outputs["single"])

        aux_out["affinity_cls_logits"] = self.affinity_cls(outputs["affinity_token"])

        aux_out["binding_site_logits"] = self.binding_site(outputs["single"], outputs["start_ligand_ind"])

        return aux_out


class Affinity2DPredictor(nn.Module):
    def __init__(self, c_z, num_bins):
        super(Affinity2DPredictor, self).__init__()

        self.c_z = c_z

        self.fc1 = Linear(self.c_z, self.c_z)
        self.weight_linear = Linear(self.c_z, 1)
        self.fc2 = Linear(self.c_z, num_bins)

    def forward(self, z, start_ligand_ind, inter_contacts_logits):
        # Extract interface part of Z
        x = z[:, :start_ligand_ind, start_ligand_ind:, :]

        x = self.fc1(x)

        batch_size, N, M, _ = x.shape
        x_flat = x.view(batch_size, N * M, -1)
        inter_contacts_flat_sigmoid = torch.sigmoid(inter_contacts_logits.reshape(batch_size, N * M, 1))
        linear_weight_logits = self.weight_linear(x_flat)
        weights = torch.softmax(linear_weight_logits * inter_contacts_flat_sigmoid, dim=1)
        weighted_sum = torch.sum(weights * x_flat, dim=1)

        affinity_logits = self.fc2(weighted_sum)

        return affinity_logits


class Affinity1DPredictor(nn.Module):
    def __init__(self, c_s, num_bins, **kwargs):
        super(Affinity1DPredictor, self).__init__()

        self.c_s = c_s

        self.linear1 = Linear(self.c_s, self.c_s, init="final")

        self.linear2 = Linear(self.c_s, num_bins, init="final")

    def forward(self, s):
        # [*, N, C_out]
        s = self.linear1(s)

        # get an average over the sequence
        s = torch.mean(s, dim=1)

        logits = self.linear2(s)
        return logits


class AffinityClsTokenPredictor(nn.Module):
    def __init__(self, c_s, num_bins, **kwargs):
        super(AffinityClsTokenPredictor, self).__init__()

        self.c_s = c_s
        self.linear = Linear(self.c_s, num_bins, init="final")

    def forward(self, s):
        return self.linear(s)


class BindingSitePredictor(nn.Module):
    def __init__(self, c_s, c_out, **kwargs):
        super(BindingSitePredictor, self).__init__()

        self.c_s = c_s
        self.c_out = c_out

        self.linear = Linear(self.c_s, self.c_out, init="final")

    def forward(self, s, start_ligand_ind):
        # [*, N, C_out]
        logits = self.linear(s[:, :start_ligand_ind, :])
        return logits


class InterContactHead(nn.Module):
    def __init__(self, c_s, c_z, c_out, **kwargs):
        """
        Args:
            c_z:
                Input channel dimension
            no_bins:
                Number of atoms per residue (37)
        """
        super(InterContactHead, self).__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_out = c_out

        self.linear = Linear(2 * self.c_s + self.c_z, self.c_out, init="final")

    def forward(self, s, z, start_ligand_ind):  # [*, N, N, C_z]
        # [*, N, N, no_bins]
        batch_size, n, s_dim = s.shape

        s_i = s.unsqueeze(2).expand(batch_size, n, n, s_dim)
        s_j = s.unsqueeze(1).expand(batch_size, n, n, s_dim)
        joined = torch.cat((s_i, s_j, z), dim=-1)

        logits = self.linear(joined)
        logits = logits + logits.transpose(-2, -3)

        inter_logits = logits[:, :start_ligand_ind, start_ligand_ind:]

        return inter_logits


class PerResidueLDDTCaPredictor(nn.Module):
    def __init__(self, no_bins, c_in, c_hidden):
        super(PerResidueLDDTCaPredictor, self).__init__()

        self.no_bins = no_bins
        self.c_in = c_in
        self.c_hidden = c_hidden

        self.layer_norm = LayerNorm(self.c_in)

        self.linear_1 = Linear(self.c_in, self.c_hidden, init="relu")
        self.linear_2 = Linear(self.c_hidden, self.c_hidden, init="relu")
        self.linear_3 = Linear(self.c_hidden, self.no_bins, init="final")

        self.relu = nn.ReLU()

    def forward(self, s):
        s = self.layer_norm(s)
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)

        return s


class DistogramHead(nn.Module):
    """
    Computes a distogram probability distribution.

    For use in computation of distogram loss, subsection 1.9.8
    """

    def __init__(self, c_z, no_bins, **kwargs):
        """
        Args:
            c_z:
                Input channel dimension
            no_bins:
                Number of distogram bins
        """
        super(DistogramHead, self).__init__()

        self.c_z = c_z
        self.no_bins = no_bins

        self.linear = Linear(self.c_z, self.no_bins, init="final")

    def _forward(self, z):  # [*, N, N, C_z]
        """
        Args:
            z:
                [*, N_res, N_res, C_z] pair embedding
        Returns:
            [*, N, N, no_bins] distogram probability distribution
        """
        # [*, N, N, no_bins]
        logits = self.linear(z)
        logits = logits + logits.transpose(-2, -3)
        return logits
    
    def forward(self, z): 
        if(is_fp16_enabled()):
            with torch.cuda.amp.autocast(enabled=False):
                return self._forward(z.float())
        else:
            return self._forward(z)


class ExperimentallyResolvedHead(nn.Module):
    """
    For use in computation of "experimentally resolved" loss, subsection
    1.9.10
    """

    def __init__(self, c_s, c_out, **kwargs):
        """
        Args:
            c_s:
                Input channel dimension
            c_out:
                Number of distogram bins
        """
        super(ExperimentallyResolvedHead, self).__init__()

        self.c_s = c_s
        self.c_out = c_out

        self.linear = Linear(self.c_s, self.c_out, init="final")

    def forward(self, s):
        """
        Args:
            s:
                [*, N_res, C_s] single embedding
        Returns:
            [*, N, C_out] logits
        """
        # [*, N, C_out]
        logits = self.linear(s)
        return logits
