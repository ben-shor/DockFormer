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
from torch.nn import Parameter

from evodocker.model.primitives import Linear, LayerNorm
from evodocker.utils.loss import (
    compute_plddt,
    compute_tm,
    compute_predicted_aligned_error,
)
from evodocker.utils.precision_utils import is_fp16_enabled


class AuxiliaryHeads(nn.Module):
    def __init__(self, config):
        super(AuxiliaryHeads, self).__init__()

        self.plddt = PerResidueLDDTCaPredictor(
            **config["lddt"],
        )

        self.distogram = DistogramHead(
            **config["distogram"],
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

    def forward(self, outputs, inter_mask, affinity_mask):
        aux_out = {}
        lddt_logits = self.plddt(outputs["sm"]["single"])
        aux_out["lddt_logits"] = lddt_logits

        # Required for relaxation later on
        aux_out["plddt"] = compute_plddt(lddt_logits)

        distogram_logits = self.distogram(outputs["pair"])
        aux_out["distogram_logits"] = distogram_logits

        aux_out["inter_contact_logits"] = self.inter_contact(outputs["single"], outputs["pair"])

        aux_out["affinity_2d_logits"] = self.affinity_2d(outputs["pair"], aux_out["inter_contact_logits"], inter_mask)

        aux_out["affinity_1d_logits"] = self.affinity_1d(outputs["single"])

        aux_out["affinity_cls_logits"] = self.affinity_cls(outputs["single"], affinity_mask)

        aux_out["binding_site_logits"] = self.binding_site(outputs["single"])

        return aux_out


class Affinity2DPredictor(nn.Module):
    def __init__(self, c_z, num_bins):
        super(Affinity2DPredictor, self).__init__()

        self.c_z = c_z

        self.weight_linear = Linear(self.c_z + 1, 1)
        self.embed_linear = Linear(self.c_z, self.c_z)
        self.bins_linear = Linear(self.c_z, num_bins)

    def forward(self, z, inter_contacts_logits, inter_pair_mask):
        z_with_inter_contacts = torch.cat((z, inter_contacts_logits), dim=-1)  # [*, N, N, c_z + 1]
        weights = self.weight_linear(z_with_inter_contacts)  # [*, N, N, 1]

        x = self.embed_linear(z)  # [*, N, N, c_z]
        batch_size, N, M, _ = x.shape

        flat_weights = weights.reshape(batch_size, N*M, -1)  # [*, N*M, 1]
        flat_x = x.reshape(batch_size, N*M, -1)  # [*, N*M, c_z]
        flat_inter_pair_mask = inter_pair_mask.reshape(batch_size, N*M, 1)

        flat_weights = flat_weights.masked_fill(~(flat_inter_pair_mask.bool()), float('-inf'))  # [*, N*N, 1]
        flat_weights = torch.nn.functional.softmax(flat_weights, dim=1)  # [*, N*N, 1]
        flat_weights = torch.nan_to_num(flat_weights, nan=0.0)  # [*, N*N, 1]
        weighted_sum = torch.sum((flat_weights * flat_x).reshape(batch_size, N*M, -1), dim=1)  # [*, c_z]

        return self.bins_linear(weighted_sum)


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

    def forward(self, s, affinity_mask):
        affinity_tokens = (s * affinity_mask.unsqueeze(-1)).sum(dim=1)
        return self.linear(affinity_tokens)


class BindingSitePredictor(nn.Module):
    def __init__(self, c_s, c_out, **kwargs):
        super(BindingSitePredictor, self).__init__()

        self.c_s = c_s
        self.c_out = c_out

        self.linear = Linear(self.c_s, self.c_out, init="final")

    def forward(self, s):
        # [*, N, C_out]
        return self.linear(s)


class InterContactHead(nn.Module):
    def __init__(self, c_s, c_z, c_out, **kwargs):
        """
        Args:
            c_z:
                Input channel dimension
            c_out:
                Number of bins, but since boolean should be 1
        """
        super(InterContactHead, self).__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_out = c_out

        self.linear = Linear(2 * self.c_s + self.c_z, self.c_out, init="final")

    def forward(self, s, z):  # [*, N, N, C_z]
        # [*, N, N, no_bins]
        batch_size, n, s_dim = s.shape

        s_i = s.unsqueeze(2).expand(batch_size, n, n, s_dim)
        s_j = s.unsqueeze(1).expand(batch_size, n, n, s_dim)
        joined = torch.cat((s_i, s_j, z), dim=-1)

        logits = self.linear(joined)

        return logits


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
