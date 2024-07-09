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

from functools import partial

import torch
import torch.nn as nn
from typing import Tuple, Optional

from evodocker.model.primitives import Linear, LayerNorm
from evodocker.utils.tensor_utils import add


class StructureInputEmbedder(nn.Module):
    """
    Embeds a subset of the input features.

    Implements a merge of Algorithms 3 and Algorithm 32.
    """

    def __init__(
        self,
        protein_tf_dim: int,
        ligand_tf_dim: int,
        ligand_bond_dim: int,
        c_z: int,
        c_m: int,
        relpos_k: int,
        prot_min_bin: float,
        prot_max_bin: float,
        prot_no_bins: int,
        lig_min_bin: float,
        lig_max_bin: float,
        lig_no_bins: int,
        inf: float = 1e8,
        **kwargs,
    ):
        """
        Args:
            tf_dim:
                Final dimension of the target features
            c_z:
                Pair embedding dimension
            c_m:
                MSA embedding dimension
            relpos_k:
                Window size used in relative positional encoding
        """
        super(StructureInputEmbedder, self).__init__()

        self.protein_tf_dim = protein_tf_dim
        self.ligand_tf_dim = ligand_tf_dim

        self.c_z = c_z
        self.c_m = c_m

        self.protein_linear_tf_z_i = Linear(protein_tf_dim, c_z)
        self.protein_linear_tf_z_j = Linear(protein_tf_dim, c_z)
        self.protein_linear_tf_m = Linear(protein_tf_dim, c_m)

        self.ligand_linear_tf_z_i = Linear(ligand_tf_dim, c_z)
        self.ligand_linear_tf_z_j = Linear(ligand_tf_dim, c_z)
        self.ligand_linear_bond_z = Linear(ligand_bond_dim, c_z)
        self.ligand_linear_tf_m = Linear(ligand_tf_dim, c_m)

        # RPE stuff
        self.relpos_k = relpos_k
        self.no_bins = 2 * relpos_k + 1
        self.linear_relpos = Linear(self.no_bins, c_z)

        # Recycling stuff
        self.prot_min_bin = prot_min_bin
        self.prot_max_bin = prot_max_bin
        self.prot_no_bins = prot_no_bins
        self.lig_min_bin = lig_min_bin
        self.lig_max_bin = lig_max_bin
        self.lig_no_bins = lig_no_bins
        self.inf = inf

        self.prot_recycling_linear = Linear(self.prot_no_bins, self.c_z)
        self.lig_recycling_linear = Linear(self.lig_no_bins, self.c_z)
        self.layer_norm_m = LayerNorm(self.c_m)
        self.layer_norm_z = LayerNorm(self.c_z)

    def relpos(self, ri: torch.Tensor):
        """
        Computes relative positional encodings

        Implements Algorithm 4.

        Args:
            ri:
                "residue_index" features of shape [*, N]
        """
        d = ri[..., None] - ri[..., None, :]
        boundaries = torch.arange(
            start=-self.relpos_k, end=self.relpos_k + 1, device=d.device
        )
        reshaped_bins = boundaries.view(((1,) * len(d.shape)) + (len(boundaries),))
        d = d[..., None] - reshaped_bins
        d = torch.abs(d)
        d = torch.argmin(d, dim=-1)
        d = nn.functional.one_hot(d, num_classes=len(boundaries)).float()
        d = d.to(ri.dtype)
        return self.linear_relpos(d)

    def _do_recycle(self, m, z, x, min_bin, max_bin, no_bins, recycling_linear, inplace_safe=False):
        m_update = self.layer_norm_m(m)
        if (inplace_safe):
            m.copy_(m_update)
            m_update = m

        # [*, N, N, C_z]
        z_update = self.layer_norm_z(z)
        if (inplace_safe):
            z.copy_(z_update)
            z_update = z

        # This squared method might become problematic in FP16 mode.
        bins = torch.linspace(
            min_bin,
            max_bin,
            no_bins,
            dtype=x.dtype,
            device=x.device,
            requires_grad=False,
        )
        squared_bins = bins ** 2
        upper = torch.cat(
            [squared_bins[1:], squared_bins.new_tensor([self.inf])], dim=-1
        )
        d = torch.sum((x[..., None, :] - x[..., None, :, :]) ** 2, dim=-1, keepdims=True)

        # [*, N, N, no_bins]
        d = ((d > squared_bins) * (d < upper)).type(x.dtype)

        # [*, N, N, C_z]
        d = recycling_linear(d)
        z_update = add(z_update, d, inplace_safe)

        return m_update, z_update

    def forward(
        self,
        protein_target_feat: torch.Tensor,
        residue_index: torch.Tensor,
        input_protein_coords: torch.Tensor,
        ref_ligand_positions: torch.Tensor,
        ligand_target_feat: torch.Tensor,
        ligand_bonds_feat: torch.Tensor,
        inplace_safe: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch: Dict containing
                "protein_target_feat":
                    Features of shape [*, N_res + N_lig_atoms, tf_dim]
                "residue_index":
                    Features of shape [*, N_res]
                input_protein_coords:
                    [*, N_res, 3] AF predicted C_beta coordinates supplied as input
                ligand_bonds_feat:
                    [*, N_lig_atoms, N_lig_atoms, tf_dim] ligand bonds features
        Returns:
            msa_emb:
                [*, N_clust, N_res + N_lig_atoms, C_m] MSA embedding
            pair_emb:
                [*, N_res + N_lig_atoms, N_res + N_lig_atoms, C_z] pair embedding

        """
        N_res = residue_index.shape[-1]
        N_lig_atoms = ligand_bonds_feat.shape[-2]
        N_batch = protein_target_feat.shape[0]
        device = protein_target_feat.device

        # Single representation embedding - Algorithm 3
        protein_tf_m = self.protein_linear_tf_m(protein_target_feat)
        ligand_tf_m = self.ligand_linear_tf_m(ligand_target_feat)

        # Pair representation
        # protein pair embedding - Algorithm 3
        # [*, N_res, c_z]
        prot_tf_emb_i = self.protein_linear_tf_z_i(protein_target_feat)
        prot_tf_emb_j = self.protein_linear_tf_z_j(protein_target_feat)

        # [*, N_res, N_res, c_z]
        protein_pair_emb = self.relpos(residue_index.type(prot_tf_emb_i.dtype))
        protein_pair_emb = add(protein_pair_emb,
            prot_tf_emb_i[..., None, :],
            inplace=inplace_safe
        )
        protein_pair_emb = add(protein_pair_emb,
            prot_tf_emb_j[..., None, :, :],
            inplace=inplace_safe
        )
        protein_tf_m, protein_pair_emb = self._do_recycle(protein_tf_m, protein_pair_emb, input_protein_coords,
                                                          self.prot_min_bin, self.prot_max_bin, self.prot_no_bins,
                                                          self.prot_recycling_linear, inplace_safe)

        # ligand pair embedding
        ligand_pair_emb = self.ligand_linear_bond_z(ligand_bonds_feat)

        lig_tf_emb_i = self.ligand_linear_tf_z_i(ligand_target_feat)
        lig_tf_emb_j = self.ligand_linear_tf_z_j(ligand_target_feat)

        ligand_pair_emb = add(ligand_pair_emb,
                              lig_tf_emb_i[..., None, :],
                              inplace=inplace_safe
                              )
        ligand_pair_emb = add(ligand_pair_emb,
                              lig_tf_emb_j[..., None, :, :],
                              inplace=inplace_safe
                              )

        ligand_tf_m, ligand_pair_emb = self._do_recycle(ligand_tf_m, ligand_pair_emb, ref_ligand_positions,
                                                        self.lig_min_bin, self.lig_max_bin, self.lig_no_bins,
                                                        self.lig_recycling_linear, inplace_safe)

        # joint representation embedding
        joint1 = prot_tf_emb_i.unsqueeze(2) + lig_tf_emb_j.unsqueeze(1)
        joint2 = (prot_tf_emb_j.unsqueeze(2) + lig_tf_emb_i.unsqueeze(1)).transpose(1, 2)

        # Merge protein and ligand embeddings
        tf_m = torch.cat([protein_tf_m, ligand_tf_m], dim=-2)
        pair_emb = torch.zeros(N_batch, N_res + N_lig_atoms, N_res + N_lig_atoms, self.c_z, device=device)
        pair_emb[..., :N_res, :N_res, :] = protein_pair_emb
        pair_emb[..., N_res:, N_res:, :] = ligand_pair_emb
        pair_emb[..., :N_res, N_res:, :] = joint1
        pair_emb[..., N_res:, :N_res, :] = joint2

        return tf_m, pair_emb


class RecyclingEmbedder(nn.Module):
    """
    Embeds the output of an iteration of the model for recycling.

    Implements Algorithm 32.
    """
    def __init__(
        self,
        c_m: int,
        c_z: int,
        min_bin: float,
        max_bin: float,
        no_bins: int,
        inf: float = 1e8,
        **kwargs,
    ):
        """
        Args:
            c_m:
                MSA channel dimension
            c_z:
                Pair embedding channel dimension
            min_bin:
                Smallest distogram bin (Angstroms)
            max_bin:
                Largest distogram bin (Angstroms)
            no_bins:
                Number of distogram bins
        """
        super(RecyclingEmbedder, self).__init__()

        self.c_m = c_m
        self.c_z = c_z
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.no_bins = no_bins
        self.inf = inf

        self.linear = Linear(self.no_bins, self.c_z)
        self.layer_norm_m = LayerNorm(self.c_m)
        self.layer_norm_z = LayerNorm(self.c_z)

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        x: torch.Tensor,
        inplace_safe: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            m:
                First row of the MSA embedding. [*, N_res, C_m]
            z:
                [*, N_res, N_res, C_z] pair embedding
            x:
                [*, N_res, 3] predicted C_beta coordinates
        Returns:
            m:
                [*, N_res, C_m] MSA embedding update
            z:
                [*, N_res, N_res, C_z] pair embedding update
        """
        # [*, N, C_m]
        m_update = self.layer_norm_m(m)
        if(inplace_safe):
            m.copy_(m_update)
            m_update = m

        # [*, N, N, C_z]
        z_update = self.layer_norm_z(z)
        if(inplace_safe):
            z.copy_(z_update)
            z_update = z

        # This squared method might become problematic in FP16 mode.
        bins = torch.linspace(
            self.min_bin,
            self.max_bin,
            self.no_bins,
            dtype=x.dtype,
            device=x.device,
            requires_grad=False,
        )
        squared_bins = bins ** 2
        upper = torch.cat(
            [squared_bins[1:], squared_bins.new_tensor([self.inf])], dim=-1
        )
        d = torch.sum(
            (x[..., None, :] - x[..., None, :, :]) ** 2, dim=-1, keepdims=True
        )

        # [*, N, N, no_bins]
        d = ((d > squared_bins) * (d < upper)).type(x.dtype)

        # [*, N, N, C_z]
        d = self.linear(d)
        z_update = add(z_update, d, inplace_safe)

        return m_update, z_update

