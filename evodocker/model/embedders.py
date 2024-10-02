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
        additional_tf_dim: int,
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
                Single embedding dimension
            relpos_k:
                Window size used in relative positional encoding
        """
        super(StructureInputEmbedder, self).__init__()

        self.tf_dim = protein_tf_dim + ligand_tf_dim + additional_tf_dim
        self.pair_tf_dim = ligand_bond_dim

        self.c_z = c_z
        self.c_m = c_m

        self.linear_tf_z_i = Linear(self.tf_dim, c_z)
        self.linear_tf_z_j = Linear(self.tf_dim, c_z)
        self.linear_tf_m = Linear(self.tf_dim, c_m)

        self.ligand_linear_bond_z = Linear(ligand_bond_dim, c_z)

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

        self.prot_recycling_linear = Linear(self.prot_no_bins + 1, self.c_z)
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

    def _get_binned_distogram(self, x, min_bin, max_bin, no_bins, recycling_linear, prot_distogram_mask=None):
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
        # print("d shape", d.shape, d[0][0][:10])

        if prot_distogram_mask is not None:
            expanded_d = torch.cat([d, torch.zeros(*d.shape[:-1], 1, device=d.device)], dim=-1)

            # Step 2: Create a mask where `input_positions_masked` is 0
            # Use broadcasting and tensor operations directly without additional variables
            input_positions_mask = (prot_distogram_mask == 1).float()  # Shape [N, crop_size]
            mask_i = input_positions_mask.unsqueeze(2)  # Shape [N, crop_size, 1]
            mask_j = input_positions_mask.unsqueeze(1)  # Shape [N, 1, crop_size]

            # Step 3: Combine masks for both [N, :, i, :] and [N, i, :, :]
            combined_mask = mask_i + mask_j  # Shape [N, crop_size, crop_size]
            combined_mask = combined_mask.clamp(max=1)  # Ensure binary mask

            # Step 4: Apply the mask
            # a. Set all but the last position in the `no_bins + 1` dimension to 0 where the mask is 1
            expanded_d[..., :-1] *= (1 - combined_mask).unsqueeze(-1)  # Shape [N, crop_size, crop_size, no_bins]

            # print("expanded_d shape1", expanded_d.shape, expanded_d[0][0][:10])

            # b. Set the last position in the `no_bins + 1` dimension to 1 where the mask is 1
            expanded_d[..., -1] += combined_mask  # Shape [N, crop_size, crop_size, 1]
            d = expanded_d
            # print("expanded_d shape2", d.shape, d[0][0][:10])

        return recycling_linear(d)

    def forward(
        self,
        token_mask: torch.Tensor,
        protein_mask: torch.Tensor,
        ligand_mask: torch.Tensor,
        target_feat: torch.Tensor,
        ligand_bonds_feat: torch.Tensor,
        input_positions: torch.Tensor,
        protein_residue_index: torch.Tensor,
        protein_distogram_mask: torch.Tensor,
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
            single_emb:
                [*, N_res + N_lig_atoms, C_m] single embedding
            pair_emb:
                [*, N_res + N_lig_atoms, N_res + N_lig_atoms, C_z] pair embedding

        """
        device = token_mask.device
        pair_protein_mask = protein_mask[..., None] * protein_mask[..., None, :]
        pair_ligand_mask = ligand_mask[..., None] * ligand_mask[..., None, :]

        # Single representation embedding - Algorithm 3
        tf_m = self.linear_tf_m(target_feat)
        tf_m = self.layer_norm_m(tf_m)  # previously this happend in the do_recycle function

        # Pair representation
        # protein pair embedding - Algorithm 3
        # [*, N_res, c_z]
        tf_emb_i = self.linear_tf_z_i(target_feat)
        tf_emb_j = self.linear_tf_z_j(target_feat)

        pair_emb = torch.zeros(*pair_protein_mask.shape, self.c_z, device=device)
        pair_emb = add(pair_emb, tf_emb_i[..., None, :], inplace=inplace_safe)
        pair_emb = add(pair_emb, tf_emb_j[..., None, :, :], inplace=inplace_safe)

        # Apply relpos
        relpos = self.relpos(protein_residue_index.type(tf_emb_i.dtype))
        pair_emb += pair_protein_mask[..., None] * relpos

        del relpos

        # apply ligand bonds
        ligand_bonds = self.ligand_linear_bond_z(ligand_bonds_feat)
        pair_emb += pair_ligand_mask[..., None] * ligand_bonds

        del ligand_bonds

        # before recycles, do z_norm, this previously was a part of the recycles
        pair_emb = self.layer_norm_z(pair_emb)

        # apply protein recycle
        prot_distogram_embed = self._get_binned_distogram(input_positions, self.prot_min_bin, self.prot_max_bin,
                                                          self.prot_no_bins, self.prot_recycling_linear,
                                                          protein_distogram_mask)


        pair_emb = add(pair_emb, prot_distogram_embed * pair_protein_mask.unsqueeze(-1), inplace_safe)

        del prot_distogram_embed

        # apply ligand recycle
        lig_distogram_embed = self._get_binned_distogram(input_positions, self.lig_min_bin, self.lig_max_bin,
                                                         self.lig_no_bins, self.lig_recycling_linear)
        pair_emb = add(pair_emb, lig_distogram_embed * pair_ligand_mask.unsqueeze(-1), inplace_safe)

        del lig_distogram_embed

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
                Single channel dimension
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
                First row of the single embedding. [*, N_res, C_m]
            z:
                [*, N_res, N_res, C_z] pair embedding
            x:
                [*, N_res, 3] predicted C_beta coordinates
        Returns:
            m:
                [*, N_res, C_m] single embedding update
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

