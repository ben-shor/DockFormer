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
import weakref

import torch
import torch.nn as nn

from dockformer.utils.tensor_utils import masked_mean
from dockformer.model.embedders import (
    StructureInputEmbedder,
    RecyclingEmbedder,
)
from dockformer.model.evoformer import EvoformerStack
from dockformer.model.heads import AuxiliaryHeads
from dockformer.model.structure_module import StructureModule
import dockformer.utils.residue_constants as residue_constants
from dockformer.utils.feats import (
    pseudo_beta_fn,
    atom14_to_atom37,
)
from dockformer.utils.tensor_utils import (
    add,
    tensor_tree_map,
)


class AlphaFold(nn.Module):
    """
    Alphafold 2.

    Implements Algorithm 2 (but with training).
    """

    def __init__(self, config):
        """
        Args:
            config:
                A dict-like config object (like the one in config.py)
        """
        super(AlphaFold, self).__init__()

        self.globals = config.globals
        self.config = config.model

        # Main trunk + structure module
        self.input_embedder = StructureInputEmbedder(
            **self.config["structure_input_embedder"],
        )

        self.recycling_embedder = RecyclingEmbedder(
            **self.config["recycling_embedder"],
        )

        self.evoformer = EvoformerStack(
            **self.config["evoformer_stack"],
        )

        self.structure_module = StructureModule(
            **self.config["structure_module"],
        )
        self.aux_heads = AuxiliaryHeads(
            self.config["heads"],
        )

    def tolerance_reached(self, prev_pos, next_pos, mask, eps=1e-8) -> bool:
        """
        Early stopping criteria based on criteria used in
        AF2Complex: https://www.nature.com/articles/s41467-022-29394-2
        Args:
          prev_pos: Previous atom positions in atom37/14 representation
          next_pos: Current atom positions in atom37/14 representation
          mask: 1-D sequence mask
          eps: Epsilon used in square root calculation
        Returns:
          Whether to stop recycling early based on the desired tolerance.
        """

        def distances(points):
            """Compute all pairwise distances for a set of points."""
            d = points[..., None, :] - points[..., None, :, :]
            return torch.sqrt(torch.sum(d ** 2, dim=-1))

        if self.config.recycle_early_stop_tolerance < 0:
            return False

        ca_idx = residue_constants.atom_order['CA']
        sq_diff = (distances(prev_pos[..., ca_idx, :]) - distances(next_pos[..., ca_idx, :])) ** 2
        mask = mask[..., None] * mask[..., None, :]
        sq_diff = masked_mean(mask=mask, value=sq_diff, dim=list(range(len(mask.shape))))
        diff = torch.sqrt(sq_diff + eps).item()
        return diff <= self.config.recycle_early_stop_tolerance

    def iteration(self, feats, prevs, _recycle=True):
        # Primary output dictionary
        outputs = {}

        # This needs to be done manually for DeepSpeed's sake
        dtype = next(self.parameters()).dtype
        for k in feats:
            if feats[k].dtype == torch.float32:
                feats[k] = feats[k].to(dtype=dtype)

        # Grab some data about the input
        batch_dims, n_total = feats["token_mask"].shape
        device = feats["token_mask"].device

        print("doing sample of size", feats["token_mask"].shape,
              feats["protein_mask"].sum(dim=1), feats["ligand_mask"].sum(dim=1))

        # Controls whether the model uses in-place operations throughout
        # The dual condition accounts for activation checkpoints
        # inplace_safe = not (self.training or torch.is_grad_enabled())
        inplace_safe = False # so we don't need attn_core_inplace_cuda

        # Prep some features
        token_mask = feats["token_mask"]
        pair_mask = token_mask[..., None] * token_mask[..., None, :]

        # Initialize the single and pair representations
        # m: [*, 1, n_total, C_m]
        # z: [*, n_total, n_total, C_z]
        m, z = self.input_embedder(
            feats["token_mask"],
            feats["protein_mask"],
            feats["ligand_mask"],
            feats["target_feat"],
            feats["ligand_bonds_feat"],
            feats["input_positions"],
            feats["protein_residue_index"],
            feats["protein_distogram_mask"],
            inplace_safe=inplace_safe,
        )

        # Unpack the recycling embeddings. Removing them from the list allows 
        # them to be freed further down in this function, saving memory
        m_1_prev, z_prev, x_prev = reversed([prevs.pop() for _ in range(3)])

        # Initialize the recycling embeddings, if needs be 
        if None in [m_1_prev, z_prev, x_prev]:
            # [*, N, C_m]
            m_1_prev = m.new_zeros(
                (batch_dims, n_total, self.config.structure_input_embedder.c_m),
                requires_grad=False,
            )

            # [*, N, N, C_z]
            z_prev = z.new_zeros(
                (batch_dims, n_total, n_total, self.config.structure_input_embedder.c_z),
                requires_grad=False,
            )

            # [*, N, 3]
            x_prev = z.new_zeros(
                (batch_dims, n_total, residue_constants.atom_type_num, 3),
                requires_grad=False,
            )

        # shape == [1, n_total, 37, 3]
        pseudo_beta_or_lig_x_prev = pseudo_beta_fn(feats["aatype"], x_prev, None).to(dtype=z.dtype)

        # m_1_prev_emb: [*, N, C_m]
        # z_prev_emb: [*, N, N, C_z]
        m_1_prev_emb, z_prev_emb = self.recycling_embedder(
            m_1_prev,
            z_prev,
            pseudo_beta_or_lig_x_prev,
            inplace_safe=inplace_safe,
        )

        del pseudo_beta_or_lig_x_prev

        # [*, S_c, N, C_m]
        m += m_1_prev_emb

        # [*, N, N, C_z]
        z = add(z, z_prev_emb, inplace=inplace_safe)

        # Deletions like these become significant for inference with large N,
        # where they free unused tensors and remove references to others such
        # that they can be offloaded later
        del m_1_prev, z_prev, m_1_prev_emb, z_prev_emb

        # Run single + pair embeddings through the trunk of the network
        # m: [*, N, C_m]
        # z: [*, N, N, C_z]
        # s: [*, N, C_s]
        m, z, s = self.evoformer(
            m,
            z,
            single_mask=token_mask.to(dtype=m.dtype),
            pair_mask=pair_mask.to(dtype=z.dtype),
            use_lma=self.globals.use_lma,
            inplace_safe=inplace_safe,
            _mask_trans=self.config._mask_trans,
        )

        outputs["pair"] = z
        outputs["single"] = s

        del z

        # Predict 3D structure
        outputs["sm"] = self.structure_module(
            outputs,
            feats["aatype"],
            mask=token_mask.to(dtype=s.dtype),
            inplace_safe=inplace_safe,
        )
        outputs["final_atom_positions"] = atom14_to_atom37(
            outputs["sm"]["positions"][-1], feats
        )
        outputs["final_atom_mask"] = feats["atom37_atom_exists"]

        # Save embeddings for use during the next recycling iteration

        # [*, N, C_m]
        m_1_prev = m[..., 0, :, :]

        # [*, N, N, C_z]
        z_prev = outputs["pair"]

        # TODO bshor: early stop depends on is_multimer, but I don't think it must
        early_stop = False
        # if self.globals.is_multimer:
        #     early_stop = self.tolerance_reached(x_prev, outputs["final_atom_positions"], seq_mask)

        del x_prev

        # [*, N, 3]
        x_prev = outputs["final_atom_positions"]

        return outputs, m_1_prev, z_prev, x_prev, early_stop

    def forward(self, batch):
        """
        Args:
            batch:
                Dictionary of arguments outlined in Algorithm 2. Keys must
                include the official names of the features in the
                supplement subsection 1.2.9.

                The final dimension of each input must have length equal to
                the number of recycling iterations.

                Features (without the recycling dimension):

                    "aatype" ([*, N_res]):
                        Contrary to the supplement, this tensor of residue
                        indices is not one-hot.
                    "protein_target_feat" ([*, N_res, C_tf])
                        One-hot encoding of the target sequence. C_tf is
                        config.model.input_embedder.tf_dim.
                    "residue_index" ([*, N_res])
                        Tensor whose final dimension consists of
                        consecutive indices from 0 to N_res.
                    "token_mask" ([*, N_token])
                        1-D token mask
                    "pair_mask" ([*, N_token, N_token])
                        2-D pair mask
        """
        # Initialize recycling embeddings
        m_1_prev, z_prev, x_prev = None, None, None
        prevs = [m_1_prev, z_prev, x_prev]

        is_grad_enabled = torch.is_grad_enabled()

        # Main recycling loop
        num_iters = batch["aatype"].shape[-1]
        early_stop = False
        num_recycles = 0
        for cycle_no in range(num_iters):
            # Select the features for the current recycling cycle
            fetch_cur_batch = lambda t: t[..., cycle_no]
            feats = tensor_tree_map(fetch_cur_batch, batch)

            # Enable grad iff we're training and it's the final recycling layer
            is_final_iter = cycle_no == (num_iters - 1) or early_stop
            with torch.set_grad_enabled(is_grad_enabled and is_final_iter):
                if is_final_iter:
                    # Sidestep AMP bug (PyTorch issue #65766)
                    if torch.is_autocast_enabled():
                        torch.clear_autocast_cache()

                # Run the next iteration of the model
                outputs, m_1_prev, z_prev, x_prev, early_stop = self.iteration(
                    feats,
                    prevs,
                    _recycle=(num_iters > 1)
                )

                num_recycles += 1

                if not is_final_iter:
                    del outputs
                    prevs = [m_1_prev, z_prev, x_prev]
                    del m_1_prev, z_prev, x_prev
                else:
                    break

        outputs["num_recycles"] = torch.tensor(num_recycles, device=feats["aatype"].device)

        # Run auxiliary heads, remove the recycling dimension batch properties
        outputs.update(self.aux_heads(outputs, batch["inter_pair_mask"][..., 0], batch["affinity_mask"][..., 0],
                                      batch["ligand_mask"][..., 0]))

        return outputs
