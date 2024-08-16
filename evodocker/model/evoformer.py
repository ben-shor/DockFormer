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
import math
import sys
import torch
import torch.nn as nn
from typing import Tuple, Sequence, Optional
from functools import partial
from abc import ABC, abstractmethod

from evodocker.model.primitives import Linear, LayerNorm
from evodocker.model.dropout import DropoutRowwise
from evodocker.model.single_attention import SingleRowAttentionWithPairBias

from evodocker.model.pair_transition import PairTransition
from evodocker.model.triangular_attention import (
    TriangleAttention,
)
from evodocker.model.triangular_multiplicative_update import (
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
)
from evodocker.utils.checkpointing import checkpoint_blocks
from evodocker.utils.tensor_utils import add


class SingleRepTransition(nn.Module):
    """
    Feed-forward network applied to single representation activations after attention.

    Implements Algorithm 9
    """
    def __init__(self, c_m, n):
        """
        Args:
            c_m:
                channel dimension
            n:
                Factor multiplied to c_m to obtain the hidden channel dimension
        """
        super(SingleRepTransition, self).__init__()

        self.c_m = c_m
        self.n = n

        self.layer_norm = LayerNorm(self.c_m)
        self.linear_1 = Linear(self.c_m, self.n * self.c_m, init="relu")
        self.relu = nn.ReLU()
        self.linear_2 = Linear(self.n * self.c_m, self.c_m, init="final")

    def _transition(self, m, mask):
        m = self.layer_norm(m)
        m = self.linear_1(m)
        m = self.relu(m)
        m = self.linear_2(m) * mask
        return m

    def forward(
        self,
        m: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_res, C_m] activation after attention
            mask:
                [*, N_res, C_m] mask
        Returns:
            m:
                [*, N_res, C_m] activation update
        """
        # DISCREPANCY: DeepMind forgets to apply the mask here.
        if mask is None:
            mask = m.new_ones(m.shape[:-1])

        mask = mask.unsqueeze(-1)

        m = self._transition(m, mask)

        return m


class PairStack(nn.Module):
    def __init__(
        self,
        c_z: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_pair: int,
        transition_n: int,
        pair_dropout: float,
        inf: float,
        eps: float
    ):
        super(PairStack, self).__init__()

        self.tri_mul_out = TriangleMultiplicationOutgoing(
            c_z,
            c_hidden_mul,
        )
        self.tri_mul_in = TriangleMultiplicationIncoming(
            c_z,
            c_hidden_mul,
        )

        self.tri_att_start = TriangleAttention(
            c_z,
            c_hidden_pair_att,
            no_heads_pair,
            inf=inf,
        )
        self.tri_att_end = TriangleAttention(
            c_z,
            c_hidden_pair_att,
            no_heads_pair,
            inf=inf,
        )

        self.pair_transition = PairTransition(
            c_z,
            transition_n,
        )

        self.ps_dropout_row_layer = DropoutRowwise(pair_dropout)

    def forward(self,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
    ) -> torch.Tensor:
        # DeepMind doesn't mask these transitions in the source, so _mask_trans
        # should be disabled to better approximate the exact activations of
        # the original.
        pair_trans_mask = pair_mask if _mask_trans else None

        tmu_update = self.tri_mul_out(
            z,
            mask=pair_mask,
            inplace_safe=inplace_safe,
            _add_with_inplace=True,
        )
        if (not inplace_safe):
            z = z + self.ps_dropout_row_layer(tmu_update)
        else:
            z = tmu_update

        del tmu_update

        tmu_update = self.tri_mul_in(
            z,
            mask=pair_mask,
            inplace_safe=inplace_safe,
            _add_with_inplace=True,
        )
        if (not inplace_safe):
            z = z + self.ps_dropout_row_layer(tmu_update)
        else:
            z = tmu_update

        del tmu_update

        z = add(z,
                self.ps_dropout_row_layer(
                    self.tri_att_start(
                        z,
                        mask=pair_mask,
                        use_memory_efficient_kernel=False,
                        use_lma=use_lma,
                    )
                ),
                inplace=inplace_safe,
                )

        z = z.transpose(-2, -3)
        if (inplace_safe):
            z = z.contiguous()

        z = add(z,
                self.ps_dropout_row_layer(
                    self.tri_att_end(
                        z,
                        mask=pair_mask.transpose(-1, -2),
                        use_memory_efficient_kernel=False,
                        use_lma=use_lma,
                    )
                ),
                inplace=inplace_safe,
                )

        z = z.transpose(-2, -3)
        if (inplace_safe):
            z = z.contiguous()

        z = add(z,
                self.pair_transition(
                    z, mask=pair_trans_mask,
                ),
                inplace=inplace_safe,
        )

        return z


class EvoformerBlock(nn.Module, ABC):
    def __init__(self,
        c_m: int,
        c_z: int,
        c_hidden_single_att: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_single: int,
        no_heads_pair: int,
        transition_n: int,
        single_dropout: float,
        pair_dropout: float,
        inf: float,
        eps: float,
    ):
        super(EvoformerBlock, self).__init__()

        self.single_att_row = SingleRowAttentionWithPairBias(
            c_m=c_m,
            c_z=c_z,
            c_hidden=c_hidden_single_att,
            no_heads=no_heads_single,
            inf=inf,
        )

        self.single_dropout_layer = DropoutRowwise(single_dropout)

        self.single_transition = SingleRepTransition(
            c_m=c_m,
            n=transition_n,
        )

        self.pair_stack = PairStack(
            c_z=c_z,
            c_hidden_mul=c_hidden_mul,
            c_hidden_pair_att=c_hidden_pair_att,
            no_heads_pair=no_heads_pair,
            transition_n=transition_n,
            pair_dropout=pair_dropout,
            inf=inf,
            eps=eps
        )

    def forward(self,
        m: Optional[torch.Tensor],
        z: Optional[torch.Tensor],
        single_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        single_trans_mask = single_mask if _mask_trans else None

        input_tensors = [m, z]

        m, z = input_tensors

        z = self.pair_stack(
            z=z,
            pair_mask=pair_mask,
            use_lma=use_lma,
            inplace_safe=inplace_safe,
            _mask_trans=_mask_trans,
        )

        m = add(m,
                self.single_dropout_layer(
                    self.single_att_row(
                        m,
                        z=z,
                        mask=single_mask,
                        use_memory_efficient_kernel=False,
                        use_lma=use_lma,
                    )
                ),
                inplace=inplace_safe,
                )

        m = add(m, self.single_transition(m, mask=single_mask), inplace=inplace_safe)

        return m, z


class EvoformerStack(nn.Module):
    """
    Main Evoformer trunk.

    Implements Algorithm 6.
    """

    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden_single_att: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        c_s: int,
        no_heads_single: int,
        no_heads_pair: int,
        no_blocks: int,
        transition_n: int,
        single_dropout: float,
        pair_dropout: float,
        blocks_per_ckpt: int,
        inf: float,
        eps: float,
        clear_cache_between_blocks: bool = False, 
        **kwargs,
    ):
        """
        Args:
            c_m:
                single channel dimension
            c_z:
                Pair channel dimension
            c_hidden_single_att:
                Hidden dimension in single representation attention
            c_hidden_mul:
                Hidden dimension in multiplicative updates
            c_hidden_pair_att:
                Hidden dimension in triangular attention
            c_s:
                Channel dimension of the output "single" embedding
            no_heads_single:
                Number of heads used for single attention
            no_heads_pair:
                Number of heads used for pair attention
            no_blocks:
                Number of Evoformer blocks in the stack
            transition_n:
                Factor by which to multiply c_m to obtain the SingleTransition
                hidden dimension
            single_dropout:
                Dropout rate for single activations
            pair_dropout:
                Dropout used for pair activations
            blocks_per_ckpt:
                Number of Evoformer blocks in each activation checkpoint
            clear_cache_between_blocks:
                Whether to clear CUDA's GPU memory cache between blocks of the
                stack. Slows down each block but can reduce fragmentation
        """
        super(EvoformerStack, self).__init__()

        self.blocks_per_ckpt = blocks_per_ckpt
        self.clear_cache_between_blocks = clear_cache_between_blocks

        self.blocks = nn.ModuleList()

        for _ in range(no_blocks):
            block = EvoformerBlock(
                c_m=c_m,
                c_z=c_z,
                c_hidden_single_att=c_hidden_single_att,
                c_hidden_mul=c_hidden_mul,
                c_hidden_pair_att=c_hidden_pair_att,
                no_heads_single=no_heads_single,
                no_heads_pair=no_heads_pair,
                transition_n=transition_n,
                single_dropout=single_dropout,
                pair_dropout=pair_dropout,
                inf=inf,
                eps=eps,
            )
            self.blocks.append(block)

        self.linear = Linear(c_m, c_s)

    def _prep_blocks(self,
        use_lma: bool,
        single_mask: Optional[torch.Tensor],
        pair_mask: Optional[torch.Tensor],
        inplace_safe: bool,
        _mask_trans: bool,
    ):
        blocks = [
            partial(
                b,
                single_mask=single_mask,
                pair_mask=pair_mask,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
                _mask_trans=_mask_trans,
            )
            for b in self.blocks
        ]

        if self.clear_cache_between_blocks:
            def block_with_cache_clear(block, *args, **kwargs):
                torch.cuda.empty_cache()
                return block(*args, **kwargs)

            blocks = [partial(block_with_cache_clear, b) for b in blocks]

        return blocks

    def forward(self,
        m: torch.Tensor,
        z: torch.Tensor,
        single_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            m:
                [*, N_res, C_m] single embedding
            z:
                [*, N_res, N_res, C_z] pair embedding
            single_mask:
                [*, N_res] single mask
            pair_mask:
                [*, N_res, N_res] pair mask
            use_lma:
                Whether to use low-memory attention during inference.

        Returns:
            m:
                [*, N_res, C_m] single embedding
            z:
                [*, N_res, N_res, C_z] pair embedding
            s:
                [*, N_res, C_s] single embedding after linear layer
        """ 
        blocks = self._prep_blocks(
            use_lma=use_lma,
            single_mask=single_mask,
            pair_mask=pair_mask,
            inplace_safe=inplace_safe,
            _mask_trans=_mask_trans,
        )

        blocks_per_ckpt = self.blocks_per_ckpt
        if(not torch.is_grad_enabled()):
            blocks_per_ckpt = None
        
        m, z = checkpoint_blocks(
            blocks,
            args=(m, z),
            blocks_per_ckpt=blocks_per_ckpt,
        )

        s = self.linear(m)

        return m, z, s
