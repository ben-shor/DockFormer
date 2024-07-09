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
import math
import torch
import torch.nn as nn
import torch.utils.checkpoint
from typing import Optional, List, Tuple

from evodocker.model.primitives import (
    Linear, 
    LayerNorm,
    Attention,
)
from evodocker.utils.tensor_utils import permute_final_dims


class MSAAttention(nn.Module):
    def __init__(
        self,
        c_in,
        c_hidden,
        no_heads,
        pair_bias=False,
        c_z=None,
        inf=1e9,
    ):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Per-head hidden channel dimension
            no_heads:
                Number of attention heads
            pair_bias:
                Whether to use pair embedding bias
            c_z:
                Pair embedding channel dimension. Ignored unless pair_bias
                is true
            inf:
                A large number to be used in computing the attention mask
        """
        super(MSAAttention, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.pair_bias = pair_bias
        self.c_z = c_z
        self.inf = inf

        self.layer_norm_m = LayerNorm(self.c_in)

        self.layer_norm_z = None
        self.linear_z = None
        if self.pair_bias:
            self.layer_norm_z = LayerNorm(self.c_z)
            self.linear_z = Linear(
                self.c_z, self.no_heads, bias=False, init="normal"
            )
        
        self.mha = Attention(
            self.c_in, 
            self.c_in, 
            self.c_in, 
            self.c_hidden, 
            self.no_heads,
        )

    def _prep_inputs(self,
        m: torch.Tensor,
        z: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        inplace_safe: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
        if mask is None:
            # [*, N_res]
            mask = m.new_ones(m.shape[:-1])

        # [*, N_seq, 1, 1, N_res]
        mask_bias = (self.inf * (mask - 1))[..., :, None, :]

        if (self.pair_bias and 
            z is not None and                       # For the 
            self.layer_norm_z is not None and       # benefit of
            self.linear_z is not None               # TorchScript
        ):
            chunks = []

            for i in range(0, z.shape[-3], 256):
                z_chunk = z[..., i: i + 256, :, :]

                # [*, N_res, N_res, C_z]
                z_chunk = self.layer_norm_z(z_chunk)
            
                # [*, N_res, N_res, no_heads]
                z_chunk = self.linear_z(z_chunk)

                chunks.append(z_chunk)
            
            z = torch.cat(chunks, dim=-3)
            
            # [*, no_heads, N_res, N_res]
            z = permute_final_dims(z, (2, 0, 1))

        return m, mask_bias, z

    def forward(self, 
        m: torch.Tensor, 
        z: Optional[torch.Tensor] = None, 
        mask: Optional[torch.Tensor] = None, 
        use_memory_efficient_kernel: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            z:
                [*, N_res, N_res, C_z] pair embedding. Required only if
                pair_bias is True
            mask:
                [*, N_seq, N_res] MSA mask
        """
        m, mask_bias, z = self._prep_inputs(
            m, z, mask, inplace_safe=inplace_safe
        )

        biases = [mask_bias]
        if(z is not None):
            biases.append(z)

        m = self.layer_norm_m(m)
        m = self.mha(
            q_x=m,
            kv_x=m,
            biases=biases,
            use_memory_efficient_kernel=use_memory_efficient_kernel,
            use_lma=use_lma,
        )

        return m


class MSARowAttentionWithPairBias(MSAAttention):
    """
    Implements Algorithm 7.
    """

    def __init__(self, c_m, c_z, c_hidden, no_heads, inf=1e9):
        """
        Args:
            c_m:
                Input channel dimension
            c_z:
                Pair embedding channel dimension
            c_hidden:
                Per-head hidden channel dimension
            no_heads:
                Number of attention heads
            inf:
                Large number used to construct attention masks
        """
        super(MSARowAttentionWithPairBias, self).__init__(
            c_m,
            c_hidden,
            no_heads,
            pair_bias=True,
            c_z=c_z,
            inf=inf,
        )
