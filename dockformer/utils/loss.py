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
import time

import ml_collections
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from dockformer.utils import residue_constants
from dockformer.utils.feats import pseudo_beta_fn
from dockformer.utils.rigid_utils import Rotation, Rigid
from dockformer.utils.geometry.vector import Vec3Array, euclidean_distance
from dockformer.utils.tensor_utils import (
    tree_map,
    masked_mean,
    permute_final_dims,
)
import logging
from dockformer.utils.tensor_utils import tensor_tree_map

logger = logging.getLogger(__name__)


def softmax_cross_entropy(logits, labels):
    loss = -1 * torch.sum(
        labels * torch.nn.functional.log_softmax(logits, dim=-1),
        dim=-1,
    )
    return loss


def sigmoid_cross_entropy(logits, labels):
    logits_dtype = logits.dtype
    try:
        logits = logits.double()
        labels = labels.double()
    except:
        logits = logits.to(dtype=torch.float32)
        labels = labels.to(dtype=torch.float32)

    log_p = torch.nn.functional.logsigmoid(logits)
    # log_p = torch.log(torch.sigmoid(logits))
    log_not_p = torch.nn.functional.logsigmoid(-1 * logits)
    # log_not_p = torch.log(torch.sigmoid(-logits))
    loss = (-1. * labels) * log_p - (1. - labels) * log_not_p
    loss = loss.to(dtype=logits_dtype)
    return loss


def torsion_angle_loss(
    a,  # [*, N, 7, 2]
    a_gt,  # [*, N, 7, 2]
    a_alt_gt,  # [*, N, 7, 2]
):
    # [*, N, 7]
    norm = torch.norm(a, dim=-1)

    # [*, N, 7, 2]
    a = a / norm.unsqueeze(-1)

    # [*, N, 7]
    diff_norm_gt = torch.norm(a - a_gt, dim=-1)
    diff_norm_alt_gt = torch.norm(a - a_alt_gt, dim=-1)
    min_diff = torch.minimum(diff_norm_gt ** 2, diff_norm_alt_gt ** 2)

    # [*]
    l_torsion = torch.mean(min_diff, dim=(-1, -2))
    l_angle_norm = torch.mean(torch.abs(norm - 1), dim=(-1, -2))

    an_weight = 0.02
    return l_torsion + an_weight * l_angle_norm


def compute_fape(
    pred_frames: Rigid,
    target_frames: Rigid,
    frames_mask: torch.Tensor,
    pred_positions: torch.Tensor,
    target_positions: torch.Tensor,
    positions_mask: torch.Tensor,
    length_scale: float,
    pair_mask: Optional[torch.Tensor] = None,
    l1_clamp_distance: Optional[float] = None,
    eps=1e-8,
) -> torch.Tensor:
    """
        Computes FAPE loss.

        Args:
            pred_frames:
                [*, N_frames] Rigid object of predicted frames
            target_frames:
                [*, N_frames] Rigid object of ground truth frames
            frames_mask:
                [*, N_frames] binary mask for the frames
            pred_positions:
                [*, N_pts, 3] predicted atom positions
            target_positions:
                [*, N_pts, 3] ground truth positions
            positions_mask:
                [*, N_pts] positions mask
            length_scale:
                Length scale by which the loss is divided
            pair_mask:
                [*,  N_frames, N_pts] mask to use for
                separating intra- from inter-chain losses.
            l1_clamp_distance:
                Cutoff above which distance errors are disregarded
            eps:
                Small value used to regularize denominators
        Returns:
            [*] loss tensor
    """
    # [*, N_frames, N_pts, 3]
    local_pred_pos = pred_frames.invert()[..., None].apply(
        pred_positions[..., None, :, :],
    )
    local_target_pos = target_frames.invert()[..., None].apply(
        target_positions[..., None, :, :],
    )

    error_dist = torch.sqrt(
        torch.sum((local_pred_pos - local_target_pos) ** 2, dim=-1) + eps
    )

    if l1_clamp_distance is not None:
        error_dist = torch.clamp(error_dist, min=0, max=l1_clamp_distance)

    normed_error = error_dist / length_scale
    normed_error = normed_error * frames_mask[..., None]
    normed_error = normed_error * positions_mask[..., None, :]

    if pair_mask is not None:
        normed_error = normed_error * pair_mask
        normed_error = torch.sum(normed_error, dim=(-1, -2))

        mask = frames_mask[..., None] * positions_mask[..., None, :] * pair_mask
        norm_factor = torch.sum(mask, dim=(-2, -1))

        normed_error = normed_error / (eps + norm_factor)
    else:
        # FP16-friendly averaging. Roughly equivalent to:
        #
        # norm_factor = (
        #     torch.sum(frames_mask, dim=-1) *
        #     torch.sum(positions_mask, dim=-1)
        # )
        # normed_error = torch.sum(normed_error, dim=(-1, -2)) / (eps + norm_factor)
        #
        # ("roughly" because eps is necessarily duplicated in the latter)
        normed_error = torch.sum(normed_error, dim=-1)
        normed_error = (
            normed_error / (eps + torch.sum(frames_mask, dim=-1))[..., None]
        )
        normed_error = torch.sum(normed_error, dim=-1)
        normed_error = normed_error / (eps + torch.sum(positions_mask, dim=-1))

    return normed_error


def backbone_loss(
    backbone_rigid_tensor: torch.Tensor,
    backbone_rigid_mask: torch.Tensor,
    traj: torch.Tensor,
    pair_mask: Optional[torch.Tensor] = None,
    use_clamped_fape: Optional[torch.Tensor] = None,
    clamp_distance: float = 10.0,
    loss_unit_distance: float = 10.0,
    eps: float = 1e-4,
    **kwargs,
) -> torch.Tensor:
    ### need to check if the traj belongs to 4*4 matrix or a tensor_7
    if traj.shape[-1] == 7:
        pred_aff = Rigid.from_tensor_7(traj)
    elif traj.shape[-1] == 4:
        pred_aff = Rigid.from_tensor_4x4(traj)

    pred_aff = Rigid(
        Rotation(rot_mats=pred_aff.get_rots().get_rot_mats(), quats=None),
        pred_aff.get_trans(),
    )

    # DISCREPANCY: DeepMind somehow gets a hold of a tensor_7 version of
    # backbone tensor, normalizes it, and then turns it back to a rotation
    # matrix. To avoid a potentially numerically unstable rotation matrix
    # to quaternion conversion, we just use the original rotation matrix
    # outright. This one hasn't been composed a bunch of times, though, so
    # it might be fine.
    gt_aff = Rigid.from_tensor_4x4(backbone_rigid_tensor)

    fape_loss = compute_fape(
        pred_aff,
        gt_aff[None],
        backbone_rigid_mask[None],
        pred_aff.get_trans(),
        gt_aff[None].get_trans(),
        backbone_rigid_mask[None],
        pair_mask=pair_mask,
        l1_clamp_distance=clamp_distance,
        length_scale=loss_unit_distance,
        eps=eps,
    )
    if use_clamped_fape is not None:
        unclamped_fape_loss = compute_fape(
            pred_aff,
            gt_aff[None],
            backbone_rigid_mask[None],
            pred_aff.get_trans(),
            gt_aff[None].get_trans(),
            backbone_rigid_mask[None],
            pair_mask=pair_mask,
            l1_clamp_distance=None,
            length_scale=loss_unit_distance,
            eps=eps,
        )

        fape_loss = fape_loss * use_clamped_fape + unclamped_fape_loss * (
            1 - use_clamped_fape
        )

    # Average over the batch dimension
    fape_loss = torch.mean(fape_loss)

    return fape_loss


def sidechain_loss(
    pred_sidechain_frames: torch.Tensor,
    pred_sidechain_atom_pos: torch.Tensor,
    rigidgroups_gt_frames: torch.Tensor,
    rigidgroups_alt_gt_frames: torch.Tensor,
    rigidgroups_gt_exists: torch.Tensor,
    renamed_atom14_gt_positions: torch.Tensor,
    renamed_atom14_gt_exists: torch.Tensor,
    alt_naming_is_better: torch.Tensor,
    ligand_mask: torch.Tensor,
    clamp_distance: float = 10.0,
    length_scale: float = 10.0,
    eps: float = 1e-4,
    only_include_ligand_atoms: bool = False,
    **kwargs,
) -> torch.Tensor:
    renamed_gt_frames = (
                            1.0 - alt_naming_is_better[..., None, None, None]
                        ) * rigidgroups_gt_frames + alt_naming_is_better[
                            ..., None, None, None
                        ] * rigidgroups_alt_gt_frames

    # Steamroll the inputs
    pred_sidechain_frames = pred_sidechain_frames[-1]  # get only the last layer of the strcuture module
    batch_dims = pred_sidechain_frames.shape[:-4]
    pred_sidechain_frames = pred_sidechain_frames.view(*batch_dims, -1, 4, 4)
    pred_sidechain_frames = Rigid.from_tensor_4x4(pred_sidechain_frames)
    renamed_gt_frames = renamed_gt_frames.view(*batch_dims, -1, 4, 4)
    renamed_gt_frames = Rigid.from_tensor_4x4(renamed_gt_frames)
    rigidgroups_gt_exists = rigidgroups_gt_exists.reshape(*batch_dims, -1)
    pred_sidechain_atom_pos = pred_sidechain_atom_pos[-1]
    pred_sidechain_atom_pos = pred_sidechain_atom_pos.view(*batch_dims, -1, 3)
    renamed_atom14_gt_positions = renamed_atom14_gt_positions.view(
        *batch_dims, -1, 3
    )
    renamed_atom14_gt_exists = renamed_atom14_gt_exists.view(*batch_dims, -1)

    atom_mask_to_apply = renamed_atom14_gt_exists
    if only_include_ligand_atoms:
        ligand_atom14_mask = torch.repeat_interleave(ligand_mask, 14, dim=-1)
        atom_mask_to_apply = atom_mask_to_apply * ligand_atom14_mask

    fape = compute_fape(
        pred_sidechain_frames,
        renamed_gt_frames,
        rigidgroups_gt_exists,
        pred_sidechain_atom_pos,
        renamed_atom14_gt_positions,
        atom_mask_to_apply,
        pair_mask=None,
        l1_clamp_distance=clamp_distance,
        length_scale=length_scale,
        eps=eps,
    )

    return fape


def fape_bb(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    config: ml_collections.ConfigDict,
) -> torch.Tensor:
    traj = out["sm"]["frames"]
    bb_loss = backbone_loss(
        traj=traj,
        **{**batch, **config},
    )
    loss = torch.mean(bb_loss)
    return loss


def fape_sidechain(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    config: ml_collections.ConfigDict,
) -> torch.Tensor:
    sc_loss = sidechain_loss(
        out["sm"]["sidechain_frames"],
        out["sm"]["positions"],
        **{**batch, **config},
    )
    loss = torch.mean(sc_loss)
    return loss


def fape_interface(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    config: ml_collections.ConfigDict,
) -> torch.Tensor:
    sc_loss = sidechain_loss(
        out["sm"]["sidechain_frames"],
        out["sm"]["positions"],
        only_include_ligand_atoms=True,
        **{**batch, **config},
    )
    loss = torch.mean(sc_loss)
    return loss


def supervised_chi_loss(
    angles_sin_cos: torch.Tensor,
    unnormalized_angles_sin_cos: torch.Tensor,
    aatype: torch.Tensor,
    protein_mask: torch.Tensor,
    chi_mask: torch.Tensor,
    chi_angles_sin_cos: torch.Tensor,
    chi_weight: float,
    angle_norm_weight: float,
    eps=1e-6,
    **kwargs,
) -> torch.Tensor:
    """
        Implements Algorithm 27 (torsionAngleLoss)

        Args:
            angles_sin_cos:
                [*, N, 7, 2] predicted angles
            unnormalized_angles_sin_cos:
                The same angles, but unnormalized
            aatype:
                [*, N] residue indices
            protein_mask:
                [*, N] protein mask
            chi_mask:
                [*, N, 7] angle mask
            chi_angles_sin_cos:
                [*, N, 7, 2] ground truth angles
            chi_weight:
                Weight for the angle component of the loss
            angle_norm_weight:
                Weight for the normalization component of the loss
        Returns:
            [*] loss tensor
    """
    pred_angles = angles_sin_cos[..., 3:, :]
    residue_type_one_hot = torch.nn.functional.one_hot(
        aatype,
        residue_constants.restype_num + 1,
    )
    chi_pi_periodic = torch.einsum(
        "...ij,jk->ik",
        residue_type_one_hot.type(angles_sin_cos.dtype),
        angles_sin_cos.new_tensor(residue_constants.chi_pi_periodic),
    )

    true_chi = chi_angles_sin_cos[None]

    shifted_mask = (1 - 2 * chi_pi_periodic).unsqueeze(-1)
    true_chi_shifted = shifted_mask * true_chi
    sq_chi_error = torch.sum((true_chi - pred_angles) ** 2, dim=-1)
    sq_chi_error_shifted = torch.sum(
        (true_chi_shifted - pred_angles) ** 2, dim=-1
    )
    sq_chi_error = torch.minimum(sq_chi_error, sq_chi_error_shifted)

    # The ol' switcheroo
    sq_chi_error = sq_chi_error.permute(
        *range(len(sq_chi_error.shape))[1:-2], 0, -2, -1
    )

    sq_chi_loss = masked_mean(
        chi_mask[..., None, :, :], sq_chi_error, dim=(-1, -2, -3)
    )

    loss = chi_weight * sq_chi_loss

    angle_norm = torch.sqrt(
        torch.sum(unnormalized_angles_sin_cos ** 2, dim=-1) + eps
    )
    norm_error = torch.abs(angle_norm - 1.0)
    norm_error = norm_error.permute(
        *range(len(norm_error.shape))[1:-2], 0, -2, -1
    )
    angle_norm_loss = masked_mean(
        protein_mask[..., None, :, None], norm_error, dim=(-1, -2, -3)
    )

    loss = loss + angle_norm_weight * angle_norm_loss

    # Average over the batch dimension
    loss = torch.mean(loss)

    return loss


def compute_plddt(logits: torch.Tensor) -> torch.Tensor:
    num_bins = logits.shape[-1]
    bin_width = 1.0 / num_bins
    bounds = torch.arange(
        start=0.5 * bin_width, end=1.0, step=bin_width, device=logits.device
    )
    probs = torch.nn.functional.softmax(logits, dim=-1)
    pred_lddt_ca = torch.sum(
        probs * bounds.view(*((1,) * len(probs.shape[:-1])), *bounds.shape),
        dim=-1,
    )
    return pred_lddt_ca * 100


def lddt(
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    cutoff: float = 15.0,
    eps: float = 1e-10,
    per_residue: bool = True,
) -> torch.Tensor:
    n = all_atom_mask.shape[-2]
    dmat_true = torch.sqrt(
        eps
        + torch.sum(
            (
                all_atom_positions[..., None, :]
                - all_atom_positions[..., None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    dmat_pred = torch.sqrt(
        eps
        + torch.sum(
            (
                all_atom_pred_pos[..., None, :]
                - all_atom_pred_pos[..., None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )
    dists_to_score = (
        (dmat_true < cutoff)
        * all_atom_mask
        * permute_final_dims(all_atom_mask, (1, 0))
        * (1.0 - torch.eye(n, device=all_atom_mask.device))
    )

    dist_l1 = torch.abs(dmat_true - dmat_pred)

    score = (
        (dist_l1 < 0.5).type(dist_l1.dtype)
        + (dist_l1 < 1.0).type(dist_l1.dtype)
        + (dist_l1 < 2.0).type(dist_l1.dtype)
        + (dist_l1 < 4.0).type(dist_l1.dtype)
    )
    score = score * 0.25

    dims = (-1,) if per_residue else (-2, -1)
    norm = 1.0 / (eps + torch.sum(dists_to_score, dim=dims))
    score = norm * (eps + torch.sum(dists_to_score * score, dim=dims))

    return score


def lddt_ca(
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    cutoff: float = 15.0,
    eps: float = 1e-10,
    per_residue: bool = True,
) -> torch.Tensor:
    ca_pos = residue_constants.atom_order["CA"]
    all_atom_pred_pos = all_atom_pred_pos[..., ca_pos, :]
    all_atom_positions = all_atom_positions[..., ca_pos, :]
    all_atom_mask = all_atom_mask[..., ca_pos: (ca_pos + 1)]  # keep dim

    return lddt(
        all_atom_pred_pos,
        all_atom_positions,
        all_atom_mask,
        cutoff=cutoff,
        eps=eps,
        per_residue=per_residue,
    )


def lddt_loss(
    logits: torch.Tensor,
    all_atom_pred_pos: torch.Tensor,
    atom37_gt_positions: torch.Tensor,
    atom37_atom_exists_in_gt: torch.Tensor,
    resolution: torch.Tensor,
    cutoff: float = 15.0,
    no_bins: int = 50,
    min_resolution: float = 0.1,
    max_resolution: float = 3.0,
    eps: float = 1e-10,
    **kwargs,
) -> torch.Tensor:
    # remove ligand
    logits = logits[:, :atom37_atom_exists_in_gt.shape[1], :]

    ca_pos = residue_constants.atom_order["CA"]
    all_atom_pred_pos = all_atom_pred_pos[..., ca_pos, :]
    atom37_gt_positions = atom37_gt_positions[..., ca_pos, :]
    atom37_atom_exists_in_gt = atom37_atom_exists_in_gt[..., ca_pos: (ca_pos + 1)]  # keep dim

    score = lddt(
        all_atom_pred_pos,
        atom37_gt_positions,
        atom37_atom_exists_in_gt,
        cutoff=cutoff,
        eps=eps
    )

    # TODO: Remove after initial pipeline testing
    score = torch.nan_to_num(score, nan=torch.nanmean(score))
    score[score < 0] = 0

    score = score.detach()
    bin_index = torch.floor(score * no_bins).long()
    bin_index = torch.clamp(bin_index, max=(no_bins - 1))
    lddt_ca_one_hot = torch.nn.functional.one_hot(
        bin_index, num_classes=no_bins
    )

    errors = softmax_cross_entropy(logits, lddt_ca_one_hot)
    atom37_atom_exists_in_gt = atom37_atom_exists_in_gt.squeeze(-1)
    loss = torch.sum(errors * atom37_atom_exists_in_gt, dim=-1) / (
        eps + torch.sum(atom37_atom_exists_in_gt, dim=-1)
    )

    loss = loss * (
        (resolution >= min_resolution) & (resolution <= max_resolution)
    )

    # Average over the batch dimension
    loss = torch.mean(loss)

    return loss


def distogram_loss(
    logits,
    gt_pseudo_beta_with_lig,
    gt_pseudo_beta_with_lig_mask,
    min_bin=2.3125,
    max_bin=21.6875,
    no_bins=64,
    eps=1e-6,
    **kwargs,
):
    boundaries = torch.linspace(
        min_bin,
        max_bin,
        no_bins - 1,
        device=logits.device,
    )
    boundaries = boundaries ** 2

    dists = torch.sum(
        (gt_pseudo_beta_with_lig[..., None, :] - gt_pseudo_beta_with_lig[..., None, :, :]) ** 2,
        dim=-1,
        keepdims=True,
    )

    true_bins = torch.sum(dists > boundaries, dim=-1)
    errors = softmax_cross_entropy(
        logits,
        torch.nn.functional.one_hot(true_bins, no_bins),
    )

    square_mask = gt_pseudo_beta_with_lig_mask[..., None] * gt_pseudo_beta_with_lig_mask[..., None, :]

    # FP16-friendly sum. Equivalent to:
    # mean = (torch.sum(errors * square_mask, dim=(-1, -2)) /
    #         (eps + torch.sum(square_mask, dim=(-1, -2))))
    denom = eps + torch.sum(square_mask, dim=(-1, -2))
    mean = errors * square_mask
    mean = torch.sum(mean, dim=-1)
    mean = mean / denom[..., None]
    mean = torch.sum(mean, dim=-1)

    # Average over the batch dimensions
    mean = torch.mean(mean)

    return mean


def inter_contact_loss(
    logits: torch.Tensor,
    gt_inter_contacts: torch.Tensor,
    inter_pair_mask: torch.Tensor,
    pos_class_weight: float = 200.0,
    contact_distance: float = 5.0,
    **kwargs,
):
    logits = logits.squeeze(-1)
    bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, gt_inter_contacts, reduction='none',
                                                                    pos_weight=logits.new_tensor([pos_class_weight]))
    masked_loss = bce_loss * inter_pair_mask
    final_loss = masked_loss.sum() / inter_pair_mask.sum()

    return final_loss


def affinity_loss(
    logits,
    affinity,
    affinity_loss_factor,
    min_bin=0,
    max_bin=15,
    no_bins=32,
    **kwargs,
):
    boundaries = torch.linspace(
        min_bin,
        max_bin,
        no_bins - 1,
        device=logits.device,
    )

    true_bins = torch.sum(affinity > boundaries, dim=-1)
    errors = softmax_cross_entropy(
        logits,
        torch.nn.functional.one_hot(true_bins, no_bins),
    )

    # print("errors dim", errors.shape, affinity_loss_factor.shape, errors)
    after_factor = errors * affinity_loss_factor.squeeze()
    if affinity_loss_factor.sum() > 0.1:
        mean_val = after_factor.sum() / affinity_loss_factor.sum()
    else:
        # If no affinity in batch - get a very small loss. the factor also makes the loss small
        mean_val = after_factor.sum() * 1e-3
    # print("after factor", after_factor.shape, after_factor, affinity_loss_factor.sum(), mean_val)
    return mean_val

def affinity_loss_reg(
    logits,
    affinity,
    affinity_loss_factor,
    **kwargs,
):
    # apply mse loss
    errors = torch.nn.functional.mse_loss(logits, affinity, reduction='none')

    # print("errors dim", errors.shape, affinity_loss_factor.shape, errors)
    after_factor = errors * affinity_loss_factor.squeeze()
    if affinity_loss_factor.sum() > 0.1:
        mean_val = after_factor.sum() / affinity_loss_factor.sum()
    else:
        # If no affinity in batch - get a very small loss. the factor also makes the loss small
        mean_val = after_factor.sum() * 1e-3
    # print("after factor", after_factor.shape, after_factor, affinity_loss_factor.sum(), mean_val)
    return mean_val


def positions_inter_distogram_loss(
    out,
    aatype: torch.Tensor,
    inter_pair_mask: torch.Tensor,
    gt_pseudo_beta_with_lig: torch.Tensor,
    max_dist=20.,
    length_scale=10.,
    eps: float = 1e-10,
    **kwargs,
):

    predicted_atoms = pseudo_beta_fn(aatype, out["final_atom_positions"], None)
    pred_dists = torch.sum(
        (predicted_atoms[..., None, :] - predicted_atoms[..., None, :, :]) ** 2,
        dim=-1,
        keepdims=True,
    )

    gt_dists = torch.sum(
        (gt_pseudo_beta_with_lig[..., None, :] - gt_pseudo_beta_with_lig[..., None, :, :]) ** 2,
        dim=-1,
        keepdims=True,
    )

    pred_dists = pred_dists.clamp(max=max_dist ** 2)
    gt_dists = gt_dists.clamp(max=max_dist ** 2)

    dists_diff = torch.abs(pred_dists - gt_dists) / (length_scale ** 2)
    dists_diff = dists_diff * inter_pair_mask.unsqueeze(-1)

    dists_diff_sum_per_batch = torch.sum(torch.sqrt(eps + dists_diff), dim=(-1, -2, -3))
    mask_size_per_batch = torch.sum(inter_pair_mask, dim=(-1, -2))
    inter_loss = torch.mean(dists_diff_sum_per_batch / (eps + mask_size_per_batch))

    return inter_loss


def positions_intra_ligand_distogram_loss(
    out,
    aatype: torch.Tensor,
    ligand_mask: torch.Tensor,
    gt_pseudo_beta_with_lig: torch.Tensor,
    max_dist=20.,
    length_scale=4.,  # similar to RosettaFoldAA
    eps=1e-10,
    **kwargs,
):
    intra_ligand_pair_mask = ligand_mask[..., None] * ligand_mask[..., None, :]
    predicted_atoms = pseudo_beta_fn(aatype, out["final_atom_positions"], None)
    pred_dists = torch.sum(
        (predicted_atoms[..., None, :] - predicted_atoms[..., None, :, :]) ** 2,
        dim=-1,
        keepdims=True,
    )

    gt_dists = torch.sum(
        (gt_pseudo_beta_with_lig[..., None, :] - gt_pseudo_beta_with_lig[..., None, :, :]) ** 2,
        dim=-1,
        keepdims=True,
    )

    pred_dists = torch.sqrt(eps + pred_dists.clamp(max=max_dist ** 2)) / length_scale
    gt_dists = torch.sqrt(eps + gt_dists.clamp(max=max_dist ** 2)) / length_scale

    # Apply L2 loss
    dists_diff = (pred_dists - gt_dists) ** 2

    dists_diff = dists_diff * intra_ligand_pair_mask.unsqueeze(-1)

    dists_diff_sum_per_batch = torch.sum(dists_diff, dim=(-1, -2, -3))
    mask_size_per_batch = torch.sum(intra_ligand_pair_mask, dim=(-1, -2))
    intra_ligand_loss = torch.mean(dists_diff_sum_per_batch / (eps + mask_size_per_batch))

    return intra_ligand_loss


def _calculate_bin_centers(boundaries: torch.Tensor):
    step = boundaries[1] - boundaries[0]
    bin_centers = boundaries + step / 2
    bin_centers = torch.cat(
        [bin_centers, (bin_centers[-1] + step).unsqueeze(-1)], dim=0
    )
    return bin_centers


def _calculate_expected_aligned_error(
    alignment_confidence_breaks: torch.Tensor,
    aligned_distance_error_probs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bin_centers = _calculate_bin_centers(alignment_confidence_breaks)
    return (
        torch.sum(aligned_distance_error_probs * bin_centers, dim=-1),
        bin_centers[-1],
    )


def compute_predicted_aligned_error(
    logits: torch.Tensor,
    max_bin: int = 31,
    no_bins: int = 64,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """Computes aligned confidence metrics from logits.

    Args:
      logits: [*, num_res, num_res, num_bins] the logits output from
        PredictedAlignedErrorHead.
      max_bin: Maximum bin value
      no_bins: Number of bins
    Returns:
      aligned_confidence_probs: [*, num_res, num_res, num_bins] the predicted
        aligned error probabilities over bins for each residue pair.
      predicted_aligned_error: [*, num_res, num_res] the expected aligned distance
        error for each pair of residues.
      max_predicted_aligned_error: [*] the maximum predicted error possible.
    """
    boundaries = torch.linspace(
        0, max_bin, steps=(no_bins - 1), device=logits.device
    )

    aligned_confidence_probs = torch.nn.functional.softmax(logits, dim=-1)
    (
        predicted_aligned_error,
        max_predicted_aligned_error,
    ) = _calculate_expected_aligned_error(
        alignment_confidence_breaks=boundaries,
        aligned_distance_error_probs=aligned_confidence_probs,
    )

    return {
        "aligned_confidence_probs": aligned_confidence_probs,
        "predicted_aligned_error": predicted_aligned_error,
        "max_predicted_aligned_error": max_predicted_aligned_error,
    }


def compute_tm(
    logits: torch.Tensor,
    residue_weights: Optional[torch.Tensor] = None,
    asym_id: Optional[torch.Tensor] = None,
    interface: bool = False,
    max_bin: int = 31,
    no_bins: int = 64,
    eps: float = 1e-8,
    **kwargs,
) -> torch.Tensor:
    if residue_weights is None:
        residue_weights = logits.new_ones(logits.shape[-2])

    boundaries = torch.linspace(
        0, max_bin, steps=(no_bins - 1), device=logits.device
    )

    bin_centers = _calculate_bin_centers(boundaries)
    clipped_n = max(torch.sum(residue_weights), 19)

    d0 = 1.24 * (clipped_n - 15) ** (1.0 / 3) - 1.8

    probs = torch.nn.functional.softmax(logits, dim=-1)

    tm_per_bin = 1.0 / (1 + (bin_centers ** 2) / (d0 ** 2))
    predicted_tm_term = torch.sum(probs * tm_per_bin, dim=-1)

    n = residue_weights.shape[-1]
    pair_mask = residue_weights.new_ones((n, n), dtype=torch.int32)
    if interface and (asym_id is not None):
        if len(asym_id.shape) > 1:
            assert len(asym_id.shape) <= 2
            batch_size = asym_id.shape[0]
            pair_mask = residue_weights.new_ones((batch_size, n, n), dtype=torch.int32)
        pair_mask *= (asym_id[..., None] != asym_id[..., None, :]).to(dtype=pair_mask.dtype)

    predicted_tm_term *= pair_mask

    pair_residue_weights = pair_mask * (
        residue_weights[..., None, :] * residue_weights[..., :, None]
    )
    denom = eps + torch.sum(pair_residue_weights, dim=-1, keepdims=True)
    normed_residue_mask = pair_residue_weights / denom
    per_alignment = torch.sum(predicted_tm_term * normed_residue_mask, dim=-1)

    weighted = per_alignment * residue_weights

    argmax = (weighted == torch.max(weighted)).nonzero()[0]
    return per_alignment[tuple(argmax)]


def compute_renamed_ground_truth(
    batch: Dict[str, torch.Tensor],
    atom14_pred_positions: torch.Tensor,
    eps=1e-10,
) -> Dict[str, torch.Tensor]:
    """
    Find optimal renaming of ground truth based on the predicted positions.

    Alg. 26 "renameSymmetricGroundTruthAtoms"

    This renamed ground truth is then used for all losses,
    such that each loss moves the atoms in the same direction.

    Args:
      batch: Dictionary containing:
        * atom14_gt_positions: Ground truth positions.
        * atom14_alt_gt_positions: Ground truth positions with renaming swaps.
        * atom14_atom_is_ambiguous: 1.0 for atoms that are affected by
            renaming swaps.
        * atom14_gt_exists: Mask for which atoms exist in ground truth.
        * atom14_alt_gt_exists: Mask for which atoms exist in ground truth
            after renaming.
        * atom14_atom_exists: Mask for whether each atom is part of the given
            amino acid type.
      atom14_pred_positions: Array of atom positions in global frame with shape
    Returns:
      Dictionary containing:
        alt_naming_is_better: Array with 1.0 where alternative swap is better.
        renamed_atom14_gt_positions: Array of optimal ground truth positions
          after renaming swaps are performed.
        renamed_atom14_gt_exists: Mask after renaming swap is performed.
    """

    pred_dists = torch.sqrt(
        eps
        + torch.sum(
            (
                atom14_pred_positions[..., None, :, None, :]
                - atom14_pred_positions[..., None, :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    atom14_gt_positions = batch["atom14_gt_positions"]
    gt_dists = torch.sqrt(
        eps
        + torch.sum(
            (
                atom14_gt_positions[..., None, :, None, :]
                - atom14_gt_positions[..., None, :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    atom14_alt_gt_positions = batch["atom14_alt_gt_positions"]
    alt_gt_dists = torch.sqrt(
        eps
        + torch.sum(
            (
                atom14_alt_gt_positions[..., None, :, None, :]
                - atom14_alt_gt_positions[..., None, :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    lddt = torch.sqrt(eps + (pred_dists - gt_dists) ** 2)
    alt_lddt = torch.sqrt(eps + (pred_dists - alt_gt_dists) ** 2)

    atom14_gt_exists = batch["atom14_atom_exists_in_gt"]
    atom14_atom_is_ambiguous = batch["atom14_atom_is_ambiguous"]
    mask = (
        atom14_gt_exists[..., None, :, None]
        * atom14_atom_is_ambiguous[..., None, :, None]
        * atom14_gt_exists[..., None, :, None, :]
        * (1.0 - atom14_atom_is_ambiguous[..., None, :, None, :])
    )

    per_res_lddt = torch.sum(mask * lddt, dim=(-1, -2, -3))
    alt_per_res_lddt = torch.sum(mask * alt_lddt, dim=(-1, -2, -3))

    fp_type = atom14_pred_positions.dtype
    alt_naming_is_better = (alt_per_res_lddt < per_res_lddt).type(fp_type)

    renamed_atom14_gt_positions = (
                                      1.0 - alt_naming_is_better[..., None, None]
                                  ) * atom14_gt_positions + alt_naming_is_better[
                                      ..., None, None
                                  ] * atom14_alt_gt_positions

    renamed_atom14_gt_mask = (
                                 1.0 - alt_naming_is_better[..., None]
                             ) * atom14_gt_exists + alt_naming_is_better[..., None] * batch[
                                 "atom14_alt_gt_exists"
                             ]

    return {
        "alt_naming_is_better": alt_naming_is_better,
        "renamed_atom14_gt_positions": renamed_atom14_gt_positions,
        "renamed_atom14_gt_exists": renamed_atom14_gt_mask,
    }


def binding_site_loss(
    logits: torch.Tensor,
    binding_site_mask: torch.Tensor,
    protein_mask: torch.Tensor,
    pos_class_weight: float,
    **kwargs,
) -> torch.Tensor:
    logits = logits.squeeze(-1)
    bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, binding_site_mask, reduction='none',
                                                                    pos_weight=logits.new_tensor([pos_class_weight]))
    masked_loss = bce_loss * protein_mask
    final_loss = masked_loss.sum() / protein_mask.sum()

    return final_loss


def chain_center_of_mass_loss(
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    asym_id: torch.Tensor,
    clamp_distance: float = -4.0,
    weight: float = 0.05,
    eps: float = 1e-10, **kwargs
) -> torch.Tensor:
    """
    Computes chain centre-of-mass loss. Implements section 2.5, eqn 1 in the Multimer paper.

    Args:
        all_atom_pred_pos:
            [*, N_pts, 37, 3] All-atom predicted atom positions
        all_atom_positions:
            [*, N_pts, 37, 3] Ground truth all-atom positions
        all_atom_mask:
            [*, N_pts, 37] All-atom positions mask
        asym_id:
            [*, N_pts] Chain asym IDs
        clamp_distance:
            Cutoff above which distance errors are disregarded
        weight:
            Weight for loss
        eps:
            Small value used to regularize denominators
    Returns:
        [*] loss tensor
    """
    ca_pos = residue_constants.atom_order["CA"]
    all_atom_pred_pos = all_atom_pred_pos[..., ca_pos, :]
    all_atom_positions = all_atom_positions[..., ca_pos, :]
    all_atom_mask = all_atom_mask[..., ca_pos: (ca_pos + 1)]  # keep dim

    one_hot = torch.nn.functional.one_hot(asym_id.long()).to(dtype=all_atom_mask.dtype)
    one_hot = one_hot * all_atom_mask
    chain_pos_mask = one_hot.transpose(-2, -1)
    chain_exists = torch.any(chain_pos_mask, dim=-1).to(dtype=all_atom_positions.dtype)

    def get_chain_center_of_mass(pos):
        center_sum = (chain_pos_mask[..., None] * pos[..., None, :, :]).sum(dim=-2)
        centers = center_sum / (torch.sum(chain_pos_mask, dim=-1, keepdim=True) + eps)
        return Vec3Array.from_array(centers)

    pred_centers = get_chain_center_of_mass(all_atom_pred_pos)  # [B, NC, 3]
    true_centers = get_chain_center_of_mass(all_atom_positions)  # [B, NC, 3]

    pred_dists = euclidean_distance(pred_centers[..., None, :], pred_centers[..., :, None], epsilon=eps)
    true_dists = euclidean_distance(true_centers[..., None, :], true_centers[..., :, None], epsilon=eps)
    losses = torch.clamp((weight * (pred_dists - true_dists - clamp_distance)), max=0) ** 2
    loss_mask = chain_exists[..., :, None] * chain_exists[..., None, :]

    loss = masked_mean(loss_mask, losses, dim=(-1, -2))
    return loss


class AlphaFoldLoss(nn.Module):
    """Aggregation of the various losses described in the supplement"""

    def __init__(self, config):
        super(AlphaFoldLoss, self).__init__()
        self.config = config

    def loss(self, out, batch, _return_breakdown=False):
        """
        Rename previous forward() as loss()
        so that can be reused in the subclass 
        """
        if "renamed_atom14_gt_positions" not in out.keys():
            batch.update(
                compute_renamed_ground_truth(
                    batch,
                    out["sm"]["positions"][-1],
                )
            )

        loss_fns = {
            "distogram": lambda: distogram_loss(
                logits=out["distogram_logits"],
                **{**batch, **self.config.distogram},
            ),
            "positions_inter_distogram": lambda: positions_inter_distogram_loss(
                out,
                **{**batch, **self.config.positions_inter_distogram},
            ),
            "positions_intra_distogram": lambda: positions_intra_ligand_distogram_loss(
                out,
                **{**batch, **self.config.positions_intra_distogram},
            ),

            "affinity1d": lambda: affinity_loss(
                logits=out["affinity_1d_logits"],
                **{**batch, **self.config.affinity1d},
            ),
            "affinity2d": lambda: affinity_loss(
                logits=out["affinity_2d_logits"],
                **{**batch, **self.config.affinity2d},
            ),
            "affinity_cls": lambda: affinity_loss(
                logits=out["affinity_cls_logits"],
                **{**batch, **self.config.affinity_cls},
            ),
            "affinity_cls_reg": lambda: affinity_loss_reg(
                logits=out["affinity_cls_reg_logits"],
                **{**batch, **self.config.affinity_cls_reg},
            ),
            "binding_site": lambda: binding_site_loss(
                logits=out["binding_site_logits"],
                **{**batch, **self.config.binding_site},
            ),
            "inter_contact": lambda: inter_contact_loss(
                logits=out["inter_contact_logits"],
                **{**batch, **self.config.inter_contact},
            ),
            # backbone is based on frames so only works on protein
            "fape_backbone": lambda: fape_bb(
                out,
                batch,
                self.config.fape_backbone,
            ),
            "fape_sidechain": lambda: fape_sidechain(
                out,
                batch,
                self.config.fape_sidechain,
            ),
            "fape_interface": lambda: fape_interface(
                out,
                batch,
                self.config.fape_interface,
            ),
            "plddt_loss": lambda: lddt_loss(
                logits=out["lddt_logits"],
                all_atom_pred_pos=out["final_atom_positions"],
                **{**batch, **self.config.plddt_loss},
            ),
            "supervised_chi": lambda: supervised_chi_loss(
                out["sm"]["angles"],
                out["sm"]["unnormalized_angles"],
                **{**batch, **self.config.supervised_chi},
            ),
        }

        if self.config.chain_center_of_mass.enabled:
            loss_fns["chain_center_of_mass"] = lambda: chain_center_of_mass_loss(
                all_atom_pred_pos=out["final_atom_positions"],
                **{**batch, **self.config.chain_center_of_mass},
            )

        cum_loss = 0.
        losses = {}
        loss_time_took = {}
        for loss_name, loss_fn in loss_fns.items():
            start_time = time.time()
            weight = self.config[loss_name].weight
            loss = loss_fn()
            if torch.isnan(loss) or torch.isinf(loss):
                # for k,v in batch.items():
                #    if torch.any(torch.isnan(v)) or torch.any(torch.isinf(v)):
                #        logging.warning(f"{k}: is nan")
                # logging.warning(f"{loss_name}: {loss}")
                logging.warning(f"{loss_name} loss is NaN. Skipping...")
                loss = loss.new_tensor(0., requires_grad=True)
            # else:
            cum_loss = cum_loss + weight * loss
            losses[loss_name] = loss.detach().clone()
            loss_time_took[loss_name] = time.time() - start_time
        losses["unscaled_loss"] = cum_loss.detach().clone()
        # print("loss took: ", round(time.time() % 10000, 3),
        #       sorted(loss_time_took.items(), key=lambda x: x[1], reverse=True))

        # Scale the loss by the square root of the minimum of the crop size and
        # the (average) sequence length. See subsection 1.9.
        seq_len = torch.mean(batch["seq_length"].float())
        crop_len = batch["aatype"].shape[-1]
        cum_loss = cum_loss * torch.sqrt(min(seq_len, crop_len))

        losses["loss"] = cum_loss.detach().clone()

        if not _return_breakdown:
            return cum_loss

        return cum_loss, losses

    def forward(self, out, batch, _return_breakdown=False):
        if not _return_breakdown:
            cum_loss = self.loss(out, batch, _return_breakdown)
            return cum_loss
        else:
            cum_loss, losses = self.loss(out, batch, _return_breakdown)
            return cum_loss, losses
