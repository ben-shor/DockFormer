import copy
import ml_collections as mlc

from evodocker.utils.config_tools import set_inf, enforce_config_constraints


def model_config(
    name, 
    train=False, 
    low_prec=False, 
    long_sequence_inference=False
):
    c = copy.deepcopy(config)
    # TRAINING PRESETS
    if name == "initial_training":
        # AF2 Suppl. Table 4, "initial training" setting

        pass
    elif name == "finetune_affinity":
        # c.loss.violation.weight = 1.  # unclear if this is essential, seems to resolve itself
        c.loss.affinity2d.weight = 0.5
        c.loss.affinity1d.weight = 0.5
        c.loss.binding_site.weight = 0.5
        c.loss.positions_inter_distogram.weight = 0.5  # this is not essential given fape?
    else:
        raise ValueError("Invalid model name")

    c.globals.use_lma = False

    if long_sequence_inference:
        assert(not train)
        c.globals.use_lma = True

    if train:
        c.globals.blocks_per_ckpt = 1
        c.globals.use_lma = False

    if low_prec:
        c.globals.eps = 1e-4
        # If we want exact numerical parity with the original, inf can't be
        # a global constant
        set_inf(c, 1e4)

    enforce_config_constraints(c)

    return c


c_z = mlc.FieldReference(128, field_type=int)
c_m = mlc.FieldReference(256, field_type=int)
c_t = mlc.FieldReference(64, field_type=int)
c_e = mlc.FieldReference(64, field_type=int)
c_s = mlc.FieldReference(384, field_type=int)


blocks_per_ckpt = mlc.FieldReference(None, field_type=int)
aux_distogram_bins = mlc.FieldReference(64, field_type=int)
aux_affinity_bins = mlc.FieldReference(32, field_type=int)
eps = mlc.FieldReference(1e-8, field_type=float)

NUM_RES = "num residues placeholder"
NUM_LIG_ATOMS = "num ligand atoms placeholder"
NUM_TOKEN = "num tokens placeholder"


config = mlc.ConfigDict(
    {
        "data": {
            "common": {
                "feat": {
                    "aatype": [NUM_RES],
                    "all_atom_mask": [NUM_RES, None],
                    "all_atom_positions": [NUM_RES, None, None],
                    "atom14_alt_gt_exists": [NUM_RES, None],
                    "atom14_alt_gt_positions": [NUM_RES, None, None],
                    "atom14_atom_exists": [NUM_RES, None],
                    "atom14_atom_is_ambiguous": [NUM_RES, None],
                    "atom14_gt_exists": [NUM_RES, None],
                    "atom14_gt_positions": [NUM_RES, None, None],
                    "atom37_atom_exists": [NUM_RES, None],
                    "backbone_rigid_mask": [NUM_RES],
                    "backbone_rigid_tensor": [NUM_RES, None, None],
                    "chi_angles_sin_cos": [NUM_RES, None, None],
                    "chi_mask": [NUM_RES, None],
                    "no_recycling_iters": [],
                    "pseudo_beta": [NUM_RES, None],
                    "pseudo_beta_mask": [NUM_RES],
                    "residue_index": [NUM_RES],
                    "in_chain_residue_index": [NUM_RES],
                    "chain_index": [NUM_RES],
                    "residx_atom14_to_atom37": [NUM_RES, None],
                    "residx_atom37_to_atom14": [NUM_RES, None],
                    "resolution": [],
                    "rigidgroups_alt_gt_frames": [NUM_RES, None, None, None],
                    "rigidgroups_group_exists": [NUM_RES, None],
                    "rigidgroups_group_is_ambiguous": [NUM_RES, None],
                    "rigidgroups_gt_exists": [NUM_RES, None],
                    "rigidgroups_gt_frames": [NUM_RES, None, None, None],
                    "seq_length": [],
                    "token_mask": [NUM_TOKEN],
                    "protein_target_feat": [NUM_RES, None],
                    "use_clamped_fape": [],
                },
                "max_recycling_iters": 1,
                "unsupervised_features": [
                    "aatype",
                    "residue_index",
                    "in_chain_residue_index",
                    "chain_index",
                    "seq_length",
                    "no_recycling_iters",
                    "all_atom_mask",
                    "all_atom_positions",
                ],
            },
            "supervised": {
                "clamp_prob": 0.9,
                "supervised_features": [
                    "resolution",
                    "use_clamped_fape",
                ],
            },
            "predict": {
                "fixed_size": True,
                "crop": False,
                "crop_size": None,
                "supervised": False,
                "uniform_recycling": False,
            },
            "eval": {
                "fixed_size": True,
                "crop": False,
                "crop_size": None,
                "supervised": True,
                "uniform_recycling": False,
            },
            "train": {
                "fixed_size": True,
                "crop": True,
                "crop_size": 256,
                "supervised": True,
                "clamp_prob": 0.9,
                "uniform_recycling": True,
            },
            "data_module": {
                "data_loaders": {
                    "batch_size": 1,
                    # "batch_size": 2,
                    "num_workers": 16,
                    "pin_memory": True,
                },
            },
        },
        # Recurring FieldReferences that can be changed globally here
        "globals": {
            "blocks_per_ckpt": blocks_per_ckpt,
            # Use Staats & Rabe's low-memory attention algorithm.
            "use_lma": False,
            "max_lr": 1e-3,
            "c_z": c_z,
            "c_m": c_m,
            "c_t": c_t,
            "c_e": c_e,
            "c_s": c_s,
            "eps": eps,
        },
        "model": {
            "_mask_trans": False,
            "structure_input_embedder": {
                "protein_tf_dim": 20,
                # len(POSSIBLE_ATOM_TYPES) + len(POSSIBLE_CHARGES) + len(POSSIBLE_CHIRALITIES)
                "ligand_tf_dim": 35,
                "ligand_bond_dim": 6,
                "c_z": c_z,
                "c_m": c_m,
                "relpos_k": 32,
                "prot_min_bin": 3.25,
                "prot_max_bin": 20.75,
                "prot_no_bins": 15,
                "lig_min_bin": 0.75,
                "lig_max_bin": 9.75,
                "lig_no_bins": 10,
                "inf": 1e8,
            },
            "recycling_embedder": {
                "c_z": c_z,
                "c_m": c_m,
                "min_bin": 3.25,
                "max_bin": 20.75,
                "no_bins": 15,
                "inf": 1e8,
            },
            "evoformer_stack": {
                "c_m": c_m,
                "c_z": c_z,
                "c_hidden_single_att": 32,
                "c_hidden_mul": 128,
                "c_hidden_pair_att": 32,
                "c_s": c_s,
                "no_heads_single": 8,
                "no_heads_pair": 4,
                # "no_blocks": 48,
                "no_blocks": 2,
                "transition_n": 4,
                "single_dropout": 0.15,
                "pair_dropout": 0.25,
                "blocks_per_ckpt": blocks_per_ckpt,
                "clear_cache_between_blocks": False,
                "inf": 1e9,
                "eps": eps,  # 1e-10,
            },
            "structure_module": {
                "c_s": c_s,
                "c_z": c_z,
                "c_ipa": 16,
                "c_resnet": 128,
                "no_heads_ipa": 12,
                "no_qk_points": 4,
                "no_v_points": 8,
                "dropout_rate": 0.1,
                "no_blocks": 8,
                "no_transition_layers": 1,
                "no_resnet_blocks": 2,
                "no_angles": 7,
                "trans_scale_factor": 10,
                "epsilon": eps,  # 1e-12,
                "inf": 1e5,
            },
            "heads": {
                "lddt": {
                    "no_bins": 50,
                    "c_in": c_s,
                    "c_hidden": 128,
                },
                "distogram": {
                    "c_z": c_z,
                    "no_bins": aux_distogram_bins,
                },
                "experimentally_resolved": {
                    "c_s": c_s,
                    "c_out": 37,
                },
                "affinity_2d": {
                    "c_z": c_z,
                    "num_bins": aux_affinity_bins,
                },
                "affinity_1d": {
                    "c_s": c_s,
                    "num_bins": aux_affinity_bins,
                },
                "affinity_cls": {
                    "c_s": c_s,
                    "num_bins": aux_affinity_bins,
                },
                "binding_site": {
                    "c_s": c_s,
                    "c_out": 1,
                },
                "inter_contact": {
                    "c_s": c_s,
                    "c_z": c_z,
                    "c_out": 1,
                },
            },
            # A negative value indicates that no early stopping will occur, i.e.
            # the model will always run `max_recycling_iters` number of recycling
            # iterations. A positive value will enable early stopping if the
            # difference in pairwise distances is less than the tolerance between
            # recycling steps.
            "recycle_early_stop_tolerance": -1.
        },
        "relax": {
            "max_iterations": 0,  # no max
            "tolerance": 2.39,
            "stiffness": 10.0,
            "max_outer_iterations": 20,
            "exclude_residues": [],
        },
        "loss": {
            "distogram": {
                "min_bin": 2.3125,
                "max_bin": 21.6875,
                "no_bins": 64,
                "eps": eps,  # 1e-6,
                "weight": 0.3,
            },
            "positions_inter_distogram": {
                "max_dist": 20.0,
                "weight": 0.0,
            },
            "positions_intra_distogram": {
                "max_dist": 10.0,
                "weight": 0.0,
            },
            "experimentally_resolved": {
                "eps": eps,  # 1e-8,
                "min_resolution": 0.1,
                "max_resolution": 3.0,
                "weight": 0.0,
            },
            "binding_site": {
                "weight": 0.0,
                "pos_class_weight": 20.0,
            },
            "inter_contact": {
                "weight": 0.0,
                "pos_class_weight": 200.0,
            },
            "affinity2d": {
                "min_bin": 0,
                "max_bin": 15,
                "no_bins": aux_affinity_bins,
                "weight": 0.0,
            },
            "affinity1d": {
                "min_bin": 0,
                "max_bin": 15,
                "no_bins": aux_affinity_bins,
                "weight": 0.0,
            },
            "affinity_cls": {
                "min_bin": 0,
                "max_bin": 15,
                "no_bins": aux_affinity_bins,
                "weight": 0.0,
            },
            "fape_backbone": {
                "clamp_distance": 10.0,
                "loss_unit_distance": 10.0,
                "weight": 0.5,
            },
            "fape_sidechain": {
                "clamp_distance": 10.0,
                "length_scale": 10.0,
                "weight": 0.5,
            },
            "fape_interface": {
                "clamp_distance": 10.0,
                "length_scale": 10.0,
                "weight": 0.0,
            },
            "plddt_loss": {
                "min_resolution": 0.1,
                "max_resolution": 3.0,
                "cutoff": 15.0,
                "no_bins": 50,
                "eps": eps,  # 1e-10,
                "weight": 0.01,
            },
            "supervised_chi": {
                "chi_weight": 0.5,
                "angle_norm_weight": 0.01,
                "eps": eps,  # 1e-6,
                "weight": 1.0,
            },
            "violation": {
                "violation_tolerance_factor": 12.0,
                "clash_overlap_tolerance": 1.5,
                "average_clashes": False,
                "eps": eps,  # 1e-6,
                # TODO bshor: this should only be enabled for fine-tuning?
                "weight": 0.00,
            },
            "chain_center_of_mass": {
                "clamp_distance": -4.0,
                "weight": 0.,
                "eps": eps,
                "enabled": False,
            },
            "eps": eps,
        },
        "ema": {"decay": 0.999},
    }
)
