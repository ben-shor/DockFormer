import json
import sys
from typing import Optional

# This import must be on top to set the environment variables before importing other modules
import env_consts
import time
import os

from lightning.pytorch import seed_everything
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from evodocker.config import model_config
from evodocker.data.data_modules import OpenFoldDataModule
from evodocker.model.model import AlphaFold
from evodocker.utils import residue_constants
from evodocker.utils.exponential_moving_average import ExponentialMovingAverage
from evodocker.utils.loss import AlphaFoldLoss, lddt_ca
from evodocker.utils.lr_schedulers import AlphaFoldLRScheduler
from evodocker.utils.script_utils import get_latest_checkpoint
from evodocker.utils.superimposition import superimpose
from evodocker.utils.tensor_utils import tensor_tree_map
from evodocker.utils.validation_metrics import (
    drmsd,
    gdt_ts,
    gdt_ha,
    rmsd,
)


class ModelWrapper(pl.LightningModule):
    def __init__(self, config):
        super(ModelWrapper, self).__init__()
        self.config = config
        self.model = AlphaFold(config)

        self.loss = AlphaFoldLoss(config.loss)

        self.ema = ExponentialMovingAverage(
            model=self.model, decay=config.ema.decay
        )
        
        self.cached_weights = None
        self.last_lr_step = -1

    def forward(self, batch):
        return self.model(batch)

    def _log(self, loss_breakdown, batch, outputs, train=True):
        phase = "train" if train else "val"
        for loss_name, indiv_loss in loss_breakdown.items():
            self.log(
                f"{phase}/{loss_name}", 
                indiv_loss, 
                on_step=train, on_epoch=(not train), logger=True, sync_dist=True
            )

            if train:
                self.log(
                    f"{phase}/{loss_name}_epoch",
                    indiv_loss,
                    on_step=False, on_epoch=True, logger=True, sync_dist=True
                )

        with torch.no_grad():
            other_metrics = self._compute_validation_metrics(
                batch, 
                outputs,
                superimposition_metrics=(not train)
            )

        for k, v in other_metrics.items():
            self.log(
                f"{phase}/{k}",
                torch.mean(v),
                on_step=False, on_epoch=True, logger=True, sync_dist=True
            )

    def training_step(self, batch, batch_idx):
        if self.ema.device != batch["aatype"].device:
            self.ema.to(batch["aatype"].device)

        # ground_truth = batch.pop('gt_features', None)

        # Run the model
        outputs = self(batch)

        # Remove the recycling dimension
        batch = tensor_tree_map(lambda t: t[..., -1], batch)

        # Compute loss
        loss, loss_breakdown = self.loss(
            outputs, batch, _return_breakdown=True
        )

        # Log it
        self._log(loss_breakdown, batch, outputs)

        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        self.ema.update(self.model)

    def validation_step(self, batch, batch_idx):
        # At the start of validation, load the EMA weights
        if self.cached_weights is None:
            # model.state_dict() contains references to model weights rather
            # than copies. Therefore, we need to clone them before calling 
            # load_state_dict().
            clone_param = lambda t: t.detach().clone()
            self.cached_weights = tensor_tree_map(clone_param, self.model.state_dict())
            self.model.load_state_dict(self.ema.state_dict()["params"])

        # Run the model
        outputs = self(batch)
        batch = tensor_tree_map(lambda t: t[..., -1], batch)

        batch["use_clamped_fape"] = 0.

        # Compute loss and other metrics
        _, loss_breakdown = self.loss(
            outputs, batch, _return_breakdown=True
        )

        self._log(loss_breakdown, batch, outputs, train=False)
        
    def on_validation_epoch_end(self):
        # Restore the model weights to normal
        self.model.load_state_dict(self.cached_weights)
        self.cached_weights = None

    def _compute_validation_metrics(self, 
        batch, 
        outputs, 
        superimposition_metrics=False
    ):
        metrics = {}
        
        protein_gt_coords = batch["all_atom_positions"]
        protein_pred_coords = outputs["final_atom_positions"]
        protein_all_atom_mask = batch["all_atom_mask"]

        ligand_positions = outputs["sm"]["ligand_atom_positions"][-1]
        ligand_gt_positions = batch["gt_ligand_positions"]
    
        # This is super janky for superimposition. Fix later
        gt_coords_masked = protein_gt_coords * protein_all_atom_mask[..., None]
        pred_coords_masked = protein_pred_coords * protein_all_atom_mask[..., None]
        ca_pos = residue_constants.atom_order["CA"]
        gt_coords_masked_ca = gt_coords_masked[..., ca_pos, :]
        pred_coords_masked_ca = pred_coords_masked[..., ca_pos, :]
        all_atom_mask_ca = protein_all_atom_mask[..., ca_pos]
    
        lddt_ca_score = lddt_ca(
            protein_pred_coords,
            protein_gt_coords,
            protein_all_atom_mask,
            eps=self.config.globals.eps,
            per_residue=False,
        )
   
        metrics["lddt_ca"] = lddt_ca_score
   
        drmsd_ca_score = drmsd(
            pred_coords_masked_ca,
            gt_coords_masked_ca,
            mask=all_atom_mask_ca, # still required here to compute n
        )
   
        metrics["drmsd_ca"] = drmsd_ca_score

        drmsd_intra_ligand_score = drmsd(
            ligand_positions,
            ligand_gt_positions,
        )

        metrics["drmsd_intra_ligand"] = drmsd_intra_ligand_score

        # --- inter contacts
        pred_contacts = torch.sigmoid(outputs["inter_contact_logits"].clone().detach()).squeeze(-1)
        pred_contacts = (pred_contacts > 0.5).float()
        gt_contacts = batch["gt_inter_contacts"]

        # Calculate True Positives, False Positives, and False Negatives
        tp = torch.sum((gt_contacts == 1) & (pred_contacts == 1))
        fp = torch.sum((gt_contacts == 0) & (pred_contacts == 1))
        fn = torch.sum((gt_contacts == 1) & (pred_contacts == 0))

        # Calculate Recall and Precision
        recall = tp / (tp + fn) if (tp + fn) > 0 else tp.float()
        precision = tp / (tp + fp) if (tp + fp) > 0 else tp.float()

        metrics["inter_contacts_recall"] = recall.clone().detach()
        metrics["inter_contacts_precision"] = precision.clone().detach()

        # print("inter_contacts recall", recall, "precision", precision, tp, fp, fn, torch.ones_like(gt_contacts).sum())

        # --- Affinity
        pred_affinity_1d = torch.sum(
            torch.softmax(outputs["affinity_1d_logits"].clone().detach(), -1).cpu() * torch.linspace(0, 15, 32), dim=-1).item()

        pred_affinity_2d = torch.sum(
            torch.softmax(outputs["affinity_2d_logits"].clone().detach(), -1).cpu() * torch.linspace(0, 15, 32), dim=-1).item()

        pred_affinity_cls = torch.sum(
            torch.softmax(outputs["affinity_cls_logits"].clone().detach(), -1).cpu() * torch.linspace(0, 15, 32), dim=-1).item()

        metrics["affinity_dist_1d"] = torch.abs(batch["affinity"] - pred_affinity_1d)
        metrics["affinity_dist_2d"] = torch.abs(batch["affinity"] - pred_affinity_2d)
        metrics["affinity_dist_cls"] = torch.abs(batch["affinity"] - pred_affinity_cls)
        metrics["affinity_dist_avg"] = torch.abs(batch["affinity"]
                                                 - (pred_affinity_cls + pred_affinity_1d + pred_affinity_2d) / 3)
        # print("affinity", batch["affinity"], pred_affinity_1d, pred_affinity_2d, pred_affinity_cls,
        #       metrics["affinity_1d_dist"], metrics["affinity_2d_dist"], metrics["affinity_cls_dist"], metrics["affinity_avg_dist"])

        if superimposition_metrics:
            superimposed_pred, alignment_rmsd, rots, transs = superimpose(
                gt_coords_masked_ca, pred_coords_masked_ca, all_atom_mask_ca,
            )
            gdt_ts_score = gdt_ts(
                superimposed_pred, gt_coords_masked_ca, all_atom_mask_ca
            )
            gdt_ha_score = gdt_ha(
                superimposed_pred, gt_coords_masked_ca, all_atom_mask_ca
            )

            metrics["protein_alignment_rmsd"] = alignment_rmsd
            metrics["gdt_ts"] = gdt_ts_score
            metrics["gdt_ha"] = gdt_ha_score

            superimposed_ligand_coords = ligand_positions @ rots + transs[:, None, :]
            metrics["ligand_alignment_rmsd"] = rmsd(ligand_gt_positions, superimposed_ligand_coords)
            metrics["ligand_alignment_rmsd_under_2"] = torch.mean((metrics["ligand_alignment_rmsd"] < 2).float())
            metrics["ligand_alignment_rmsd_under_5"] = torch.mean((metrics["ligand_alignment_rmsd"] < 5).float())

            print("ligand rmsd:", metrics["ligand_alignment_rmsd"])

        return metrics

    def configure_optimizers(self, 
        learning_rate: Optional[float] = None,
        eps: float = 1e-5,
    ) -> torch.optim.Adam:
        if learning_rate is None:
            learning_rate = self.config.globals.max_lr

        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            eps=eps
        )

        if self.last_lr_step != -1:
            for group in optimizer.param_groups:
                if 'initial_lr' not in group:
                    group['initial_lr'] = learning_rate

        lr_scheduler = AlphaFoldLRScheduler(
            optimizer,
            last_epoch=self.last_lr_step,
            max_lr=self.config.globals.max_lr,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "name": "AlphaFoldLRScheduler",
            }
        }

    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint["ema"]
        self.ema.load_state_dict(ema)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema"] = self.ema.state_dict()

    def resume_last_lr_step(self, lr_step):
        self.last_lr_step = lr_step


def override_config(base_config, overriding_config):
    for k, v in overriding_config.items():
        if isinstance(v, dict):
            base_config[k] = override_config(base_config[k], v)
        else:
            base_config[k] = v
    return base_config


def train(override_config_path: str):
    run_config = json.load(open(override_config_path, "r"))
    seed = 42
    seed_everything(seed, workers=True)
    output_dir = run_config["train_output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    config = model_config(
        run_config.get("stage", "initial_training"),
        train=True,
        low_prec=True
    )
    config = override_config(config, run_config.get("override_conf", {}))

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = "mps" if device_name == "cpu" and torch.backends.mps.is_available() else device_name
    model_module = ModelWrapper(config)

    # for debugging memory:
    # torch.cuda.memory._record_memory_history()

    data_module = OpenFoldDataModule(
        config=config.data,
        batch_seed=seed,
        train_data_dir=run_config["train_input_dir"],
        val_data_dir=run_config["val_input_dir"],
        train_epoch_len=run_config.get("train_epoch_len", 1000),
    )

    checkpoint_dir = os.path.join(output_dir, "checkpoint")
    ckpt_path = run_config.get("ckpt_path", get_latest_checkpoint(checkpoint_dir))

    if ckpt_path:
        print(f"Resuming from checkpoint: {ckpt_path}")
        sd = torch.load(ckpt_path)
        last_global_step = int(sd['global_step'])
        model_module.resume_last_lr_step(last_global_step)

    data_module.prepare_data()
    data_module.setup("fit")

    callbacks = []

    mc = ModelCheckpoint(
        dirpath=checkpoint_dir,
        every_n_epochs=1,
        auto_insert_metric_name=False,
        save_top_k=1,
        save_on_train_epoch_end=True,  # before validation
    )
    callbacks.append(mc)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    loggers = []

    wandb_project_name = "EvoDocker3"
    wandb_run_id_path = os.path.join(output_dir, "wandb_run_id.txt")

    # Initialize WandbLogger and save run_id
    local_rank = int(os.getenv('LOCAL_RANK', os.getenv("SLURM_PROCID", '0')))
    global_rank = int(os.getenv('GLOBAL_RANK', os.getenv("SLURM_LOCALID", '0')))
    print("ranks", os.getenv('LOCAL_RANK', 'd0'), os.getenv('local_rank', 'd0'), os.getenv('GLOBAL_RANK', 'd0'),
          os.getenv('global_rank', 'd0'), os.getenv("SLURM_PROCID", 'd0'), os.getenv('SLURM_LOCALID', 'd0'), flush=True)
    if local_rank == 0 and global_rank == 0 and not os.path.exists(wandb_run_id_path):
        wandb_logger = WandbLogger(project=wandb_project_name, save_dir=output_dir)
        with open(wandb_run_id_path, 'w') as f:
            f.write(wandb_logger.experiment.id)
        wandb_logger.experiment.config.update(run_config, allow_val_change=True)
    else:
        # Necessary for multi-node training https://github.com/rstrudel/segmenter/issues/22
        while not os.path.exists(wandb_run_id_path):
            print(f"Waiting for run_id file to be created ({local_rank})", flush=True)
            time.sleep(1)
        with open(wandb_run_id_path, 'r') as f:
            run_id = f.read().strip()
        wandb_logger = WandbLogger(project=wandb_project_name, save_dir=output_dir, resume='must', id=run_id)
    loggers.append(wandb_logger)

    strategy_params = {"strategy": "auto"}
    if run_config.get("multi_node", False):
        strategy_params["strategy"] = "ddp"
        strategy_params["num_nodes"] = run_config["multi_node"]["num_nodes"]
        strategy_params["devices"] = run_config["multi_node"]["devices"]

    trainer = pl.Trainer(
        accelerator=device_name,
        default_root_dir=output_dir,
        **strategy_params,
        reload_dataloaders_every_n_epochs=1,
        # accumulate_grad_batches=32, # can be used to simulate larger batch sizes
        check_val_every_n_epoch=run_config.get("check_val_every_n_epoch", 10),
        callbacks=callbacks,
        logger=loggers,
    )

    trainer.fit(
        model_module,
        datamodule=data_module,
        ckpt_path=ckpt_path,
    )

    # torch.cuda.memory._dump_snapshot("my_train_snapshot.pickle")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        train(sys.argv[1])
    else:
        train(os.path.join(os.path.dirname(__file__), "run_config.json"))

