import torch
from torch_geometric.data.lightning import LightningDataset
import lightning as L
from models.gnn import GNN
from models.adgn import ADGN
from models.drew_delay import DRew_GCN
from models.graphcon import GraphCON
from models.phdgn import PHDGN
from models.swan import SWAN
from models.gssm import GSSM
from models.glgsm import GenLGSM
import time
import csv
import pathlib

from typing import Optional


class DiversityPlotCallback(L.Callback):
    """Every `every_n_epochs`, plots per-step node-rep diversity for one training batch."""

    def __init__(self, plot_dir: str, every_n_epochs: int = 10):
        self.plot_dir = pathlib.Path(plot_dir)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self._capturing = False

        self.every_n_epochs = every_n_epochs

    def _gssm_layer(self):
        try:
            from gssm_layer import GSSMLayer
            return GSSMLayer
        except ImportError:
            return None

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            cls = self._gssm_layer()
            if cls is not None:
                cls.enable_diversity_tracking()
                self._capturing = True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._capturing and batch_idx == 0:
            cls = self._gssm_layer()
            if cls is not None:
                cls.disable_diversity_tracking()
                self._capturing = False
                diversities = cls.get_diversity_data()
                if diversities:
                    self._plot(trainer.current_epoch, diversities)

    def _plot(self, epoch: int, diversities: list):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[DiversityPlot] matplotlib not available — skipping plot")
            return
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(len(diversities)), diversities, marker="o", markersize=3, linewidth=1.5)
        ax.set_xlabel("Recurrence step")
        ax.set_ylabel("Node rep std (diversity)")
        ax.set_title(f"Node representation diversity across steps — epoch {epoch}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = self.plot_dir / f"diversity_epoch_{epoch:04d}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[DiversityPlot] saved → {path}")


models_map = {
    "GNN": GNN,
    "ADGN": ADGN,
    "DRew_GCN": DRew_GCN,
    "GraphCON": GraphCON,
    "PHDGN": PHDGN,
    "SWAN": SWAN,
    "GSSM": GSSM,
    "GenLGSM": GenLGSM,
}

def convert_to_lit_dataset(data):
    return LightningDataset(data)


class LitGraphNN(L.LightningModule):
    def __init__(
        self,
        gnn_type: str,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 1,
        node_level_task: bool = False,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        scaling_factor: float = 1.0,
        enable_timing: bool = False,
        timing_csv_base_path: str = "training_timings", # New parameter for base path
        task: str = "sssp",
        lr_scheduler: str = "none",
        max_lr: Optional[float] = None,
        lr_min: float = 1e-6,
        cosine_T0: int = 25,
        scheduler_patience: int = 20,
        **kwargs,
    ) -> None:
        super().__init__()
        self.gnn_type = gnn_type
        self.conv_layer = kwargs.get("conv_layer")
        self.enable_timing = enable_timing
        self._epoch_start_time = None
        self.timing_csv_file = None
        self.task = task
        self.scaling_factor = scaling_factor

        self.model = models_map[gnn_type](
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            node_level_task=node_level_task,
            **kwargs,
        )
            
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scaling_factor = scaling_factor
        self.lr_scheduler_type = lr_scheduler
        self.max_lr = max_lr if max_lr is not None else lr * 5
        self.lr_min = lr_min
        self.cosine_T0 = cosine_T0
        self.scheduler_patience = scheduler_patience

        # save hyperparameters
        self.save_hyperparameters()

        if self.enable_timing:
            self.timing_csv_base_path = pathlib.Path(timing_csv_base_path)
            try:
                self.timing_csv_base_path.mkdir(parents=True, exist_ok=True)
                print(f"Timing CSV base path: {self.timing_csv_base_path.resolve()}")
            except OSError as e:
                print(f"Error creating directory {self.timing_csv_base_path}: {e}")
                self.enable_timing = False # Disable timing if directory creation fails

        if self.enable_timing:
            timing_filename_parts = [self.gnn_type]
            if self.conv_layer:
                timing_filename_parts.append(str(self.conv_layer))
            timing_filename_parts.append(str(self.task))
            timing_filename_parts.append("timing.csv")
            filename = "_".join(filter(None, timing_filename_parts))
            self.timing_csv_file = self.timing_csv_base_path / filename

            print(f"Timing enabled. Data will be saved to: {self.timing_csv_file.resolve()}")
            if not self.timing_csv_file.exists():
                try:
                    with open(self.timing_csv_file, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(["epoch", "training_time_seconds"])
                except OSError as e:
                    print(f"Error creating or writing header to {self.timing_csv_file}: {e}")
                    self.timing_csv_file = None # Disable CSV writing for this file
                    self.enable_timing = False # Or disable timing altogether
    
    def on_train_epoch_start(self):
        if self.enable_timing:
            self._epoch_start_time = time.time()

    def on_train_epoch_end(self):
        if self.enable_timing and self._epoch_start_time is not None and self.timing_csv_file:
            epoch_duration = time.time() - self._epoch_start_time
            current_epoch_to_log = self.current_epoch # In PL, current_epoch is 0-indexed during training
            try:
                with open(self.timing_csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([current_epoch_to_log, epoch_duration])
            except Exception as e:
                print(f"Error writing to timing CSV {self.timing_csv_file}: {e}")
            self._epoch_start_time = None

        self._log_coeff_per_step()

    def _hyper_hop_layer(self):
        """GenLGSM's hop layer when running glgsm_mode=hyper, else None."""
        inner = getattr(self.model, "model", None)   # GenLGSM → GenLGSMModel
        hop = getattr(inner, "hop_layer", None)
        return hop if getattr(hop, "mode", None) == "hyper" else None

    def _log_hyper_coeffs(self, batch_size: int):
        """Whole-model summary scalars for the hypernetwork's α/β, averaged over the epoch.

        graph_std is the key diagnostic: the coefficients are only graph-conditioned if
        the hypernetwork makes them differ between graphs, so a graph_std pinned at 0
        means the MLP has learned nothing graph-specific.
        """
        hop = self._hyper_hop_layer()
        if hop is None or hop.last_coeffs is None:
            return

        for name in ("alpha_A", "alpha_D", "alpha_I", "beta"):
            a = hop.last_coeffs[name]                    # (B, L, M) — (B, L) for beta
            self.log(f"coeff/{name}_mean", a.mean(),
                     on_step=False, on_epoch=True, batch_size=batch_size)
            self.log(f"coeff/{name}_absmax", a.abs().max(), reduce_fx="max",
                     on_step=False, on_epoch=True, batch_size=batch_size)
            if a.size(0) > 1:                            # std over dim 0 is NaN when B == 1
                self.log(f"coeff/{name}_graph_std", a.std(dim=0).mean(),
                         on_step=False, on_epoch=True, batch_size=batch_size)

    def _log_coeff_per_step(self):
        """One line series per (sequence step, memory slot), grouped by step in wandb.

        Keys are coeff/step{k}/alpha_A_j{j}, so wandb puts each sequence step in its own
        panel section with 3M+1 curves tracked across epochs. Logged once per epoch from
        the epoch-end hook rather than per batch: L*(3M+1) is 800 series at L=32, M=8,
        which is far too many self.log calls to make on every training step. The values
        are therefore the epoch's last batch, not an epoch average.
        """
        hop = self._hyper_hop_layer()
        if hop is None or hop.last_coeffs is None:
            return

        c = hop.last_coeffs
        L = c["beta"].size(1)
        for k in range(L):
            for name in ("alpha_A", "alpha_D", "alpha_I"):
                a_k = c[name][:, k]                          # (B, M) — this hop, all slots
                for j in range(a_k.size(-1)):
                    self.log(f"coeff/step{k:02d}/{name}_j{j}", a_k[:, j].mean(), batch_size=1)
            self.log(f"coeff/step{k:02d}/beta", c["beta"][:, k].mean(), batch_size=1)

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        out: torch.Tensor = self.model(batch).squeeze(-1)
        loss = self.criterion(out, batch.y)
        loss = torch.log10(loss)

        if self.task == "energy":
            # For energy task, we need to convert the output and target to the same scale
            out = 10**out.detach()
            batch.y = 10**batch.y

        self.log(
            "train_loss",
            loss,
            sync_dist=True,
            batch_size=batch.y.size(0),
        )
        self.log(
            "train_mae",
            torch.nn.functional.l1_loss(
                out.detach() * self.scaling_factor, batch.y * self.scaling_factor
            ),
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch.y.size(0),
        )
        self.log(
            "train_mse",
            torch.nn.functional.mse_loss(
                out.detach() * self.scaling_factor, batch.y * self.scaling_factor
            ),
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch.y.size(0),
        )
        self._log_hyper_coeffs(batch.y.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        # check if batch.x is double cast to float
        if batch.x.dtype == torch.float64:
            batch.x = batch.x.float()
        if batch.edge_attr is not None and batch.edge_attr.dtype == torch.float64:
            batch.edge_attr = batch.edge_attr.float()
        # check if batch.y is double cast to float
        if batch.y.dtype == torch.float64:
            batch.y = batch.y.float()

        out = self.model(batch).squeeze(-1)
        loss = self.criterion(out, batch.y)
        loss = torch.log10(loss)

        if self.task == "energy":
            # For energy task, we need to convert the output and target to the same scale
            out = 10**out.detach()
            batch.y = 10**batch.y
        

        self.log("val_loss", loss, sync_dist=True, batch_size=batch.y.size(0))
        self.log(
            "val_mae",
            torch.nn.functional.l1_loss(
                out.detach() * self.scaling_factor, batch.y * self.scaling_factor
            ),
            sync_dist=True,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch.y.size(0),
        )
        self.log(
            "val_mse",
            torch.nn.functional.mse_loss(
                out.detach() * self.scaling_factor, batch.y * self.scaling_factor
            ),
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch.y.size(0),
        )
        return loss



    def test_step(self, batch, batch_idx):        
        out = self.model(batch).squeeze(-1)
        loss = self.criterion(out, batch.y)
        loss = torch.log10(loss)


        if self.task == "energy":
            # For energy task, we need to convert the output and target to the same scale
            out = 10**out.detach()
            batch.y = 10**batch.y
        

        self.log("test_loss", loss, sync_dist=True, batch_size=batch.y.size(0))
        self.log(
            "test_mae",
            torch.nn.functional.l1_loss(
                out.detach() * self.scaling_factor, batch.y * self.scaling_factor
            ),
            sync_dist=True,
            batch_size=batch.y.size(0),
        )
        self.log(
            "test_mse",
            torch.nn.functional.mse_loss(
                out.detach() * self.scaling_factor, batch.y * self.scaling_factor
            ),
            sync_dist=True,
            batch_size=batch.y.size(0),
        )
        return loss

    def configure_optimizers(self):
        opt = self.optimizer
        sched_type = self.lr_scheduler_type

        if sched_type == "none":
            return opt

        if sched_type == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=self.max_lr,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.1,
                anneal_strategy="cos",
                final_div_factor=1e3,
            )
            return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

        if sched_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.trainer.max_epochs, eta_min=self.lr_min
            )
            return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

        if sched_type == "cosine_wr":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt, T_0=self.cosine_T0, T_mult=2, eta_min=self.lr_min
            )
            return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

        if sched_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=0.5, patience=self.scheduler_patience
            )
            return {
                "optimizer": opt,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss", "interval": "epoch"},
            }

        raise ValueError(f"Unknown lr_scheduler: {sched_type!r}")
    

    def __str__(self):
        params = ", ".join(f"{k}={v}" for k, v in self.hparams.items())
        return f"LitGraphNN({params})" + f" with underlying model: {str(self.model)}"
    

