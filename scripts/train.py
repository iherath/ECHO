## add parent directory to path
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected for --selective.')

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, help="Task to run: [sssp, ecc, diam, charge, energy, lrim]",)

parser.add_argument("--device", type=str, default="gpu", help="Device to use for training")
parser.add_argument("--devices", type=int, default=1,
                    help="GPUs per node for data-parallel (DDP) training; >1 shards each epoch across GPUs")
parser.add_argument("--num_nodes", type=int, default=1,
                    help="Nodes for multi-node DDP (Bridges-2 GPU partition: 1 node = 8 GPUs, 2 = 16 GPUs)")
parser.add_argument("--resume", action="store_true",
                    help="Resume from ./checkpoints/<task>_seed<seed>/last.ckpt if present (for chaining 48h jobs)")
# general gnn parameters
parser.add_argument("--conv_layer", type=str)
parser.add_argument("--num_layers", type=int, help="Number of layers in the GNN")
parser.add_argument("--hidden_dim", type=int, help="Hidden dimension of the GNN")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer")
parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for the optimizer")
parser.add_argument("--lr_scheduler", type=str, default="none",
                    choices=["none", "onecycle", "cosine", "cosine_wr", "plateau"],
                    help="LR scheduler: none (fixed), onecycle, cosine, cosine_wr, plateau")
parser.add_argument("--max_lr", type=float, default=None,
                    help="Peak LR for onecycle (defaults to 5x --lr if not set)")
parser.add_argument("--lr_min", type=float, default=1e-6,
                    help="Minimum LR floor for cosine/cosine_wr schedulers")
parser.add_argument("--cosine_T0", type=int, default=25,
                    help="Epochs per first restart cycle for cosine_wr")
parser.add_argument("--scheduler_patience", type=int, default=20,
                    help="Epochs of stagnant val_loss before LR is halved (plateau only)")
parser.add_argument("--max_epochs", type=int, default=1000, help="Maximum training epochs")
parser.add_argument("--es_patience", type=int, default=100, help="Early stopping patience (epochs)")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size for the DataLoader")
parser.add_argument("--accumulate_grad_batches", type=int, default=1,
                    help="Micro-batches per optimizer step; effective batch = batch_size x this")
parser.add_argument("--gnn_type", type=str)

# adgn, swan specific params
parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon for the ADGN model")
parser.add_argument("--gamma", type=float, default=0.1, help="Gamma for the ADGN model")
parser.add_argument("--activ_fun", type=str, default="tanh", help="Activation function for the ADGN model")
parser.add_argument("--graph_conv", type=str, default="GCNConv", help="Graph convolution layer for the ADGN model")
parser.add_argument("--bias", type=bool, help="Use bias in the ADGN model")
parser.add_argument("--train_weights", type=bool)
parser.add_argument("--weight_sharing", type=bool, help="Use weight sharing in the ADGN model")

# drew specific parameters
parser.add_argument("--khop", type=int)
parser.add_argument("--delay", type=bool)
parser.add_argument("--constant_feature", type=float, help="Constant feature")

# gcn2 params
parser.add_argument("--alpha", type=float, help="Alpha for the GCN2 model")

# gssm specific parameters
parser.add_argument("--d_state", type=int, default=16, help="SSM state dimension M for GSSM")
parser.add_argument("--num_steps", type=int, default=40, help="Fixed recurrence steps per GSSMLayer (replaces convergence-gating)")
parser.add_argument("--selective", type=str2bool, nargs='?', const=True, default=False, help="Use selective GSSMLayer (Mamba-style selectivity) [True|False]")
parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate in FFN layers (GSSM only)")
parser.add_argument("--convergence_log", type=str, default=None, help="Path to txt file for logging GSSMLayer step counts")
parser.add_argument("--diversity_plot_dir", type=str, default=None, help="Directory to save per-step node diversity plots every 10 epochs (GSSM only)")
parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")

# GenLGSM parameters
parser.add_argument("--glgsm_mode", type=str, default="option1",
                    help="Hop mode: option1/option2/path_b (learnable) or lgsm_adj/lgsm_nbt (paper Seq) or hyper")
parser.add_argument("--window_size", type=int, default=2,
                    help="Memory depth M for GenLGSM 'hyper' mode (M=2 suffices to recover NBT recurrence)")
parser.add_argument("--hyper_hidden_dim", type=int, default=64,
                    help="Hypernetwork MLP hidden dim for GenLGSM 'hyper' mode")
parser.add_argument("--hyper_init", type=str, default="gcn", choices=["gcn", "chebyshev"],
                    help="Initial coefficient prior for 'hyper' mode: gcn ([H, SH, S^2H, ...]) "
                         "or chebyshev ([H, T_1(S)H, T_2(S)H, ...]); chebyshev needs window_size>=2")
parser.add_argument("--hyper_init_noise", type=float, default=0.0,
                    help="Std of N(0, sigma^2) noise added to the alpha coefficients of the "
                         "hyper_init prior; 0.0 = exact init. Beta is left unperturbed.")
parser.add_argument("--batched", action="store_true",
                    help="Use padded-batch vectorized layers for GenLGSM (8-15x faster)")

# LRIM (Long-Range Ising Model) benchmark: --task lrim selects the HF file below
parser.add_argument("--lrim_name", type=str, default="lrim_16_0.6_10k",
                    help="LRIM dataset file on HuggingFace (e.g. lrim_16_0.6_10k = 16x16 grid, sigma=0.6 hard)")

# phdgn specific parameters
parser.add_argument("--beta", type=float, help="Beta parameter for the PHDGN model")
parser.add_argument("--p_conv_mode", type=str, choices=["naive", "gcn"], help="P convolution mode for the PhDGN model")
parser.add_argument("--q_conv_mode", type=str, choices=["naive", "gcn"], help="Q convolution mode for the PhDGN model")
parser.add_argument("--doubled_dim", type=bool, choices=[True, False], help="Whether to double the dimension in the PhDGN model")
parser.add_argument("--final_state", type=str, choices=["p", "q", "pq"], help="Final state mode for the PhDGN model")
parser.add_argument("--dampening_mode", type=str, choices=["param", "param+", "MLP4ReLU", "DGNReLU", "none"], help="Dampening mode for the PhDGN model")
parser.add_argument("--external_mode", type=str, choices=["MLP4Sin", "DGNtanh", "none"], help="External mode for the PhDGN model")


from torch_geometric.loader import DataLoader

import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger

import os
from utils import get_dataset, KHopTransform
from utils.litmodels import LitGraphNN, DiversityPlotCallback
import re


torch.set_float32_matmul_precision("high")
get_epoch = lambda path: int(re.findall(r"epoch=(\d+)", path)[0])


def train(seed, config):
    """Train and validate the model"""
    config = parser.parse_args()
    task = config.task

    L.seed_everything(seed) 
    batch_size = config.batch_size

    print("Current directory: ", os.getcwd())

    if task == "lrim":
        # LRIM: per-node energy regression, no target normalization -> native scale
        from utils.lrim_dataset import get_lrim_dataset
        data_train, data_val, data_test, num_feat, num_class = get_lrim_dataset(
            root="./data/", name=config.lrim_name)
        scaling_factor = 1.0
    else:
        data_train, data_val, data_test, num_feat, num_class = get_dataset(
            root="./data/",
            task=task,
            pre_transform=(
                KHopTransform(k=config.k_hop)
                if config.gnn_type == "DRew_GCN"
                else None
            ),
            constant_feature=config.constant_feature,
        )

        scaling_factor = data_train.scaling_factor[task]

        # charge/energy have no target normalization (max_charge is None) -> report in native scale
        if scaling_factor is None:
            scaling_factor = 1.0


    print(f"Scaling factor for {task}: {scaling_factor}")
    print(f"Scaling factor: {scaling_factor}")

    train_loader = DataLoader(
        data_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        data_val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        data_test, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )


    print("Data loaded")

    hp_conf = vars(config)

    model = LitGraphNN(
        input_dim=num_feat,
        output_dim=num_class,
        node_level_task=task not in ("diam", "energy"),  # graph-level tasks
        scaling_factor=scaling_factor,
        **hp_conf,
    )

    # encode the key hyperparams in the run name so wandb runs are self-describing.
    # Non-default hyper_init is appended so it doesn't collide with an otherwise
    # identically-named run; the default keeps the original naming.
    init_tag = "" if config.hyper_init == "gcn" else f"-{config.hyper_init}"
    if config.hyper_init_noise > 0.0:
        init_tag += f"-noise{config.hyper_init_noise:g}"
    run_name = (
        f"{task}-{config.gnn_type}-layers{config.num_layers}-steps{config.num_steps}"
        f"-window{config.window_size}-seed{config.seed}{init_tag}"
    )
    logger = (
        WandbLogger(project="ECHO-GSSM", name=run_name)
        if config.wandb
        else CSVLogger(save_dir=".", name="lightning_logs", version=f"{task}_{config.seed}")
    )
    # per-(task,seed) checkpoint dir + save_last so a re-submitted job can resume from last.ckpt
    ckpt_dir = f"./checkpoints/{task}_seed{config.seed}"
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=config.es_patience),
        ModelCheckpoint(dirpath=ckpt_dir, monitor="val_loss", save_top_k=1, save_last=True),
    ]
    if config.diversity_plot_dir and config.gnn_type == "GSSM":
        callbacks.append(DiversityPlotCallback(config.diversity_plot_dir))

    # DDP: each of (devices x num_nodes) GPUs trains a different batch shard in parallel (~N x more epochs/hour)
    world_size = config.devices * config.num_nodes
    trainer = L.Trainer(
        max_epochs=config.max_epochs,
        accelerator=config.device,
        devices=config.devices,
        num_nodes=config.num_nodes,
        strategy="ddp" if world_size > 1 else "auto",
        gradient_clip_val=1.0,
        accumulate_grad_batches=config.accumulate_grad_batches,
        callbacks=callbacks,
        logger=logger,
    )

    # --resume continues from last.ckpt (EarlyStopping counter restored too); None = fresh start
    last_ckpt = os.path.join(ckpt_dir, "last.ckpt")
    resume_path = last_ckpt if config.resume and os.path.exists(last_ckpt) else None
    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_path)

    if config.convergence_log and config.gnn_type == "GSSM":
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "../../Graph-SSM"))
        from gssm_layer import GSSMLayer
        GSSMLayer.flush_log()

    best_epoch = get_epoch(trainer.checkpoint_callback.best_model_path) #type: ignore
    print(f"Best epoch: {best_epoch}")
    print("Best checkpoint path: ", trainer.checkpoint_callback.best_model_path) #type: ignore

    trainer.validate(model, val_loader, ckpt_path="best")
    print("Testing model")
    print(test_loader)
    trainer.test(model, test_loader, ckpt_path="best")

    # log metrics to a dictionary and return it.
    metrics = {
        "train_loss": trainer.callback_metrics["train_loss"].item(),
        "val_loss": trainer.callback_metrics["val_loss"].item(),
        "val_mse": trainer.callback_metrics["val_mse"].item(),
        "val_mae": trainer.callback_metrics["val_mae"].item(),
        "test_loss": trainer.callback_metrics["test_loss"].item(),
        "test_mse": trainer.callback_metrics["test_mse"].item(),
        "test_mae": trainer.callback_metrics["test_mae"].item(),
        "best_epoch": best_epoch,
        "best_checkpoint_path": trainer.checkpoint_callback.best_model_path, #type: ignore
    }

    return metrics



if __name__ == "__main__":
    args = parser.parse_args()
    metrics = train(
        seed=args.seed,
        config=args,
    )

    # under DDP srun runs one process per GPU; only rank 0 prints the summary (SLURM_PROCID=0)
    if int(os.environ.get("SLURM_PROCID", 0)) == 0:
        print("\n" + "=" * 40)
        print(f"  Task : {args.task.upper()}   Model : {args.gnn_type}")
        print("=" * 40)
        print(f"  val   MAE : {metrics['val_mae']:.6f}")
        print(f"  val   MSE : {metrics['val_mse']:.6f}")
        print(f"  test  MAE : {metrics['test_mae']:.6f}")
        print(f"  test  MSE : {metrics['test_mse']:.6f}")
        print(f"  best epoch: {metrics['best_epoch']}")
        print("=" * 40 + "\n")
