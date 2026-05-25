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
parser.add_argument("--task", type=str, help="Task to run: [sssp, ecc, diam, chem]",)

parser.add_argument("--device", type=str, default="gpu", help="Device to use for training")
# general gnn parameters
parser.add_argument("--conv_layer", type=str)
parser.add_argument("--num_layers", type=int, help="Number of layers in the GNN")
parser.add_argument("--hidden_dim", type=int, help="Hidden dimension of the GNN")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer")
parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for the optimizer")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size for the DataLoader")
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

# GenLGSM parameters
parser.add_argument("--glgsm_mode", type=str, default="option1",
                    help="Hop mode: option1/option2/path_b (learnable) or lgsm_adj/lgsm_nbt (paper Seq)")

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
from lightning.pytorch.loggers import WandbLogger

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

    if scaling_factor is None and task == "chem":
        scaling_factor = 1.0


    print(f"Scaling factor for {task}: {scaling_factor}")
    print(f"Scaling factor: {scaling_factor}")

    train_loader = DataLoader(
        data_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(
        data_val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True
    )
    test_loader = DataLoader(
        data_test, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True
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

    logger = (
        WandbLogger(project="ECHO-GSSM", name=f"{task}-{config.gnn_type}")
        if config.wandb else True  # True = Lightning's default CSVLogger
    )
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=100),
        ModelCheckpoint(monitor="val_loss", save_top_k=1),
    ]
    if config.diversity_plot_dir and config.gnn_type == "GSSM":
        callbacks.append(DiversityPlotCallback(config.diversity_plot_dir))

    trainer = L.Trainer(
        max_epochs=1000,
        accelerator="gpu",
        gradient_clip_val=1.0,
        callbacks=callbacks,
        logger=logger,
    )

    trainer.fit(model, train_loader, val_loader)

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
        "test_acc": trainer.callback_metrics["test_acc"].item(),
        "best_epoch": best_epoch,
        "best_checkpoint_path": trainer.checkpoint_callback.best_model_path, #type: ignore
    }

    return metrics



if __name__ == "__main__":
    args = parser.parse_args()
    metrics = train(
        seed=1,
        config=args,
    )

    print("\n" + "=" * 40)
    print(f"  Task : {args.task.upper()}   Model : {args.gnn_type}")
    print("=" * 40)
    print(f"  val   MAE : {metrics['val_mae']:.6f}")
    print(f"  val   MSE : {metrics['val_mse']:.6f}")
    print(f"  test  MAE : {metrics['test_mae']:.6f}")
    print(f"  test  MSE : {metrics['test_mse']:.6f}")
    print(f"  best epoch: {metrics['best_epoch']}")
    print("=" * 40 + "\n")
