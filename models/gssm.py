"""
ECHO-compatible adapter for FullGSSMModel.

ECHO expects every model to accept:
    __init__(input_dim, output_dim, hidden_dim, num_layers, node_level_task, **kwargs)
    forward(data: torch_geometric.data.Data) -> Tensor

This file translates those conventions to FullGSSMModel's interface.

Dataset feature dims (from ECHO repo):
    ECHO-Synth : node_dim=2, no edge_attr
    ECHO-Chem  : node_dim=2, edge_attr_dim=2
"""

import os
import sys

# Allow importing from the sibling Graph-SSM project without installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../Graph-SSM"))

import torch.nn as nn
from gssm_model import FullGSSMModel


class GSSM(nn.Module):
    """
    Thin ECHO wrapper around FullGSSMModel.

    Extra kwargs forwarded from train.py argparse (e.g. --d_state 16):
        d_state   (int)   : SSM state dimension M  (default 16)
        tol       (float) : convergence tolerance   (default 1e-5)
        max_steps (int)   : iteration hard cap      (default 500)
    """

    def __init__(
        self,
        input_dim: int,          # ECHO: num_feat (2 for Synth and Chem)
        output_dim: int,         # ECHO: num_class (1 for all regression tasks)
        hidden_dim: int = 64,    # d_model
        num_layers: int = 4,
        node_level_task: bool = False,
        d_state: int = 16,
        num_steps: int = 40,
        **kwargs,
    ):
        super().__init__()

        # ECHO-Chem passes edge_attr with dim=2; Synth has no edge features
        d_edge     = kwargs.get("d_edge",          None)
        selective  = kwargs.get("selective",       False)
        dropout    = kwargs.get("dropout",         0.0)
        conv_log   = kwargs.get("convergence_log", None)

        if conv_log:
            from gssm_layer import GSSMLayer
            GSSMLayer.configure_logging(conv_log)

        self.model = FullGSSMModel(
            d_input=input_dim,                               # (N, input_dim)
            d_model=hidden_dim,                              # hidden dim D
            d_state=d_state,                                 # SSM state dim M
            num_layers=num_layers,
            num_tasks=output_dim,                            # 1 for regression
            task_type="node" if node_level_task else "graph",
            d_edge=d_edge,
            num_steps=num_steps,
            selective=selective,
            dropout=dropout,
        )

    def forward(self, data):                                 # data: PyG Data/Batch
        edge_attr = getattr(data, "edge_attr", None)         # (E, F_e) or None
        batch     = getattr(data, "batch",     None)         # (N,)     or None
        return self.model(data.x, data.edge_index, edge_attr, batch)  # (B,1) or (N,1)
