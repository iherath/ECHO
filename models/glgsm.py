"""
ECHO adapter for GenLGSMModel (PyG sparse batch, no cross-graph padding).

kwargs:
    glgsm_mode       : option1 | option2 | path_b | lgsm_adj | lgsm_nbt | hyper
    num_layers       : processing depth (default 4)
    num_steps        : sequence length L (default 40)
    window_size      : memory depth M for 'hyper' mode (default 2)
    hyper_hidden_dim : hypernetwork MLP hidden dim for 'hyper' mode (default 64)
    batched          : use padded-batch vectorized layers (default False)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../HOPPER"))

import torch.nn as nn

from glgsm_model import GenLGSMModel


class GenLGSM(nn.Module):
    """ECHO wrapper — same forward contract as GSSM (sparse x, edge_index, batch)."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 4,
        node_level_task: bool = False,
        d_state: int = 16,
        num_steps: int = 40,
        **kwargs,
    ):
        super().__init__()
        mode             = kwargs.get('glgsm_mode', 'option1')
        dropout          = kwargs.get('dropout', 0.0)
        window_size      = kwargs.get('window_size', 2)
        hyper_hidden_dim = kwargs.get('hyper_hidden_dim', 64)
        hyper_init       = kwargs.get('hyper_init', 'gcn')
        hyper_init_noise = kwargs.get('hyper_init_noise', 0.0)
        batched          = kwargs.get('batched', False)

        self.model = GenLGSMModel(
            in_dim=input_dim,
            d_model=hidden_dim,
            d_state=d_state,
            out_dim=output_dim,
            max_hops=num_steps,
            mode=mode,
            num_blocks=num_layers,
            task_type="node" if node_level_task else "graph",
            dropout=dropout,
            window_size=window_size,
            hyper_hidden_dim=hyper_hidden_dim,
            hyper_init=hyper_init,
            hyper_init_noise=hyper_init_noise,
            batched=batched,
        )

    def forward(self, data):
        batch = getattr(data, "batch", None)
        return self.model(data.x, data.edge_index, batch)
