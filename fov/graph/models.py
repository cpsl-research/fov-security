from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from torch_geometric.data import Data
    from avstack.geometry import ReferenceFrame
    from avstack.modules.perception.fov_estimator import _LidarFovEstimator

import torch
from avstack.config import MODELS
from torch.nn import Sigmoid
from torch_geometric.nn.models import GAT


@MODELS.register_module()
class GATModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.GAT = GAT(**kwargs)
        self.sigmoid = Sigmoid()

    def forward(self, data: "Data"):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        logits = self.GAT(x, edge_index, edge_attr)
        return self.sigmoid(logits).squeeze()


class AVstackGraphWrapper:
    def __init__(self, model: "_LidarFovEstimator"):
        self.model = model

    def __call__(self, data: "Data", reference: "ReferenceFrame", bev: bool = True):
        """Wrapper to classic AVstack fov polygon estimators"""
        if bev:
            return self.model._execute_on_bev_array(data.x, reference)
        else:
            return self.model._execute_on_array(data.x, reference)

    def eval(self):
        pass
