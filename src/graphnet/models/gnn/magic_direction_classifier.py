"""MAGIC Direction Classifier model for direction reconstruction.

Based on IceCube Kaggle competition winning strategies using fine-grained
angular classification with Von Mises-Fisher distribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from torch import Tensor
from torch_geometric.data import Data
from torch_scatter import scatter_mean, scatter_max, scatter_sum

from graphnet.models.gnn.gnn import GNN
from graphnet.models.components.layers import DynEdgeConv
from graphnet.utilities.maths import eps_like


class MAGICDirectionClassifier(GNN):
    """Direction classifier using fine bins near pointing direction.
    
    Based on IceCube 3rd place solution:
    - Fine angular bins (128+) for region of interest
    - VMF-inspired angular representation
    - Hybrid classification + regression approach
    """
    
    def __init__(
        self,
        nb_inputs: int = 5,
        hidden_dim: int = 256,
        num_fine_bins: int = 128,  # Fine bins for ROI
        roi_radius: float = 0.5,   # Region of interest in degrees
        num_coarse_bins: int = 8,  # Coarse bins outside ROI
        backbone_layers: Optional[List[int]] = None,
        use_dynedge: bool = True,
        global_pooling: Optional[List[str]] = None,
    ):
        """Initialize direction classifier.
        
        Args:
            nb_inputs: Number of input features on each node.
            hidden_dim: Hidden dimension size.
            num_fine_bins: Number of fine bins for region of interest.
            roi_radius: Region of interest radius in degrees.
            num_coarse_bins: Number of coarse bins outside ROI.
            backbone_layers: Hidden layer sizes for feature extraction.
            use_dynedge: Whether to use DynEdge convolutions.
            global_pooling: List of global pooling schemes.
        """
        if backbone_layers is None:
            backbone_layers = [256, 512, 512, 256]
        if global_pooling is None:
            global_pooling = ["mean", "max", "sum"]
            
        backbone_output_dim = backbone_layers[-1] * len(global_pooling)
        # Output is just the backbone features, tasks handle their own heads
        super().__init__(nb_inputs, backbone_output_dim)
        
        self.roi_radius = roi_radius
        self.num_fine_bins = num_fine_bins
        self.num_coarse_bins = num_coarse_bins
        self.total_bins = num_fine_bins + num_coarse_bins
        self.global_pooling = global_pooling
        
        # Feature extraction backbone
        if use_dynedge:
            self.backbone = SimplifiedDynEdge(
                nb_inputs, backbone_layers, global_pooling
            )
        else:
            self.backbone = MLPBackbone(nb_inputs, backbone_layers)
        
        # Classification head for angular bins
        self.classifier = nn.Sequential(
            nn.Linear(backbone_output_dim, backbone_output_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(backbone_output_dim, self.total_bins)
        )
        
        # VMF regression head for continuous direction
        self.vmf_head = nn.Sequential(
            nn.Linear(backbone_output_dim, backbone_output_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(backbone_output_dim, 3)  # 3D unit vector
        )
        
        # Uncertainty estimation head
        self.kappa_head = nn.Sequential(
            nn.Linear(backbone_output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Softplus()  # Ensure positive kappa
        )
        
    def forward(self, data: Data) -> Tensor:
        """Forward pass returning feature representation for tasks."""
        # Extract features from backbone and return them
        # Let the tasks handle their own prediction heads
        features = self.backbone(data)
        return features
    
    def get_bin_edges(self, pointing_direction):
        """Generate bin edges based on pointing direction."""
        # Fine bins within ROI
        fine_edges = torch.linspace(0, self.roi_radius, self.num_fine_bins + 1)
        
        # Coarse bins outside ROI
        coarse_edges = torch.linspace(self.roi_radius, 3.5, self.num_coarse_bins)
        
        return torch.cat([fine_edges, coarse_edges[1:]])


class SimplifiedDynEdge(nn.Module):
    """Simplified DynEdge backbone for feature extraction."""
    
    def __init__(self, nb_inputs, layer_sizes, global_pooling):
        super().__init__()
        self.global_pooling = global_pooling
        
        # Dynamic edge convolutions
        self.conv_layers = nn.ModuleList()
        in_features = nb_inputs
        
        for out_features in layer_sizes:
            conv = DynEdgeConv(
                nn.Sequential(
                    nn.Linear(in_features * 2, out_features),
                    nn.ReLU(),
                    nn.BatchNorm1d(out_features)
                ),
                aggr="add",
                nb_neighbors=16,
                features_subset=slice(0, 3)  # Use x, y, t for edges
            )
            self.conv_layers.append(conv)
            in_features = out_features
        
    def forward(self, data: Data) -> Tensor:
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        
        assert x is not None, "Input features cannot be None"
        assert batch is not None, "Batch indices cannot be None"
        
        # Apply convolutions
        for conv in self.conv_layers:
            x, edge_index = conv(x, edge_index, batch)
        
        # Global pooling
        pooled = []
        for pool_type in self.global_pooling:
            if pool_type == "mean":
                pooled.append(scatter_mean(x, batch, dim=0))
            elif pool_type == "max":
                pooled.append(scatter_max(x, batch, dim=0)[0])
            elif pool_type == "sum":
                pooled.append(scatter_sum(x, batch, dim=0))
        
        return torch.cat(pooled, dim=1) if pooled else x


class MLPBackbone(nn.Module):
    """Simple MLP backbone for feature extraction."""
    
    def __init__(self, nb_inputs, layer_sizes):
        super().__init__()
        
        layers = []
        in_features = nb_inputs
        
        for out_features in layer_sizes:
            layers.extend([
                nn.Linear(in_features, out_features),
                nn.ReLU(),
                nn.BatchNorm1d(out_features),
                nn.Dropout(0.2)
            ])
            in_features = out_features
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, data: Data) -> Tensor:
        x = data.x
        batch = data.batch
        
        assert x is not None, "Input features cannot be None"
        assert batch is not None, "Batch indices cannot be None"
        
        # Apply MLP
        x = self.mlp(x)
        
        # Global mean pooling
        return scatter_mean(x, batch, dim=0) 
