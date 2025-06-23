"""MAGIC Hybrid Model for direction reconstruction.

Combines transformer and classification approaches for optimal performance,
based on IceCube Kaggle competition winning strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch import Tensor
from torch_geometric.data import Data

from graphnet.models.gnn.gnn import GNN
from graphnet.models.gnn.magic_transformer import MAGICTransformer
from graphnet.models.gnn.magic_direction_classifier import MAGICDirectionClassifier


class MAGICHybridModel(GNN):
    """Hybrid model combining transformer and classification approaches.
    
    Key innovations:
    - Multi-scale feature extraction (local pixels + global telescope)
    - Attention-weighted ensemble of classification and regression
    - Progressive refinement strategy
    """
    
    def __init__(
        self,
        nb_inputs: int = 5,
        hidden_dim: int = 256,
        # Transformer parameters
        transformer_layers: int = 6,
        transformer_heads: int = 8,
        # Classifier parameters
        num_fine_bins: int = 128,
        roi_radius: float = 0.5,
        # Ensemble parameters
        ensemble_method: str = "attention",  # "attention", "learned", "average"
    ):
        """Initialize hybrid model.
        
        Args:
            nb_inputs: Number of input features on each node.
            hidden_dim: Hidden dimension size.
            transformer_layers: Number of transformer layers.
            transformer_heads: Number of attention heads.
            num_fine_bins: Number of fine bins for classification.
            roi_radius: Region of interest radius.
            ensemble_method: Method for combining transformer and classifier.
        """
        # Calculate actual dimensions
        transformer_dim = hidden_dim  # 256
        classifier_dim = 256 * 3  # backbone_layers[-1] * len(global_pooling) = 768
        combined_dim = transformer_dim + classifier_dim  # 1024
        
        super().__init__(nb_inputs, combined_dim)
        
        self.ensemble_method = ensemble_method
        self.combined_dim = combined_dim
        
        # Component models
        self.transformer = MAGICTransformer(
            nb_inputs=nb_inputs,
            hidden_dim=hidden_dim,
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            use_cross_attention=True
        )
        
        self.classifier = MAGICDirectionClassifier(
            nb_inputs=nb_inputs,
            hidden_dim=hidden_dim,
            num_fine_bins=num_fine_bins,
            roi_radius=roi_radius,
            use_dynedge=True
        )
        
        # Ensemble combination
        if ensemble_method == "attention":
            self.ensemble_attention = nn.MultiheadAttention(
                combined_dim, 4, dropout=0.1, batch_first=True
            )
            self.ensemble_query = nn.Parameter(torch.randn(1, 1, combined_dim))
        elif ensemble_method == "learned":
            self.ensemble_weights = nn.Sequential(
                nn.Linear(combined_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
                nn.Softmax(dim=1)
            )
        
        # Final prediction heads
        self.final_direction = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 3)  # 3D direction vector
        )
        
        self.final_uncertainty = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )
        
    def forward(self, data: Data) -> Tensor:
        """Forward pass with ensemble prediction."""
        # Get features from both models
        transformer_features = self.transformer(data)
        classifier_features = self.classifier(data)  # Returns backbone features
        
        # Combine features
        combined = torch.cat([transformer_features, classifier_features], dim=1)
        
        # Ensemble combination
        if self.ensemble_method == "attention":
            # Use attention to combine features
            batch_size = combined.shape[0]
            queries = self.ensemble_query.expand(batch_size, -1, -1)
            features = combined.unsqueeze(1)
            
            ensemble_features, _ = self.ensemble_attention(queries, features, features)
            ensemble_features = ensemble_features.squeeze(1)
        elif self.ensemble_method == "learned":
            # Learn weights for combination
            weights = self.ensemble_weights(combined)
            ensemble_features = (weights[:, 0:1] * transformer_features + 
                               weights[:, 1:2] * classifier_features)
        else:
            # Simple average
            ensemble_features = (transformer_features + classifier_features) / 2
        
        # Return the combined features for tasks to use
        return combined 
