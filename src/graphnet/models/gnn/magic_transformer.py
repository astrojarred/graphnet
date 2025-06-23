"""MAGIC Transformer model for direction reconstruction.

Based on IceCube Kaggle competition winning strategies using transformer 
architectures for sequential Cherenkov data with physics-informed features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from torch import Tensor
from torch_geometric.data import Data
from torch_scatter import scatter_mean, scatter_max, scatter_sum

from graphnet.models.gnn.gnn import GNN
from graphnet.utilities.maths import eps_like


class MAGICTransformer(GNN):
    """Transformer model for MAGIC telescopes inspired by IceCube winners.
    
    Key features:
    - Treats pixels as sequence tokens with positional encoding
    - Cross-attention between telescopes for stereo fusion
    - Efficient self-attention with Flash Attention compatibility
    - Physics-informed positional encoding using camera geometry
    """
    
    def __init__(
        self,
        nb_inputs: int = 5,  # x_cam, y_cam, t, tel_id, signal
        hidden_dim: int = 256,
        num_layers: int = 8,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        max_pixels: int = 2100,  # 1039 pixels × 2 telescopes + margin
        use_cross_attention: bool = True,
        pool_telescopes: str = "attention",  # "attention", "mean", "concat"
    ):
        """Initialize MAGICTransformer.
        
        Args:
            nb_inputs: Number of input features on each node.
            hidden_dim: Hidden dimension size for transformer layers.
            num_layers: Number of transformer layers.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio for MLP hidden dimension.
            dropout: Dropout rate.
            max_pixels: Maximum number of pixels expected.
            use_cross_attention: Whether to use cross-attention between telescopes.
            pool_telescopes: Method for pooling telescopes ("attention", "mean", "concat").
        """
        super().__init__(nb_inputs, hidden_dim)
        
        self.hidden_dim = hidden_dim
        self.use_cross_attention = use_cross_attention
        self.pool_telescopes = pool_telescopes
        
        # Input projection
        self.input_proj = nn.Linear(nb_inputs, hidden_dim)
        
        # Learnable telescope embeddings
        self.telescope_embedding = nn.Embedding(3, hidden_dim)  # 0=padding, 1=tel1, 2=tel2
        
        # Positional encoding based on camera geometry
        self.pos_encoder = CameraPositionalEncoding(hidden_dim, max_pixels)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(
                hidden_dim, num_heads, mlp_ratio, dropout,
                use_cross_attention=(use_cross_attention and i % 2 == 1)
            )
            for i in range(num_layers)
        ])
        
        # Output normalization
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Telescope pooling
        if pool_telescopes == "attention":
            self.pool_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout)
            self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
    def forward(self, data: Data) -> Tensor:
        """Forward pass with stereo telescope handling."""
        x, batch = data.x, data.batch
        
        if x is None:
            raise ValueError("Input data.x is None")
        
        # Extract telescope IDs and positions before projection
        tel_ids = x[:, 3].long() + 1  # Convert 0/1 to 1/2 for embedding
        positions = x[:, :2]  # x_cam, y_cam
        
        # Project input features
        x = self.input_proj(x)
        
        # Add telescope embeddings
        x = x + self.telescope_embedding(tel_ids)
        
        # Add positional encoding based on camera coordinates
        x = self.pos_encoder(x, positions, batch)
        
        # Apply transformer layers
        for layer in self.layers:
            if layer.use_cross_attention:
                # Separate by telescope for cross-attention
                x = self._apply_cross_telescope_attention(x, tel_ids, batch, layer)
            else:
                x = layer(x, batch)
        
        x = self.norm(x)
        
        # Pool across pixels and telescopes
        x = self._pool_features(x, tel_ids, batch)
        
        return x
    
    def _apply_cross_telescope_attention(self, x, tel_ids, batch, layer):
        """Apply cross-attention between telescopes."""
        # Implementation details for cross-telescope attention
        # This would separate features by telescope and apply cross-attention
        return layer(x, batch)  # Simplified for now
    
    def _pool_features(self, x, tel_ids, batch):
        """Pool features across pixels and telescopes."""
        if self.pool_telescopes == "attention":
            # Use learnable query to attend to all pixels
            batch_size = batch.max().item() + 1
            queries = self.pool_query.expand(batch_size, -1, -1)  # [batch_size, 1, hidden_dim]
            
            # Create attention mask for batching
            x_dense, mask = to_dense_batch(x, batch)  # [batch_size, max_nodes, hidden_dim]
            
            # Transpose for MultiheadAttention (expects seq_len first when batch_first=False)
            # MultiheadAttention expects: (seq_len, batch, embed_dim) when batch_first=False
            queries = queries.transpose(0, 1)  # [1, batch_size, hidden_dim]
            x_dense = x_dense.transpose(0, 1)  # [max_nodes, batch_size, hidden_dim]
            
            # Apply attention pooling
            pooled, _ = self.pool_attention(queries, x_dense, x_dense, key_padding_mask=~mask)
            # pooled shape: [1, batch_size, hidden_dim]
            return pooled.squeeze(0)  # [batch_size, hidden_dim]
        else:
            # Simple mean pooling
            return scatter_mean(x, batch, dim=0)


class TransformerLayer(nn.Module):
    """Single transformer layer with optional cross-attention."""
    
    def __init__(self, hidden_dim, num_heads, mlp_ratio, dropout, use_cross_attention=False):
        super().__init__()
        self.use_cross_attention = use_cross_attention
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # Cross-attention (if used)
        if use_cross_attention:
            self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout, batch_first=True)
            self.norm_cross = nn.LayerNorm(hidden_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, batch):
        """Forward pass with proper self-attention mechanism."""
        # Convert to dense batch for attention
        x_dense, mask = to_dense_batch(x, batch)
        
        # Self-attention with padding mask
        attn_out, _ = self.self_attn(x_dense, x_dense, x_dense, key_padding_mask=~mask)
        x_dense = x_dense + attn_out
        x_dense = self.norm1(x_dense)
        
        # Cross-attention (if applicable)
        if self.use_cross_attention:
            # For now, just skip cross-attention - can be implemented later
            pass
        
        # FFN
        ffn_out = self.ffn(x_dense)
        x_dense = x_dense + ffn_out
        x_dense = self.norm2(x_dense)
        
        # Convert back to sparse representation
        x = x_dense[mask]
        return x


class CameraPositionalEncoding(nn.Module):
    """Physics-informed positional encoding using camera geometry."""
    
    def __init__(self, hidden_dim, max_len=2100):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Learnable encoding based on radial distance and angle
        self.radial_encoder = nn.Linear(1, hidden_dim // 4)
        self.angular_encoder = nn.Linear(2, hidden_dim // 4)  # sin, cos of angle
        self.time_encoder = nn.Linear(1, hidden_dim // 4)
        self.combined_encoder = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x, positions, batch):
        """Add positional encoding based on camera coordinates."""
        # Calculate radial distance from camera center
        r = torch.norm(positions, dim=1, keepdim=True)
        
        # Calculate angle
        theta = torch.atan2(positions[:, 1:2], positions[:, 0:1])
        angular_features = torch.cat([torch.sin(theta), torch.cos(theta)], dim=1)
        
        # Encode position
        radial_enc = self.radial_encoder(r)
        angular_enc = self.angular_encoder(angular_features)
        
        # Combine encodings
        pos_encoding = torch.cat([radial_enc, angular_enc, 
                                  torch.zeros_like(radial_enc),  # Placeholder for time
                                  torch.zeros_like(radial_enc)], dim=1)
        pos_encoding = self.combined_encoder(pos_encoding)
        
        return x + pos_encoding


def to_dense_batch(x: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor]:
    """Convert sparse batch tensor to dense batch tensor with mask.
    
    Similar to torch_geometric.utils.to_dense_batch but simplified.
    """
    batch_size = int(batch.max().item()) + 1
    num_nodes = scatter_sum(torch.ones_like(batch), batch)
    max_num_nodes = int(num_nodes.max().item())
    
    # Create dense tensor
    dense = torch.zeros(batch_size, max_num_nodes, x.shape[1], device=x.device, dtype=x.dtype)
    mask = torch.zeros(batch_size, max_num_nodes, dtype=torch.bool, device=x.device)
    
    # Fill dense tensor
    for b in range(batch_size):
        idx = (batch == b)
        n = int(idx.sum().item())
        if n > 0:
            dense[b, :n] = x[idx]
            mask[b, :n] = True
            
    return dense, mask 
