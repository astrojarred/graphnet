"""MAGIC-specific IceMix architecture adapted from IceCube 2nd place solution.

Implementation of IceMix transformer architecture adapted for MAGIC telescope
direction reconstruction. Based on the winning IceCube solution but modified
for MAGIC's dual-telescope stereo data structure.

Original IceCube solution: https://github.com/DrHB/icecube-2nd-place
"""
import torch
import torch.nn as nn
from typing import Set, Dict, Any, Optional

from graphnet.models.components.layers import (
    Block_rel,
    Block,
)
from graphnet.models.components.embedding import (
    FourierEncoderMAGIC,
    SpacetimeEncoderMAGIC,
)
from graphnet.models.gnn.dynedge import DynEdge
from graphnet.models.gnn.gnn import GNN
from graphnet.models.utils import array_to_sequence

from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Data
from torch import Tensor


class MAGICIceMix(GNN):
    """MAGIC IceMix model - transformer architecture for dual-telescope direction reconstruction."""

    def __init__(
        self,
        hidden_dim: int = 384,
        mlp_ratio: int = 4,
        seq_length: int = 512,
        depth: int = 12,
        head_size: int = 32,
        depth_rel: int = 4,
        n_rel: int = 1,
        scaled_emb: bool = False,
        include_dynedge: bool = False,
        dynedge_args: Optional[Dict[str, Any]] = None,
        n_features: int = 7,
        tel_id_feature_idx: int = 3,
        exclude_tel_id_from_processing: bool = True,
        dual_telescope_processing: bool = True,
    ):
        """Construct `MAGICIceMix`.

        Args:
            hidden_dim: The latent feature dimension.
            mlp_ratio: Mlp expansion ratio of FourierEncoder and Transformer.
            seq_length: The maximum sequence length for each telescope.
            depth: The depth of the transformer.
            head_size: The size of the attention heads.
            depth_rel: The depth of the relative transformer.
            n_rel: The number of relative transformer layers to use.
            scaled_emb: Whether to scale the sinusoidal positional embeddings.
            include_dynedge: If True, pulse-level predictions from `DynEdge`
                will be added as features to the model.
            dynedge_args: Initialization arguments for DynEdge. If not
                provided, DynEdge will be initialized with MAGIC-optimized settings.
            n_features: The number of features in the input data (7 for MAGIC).
            tel_id_feature_idx: Index of telescope ID in features (default: 3).
            exclude_tel_id_from_processing: Whether to exclude tel_id from processing.
            dual_telescope_processing: Whether to process telescopes separately then fuse.
        """
        super().__init__(seq_length, hidden_dim)
        
        # Store MAGIC-specific parameters
        self._tel_id_feature_idx = tel_id_feature_idx
        self._exclude_tel_id_from_processing = exclude_tel_id_from_processing
        self._dual_telescope_processing = dual_telescope_processing
        
        # Calculate effective features (excluding tel_id if specified)
        effective_features = n_features - 1 if exclude_tel_id_from_processing else n_features
        
        # Fourier encoder output dimension (halved if including DynEdge)
        fourier_out_dim = hidden_dim // 2 if include_dynedge else hidden_dim
        
        self.fourier_ext = FourierEncoderMAGIC(
            seq_length=seq_length,
            mlp_dim=None,
            output_dim=fourier_out_dim,
            scaled=scaled_emb,
            n_features=effective_features,
        )
        
        # Spacetime encoder for relative bias (adapted for MAGIC's 2D+time structure)
        self.rel_pos = SpacetimeEncoderMAGIC(head_size)
        
        # Relative attention layers (physics-informed local processing)
        self.sandwich = nn.ModuleList([
            Block_rel(
                input_dim=hidden_dim, 
                num_heads=hidden_dim // head_size
            )
            for _ in range(depth_rel)
        ])
        
        # CLS token for global event representation
        self.cls_token = nn.Linear(hidden_dim, 1, bias=False)
        
        # Standard transformer blocks (global attention)
        self.blocks = nn.ModuleList([
            Block(
                input_dim=hidden_dim,
                num_heads=hidden_dim // head_size,
                mlp_ratio=mlp_ratio,
                drop_path=0.0 * (i / (depth - 1)),
                init_values=1,
            )
            for i in range(depth)
        ])
        
        self.n_rel = n_rel
        self.include_dynedge = include_dynedge

        # Optional DynEdge for hybrid GNN+transformer approach
        if include_dynedge:
            if dynedge_args is None:
                # MAGIC-optimized DynEdge settings
                self.warning_once("Running with MAGIC-optimized DynEdge settings")
                self.dyn_edge = DynEdge(
                    nb_inputs=effective_features,
                    nb_neighbours=12,  # Increased for MAGIC's hexagonal geometry
                    post_processing_layer_sizes=[256, hidden_dim // 2],
                    dynedge_layer_sizes=[
                        (128, 256),
                        (256, 256),
                        (256, 256),
                    ],
                    global_pooling_schemes=None,
                    activation_layer="gelu",
                    add_norm_layer=True,
                    skip_readout=True,
                )
            else:
                self.dyn_edge = DynEdge(**dynedge_args)

        # Telescope fusion layer for dual processing
        if dual_telescope_processing:
            self.telescope_fusion = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=hidden_dim // head_size,
                dropout=0.1,
                batch_first=True
            )

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        """cls_token should not be subject to weight decay during training."""
        return {"cls_token"}

    def _split_by_telescope(self, data: Data) -> tuple[Data, Data]:
        """Split data by telescope ID for separate processing."""
        tel_id = data.x[:, self._tel_id_feature_idx]
        
        # Create masks for each telescope
        tel1_mask = (tel_id == 0)
        tel2_mask = (tel_id == 1)
        
        # Remove tel_id from features if specified
        if self._exclude_tel_id_from_processing:
            features_to_keep = [i for i in range(data.x.shape[1]) if i != self._tel_id_feature_idx]
            x_clean = data.x[:, features_to_keep]
        else:
            x_clean = data.x
            
        # Split data
        tel1_x = x_clean[tel1_mask]
        tel2_x = x_clean[tel2_mask]
        tel1_batch = data.batch[tel1_mask]
        tel2_batch = data.batch[tel2_mask]
        
        # Create Data objects for each telescope
        tel1_data = Data(x=tel1_x, batch=tel1_batch)
        tel2_data = Data(x=tel2_x, batch=tel2_batch)
        
        return tel1_data, tel2_data

    def _process_telescope(self, tel_data: Data) -> Tensor:
        """Process a single telescope's data through the transformer."""
        # Convert to sequence format
        x0, mask, seq_length = array_to_sequence(
            tel_data.x, tel_data.batch, padding_value=0
        )
        
        # Fourier encoding
        x = self.fourier_ext(x0, seq_length)
        
        # Relative position bias for physics-informed attention
        rel_pos_bias = self.rel_pos(x0)
        batch_size = mask.shape[0]
        
        # Optional DynEdge features
        if self.include_dynedge:
            # Create full data object for DynEdge (needs edge_index)
            full_data = Data(x=tel_data.x, batch=tel_data.batch)
            graph = self.dyn_edge(full_data)
            graph, _ = to_dense_batch(graph, tel_data.batch)
            x = torch.cat([x, graph], 2)

        # Attention mask for padding
        attn_mask = torch.zeros(mask.shape, device=mask.device)
        attn_mask[~mask] = -torch.inf

        # Relative attention layers (physics-informed local processing)
        for i, blk in enumerate(self.sandwich):
            x = blk(x, attn_mask, rel_pos_bias)
            if i + 1 == self.n_rel:
                rel_pos_bias = None

        # Add CLS token
        mask = torch.cat([
            torch.ones(batch_size, 1, dtype=mask.dtype, device=mask.device),
            mask,
        ], 1)
        attn_mask = torch.zeros(mask.shape, device=mask.device)
        attn_mask[~mask] = -torch.inf
        cls_token = self.cls_token.weight.unsqueeze(0).expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], 1)

        # Global transformer layers
        for blk in self.blocks:
            x = blk(x, None, attn_mask)

        return x[:, 0]  # Return CLS token

    def forward(self, data: Data) -> Tensor:
        """Apply learnable forward pass."""
        if self._dual_telescope_processing:
            # Process each telescope separately
            tel1_data, tel2_data = self._split_by_telescope(data)
            
            # Get representations from each telescope
            tel1_repr = self._process_telescope(tel1_data)
            tel2_repr = self._process_telescope(tel2_data)
            
            # Fuse telescope representations using cross-attention
            # Shape: [batch_size, 1, hidden_dim]
            tel1_repr_expanded = tel1_repr.unsqueeze(1)
            tel2_repr_expanded = tel2_repr.unsqueeze(1)
            
            # Cross-attention: M1 attends to M2 and vice versa
            tel1_attended, _ = self.telescope_fusion(
                tel1_repr_expanded, tel2_repr_expanded, tel2_repr_expanded
            )
            tel2_attended, _ = self.telescope_fusion(
                tel2_repr_expanded, tel1_repr_expanded, tel1_repr_expanded
            )
            
            # Combine attended representations
            fused_repr = (tel1_attended.squeeze(1) + tel2_attended.squeeze(1)) / 2
            
            return fused_repr
        else:
            # Process as single combined sequence (original IceMix approach)
            # Remove tel_id if specified
            if self._exclude_tel_id_from_processing:
                features_to_keep = [i for i in range(data.x.shape[1]) if i != self._tel_id_feature_idx]
                data.x = data.x[:, features_to_keep]
            
            # Standard IceMix processing
            x0, mask, seq_length = array_to_sequence(
                data.x, data.batch, padding_value=0
            )
            x = self.fourier_ext(x0, seq_length)
            rel_pos_bias = self.rel_pos(x0)
            batch_size = mask.shape[0]
            
            if self.include_dynedge:
                graph = self.dyn_edge(data)
                graph, _ = to_dense_batch(graph, data.batch)
                x = torch.cat([x, graph], 2)

            attn_mask = torch.zeros(mask.shape, device=mask.device)
            attn_mask[~mask] = -torch.inf

            for i, blk in enumerate(self.sandwich):
                x = blk(x, attn_mask, rel_pos_bias)
                if i + 1 == self.n_rel:
                    rel_pos_bias = None

            mask = torch.cat([
                torch.ones(batch_size, 1, dtype=mask.dtype, device=mask.device),
                mask,
            ], 1)
            attn_mask = torch.zeros(mask.shape, device=mask.device)
            attn_mask[~mask] = -torch.inf
            cls_token = self.cls_token.weight.unsqueeze(0).expand(batch_size, -1, -1)
            x = torch.cat([cls_token, x], 1)

            for blk in self.blocks:
                x = blk(x, None, attn_mask)

            return x[:, 0]