"""A Stereo DynEdge model designed for cherenkov telescope pairs."""

from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor, LongTensor
from torch_geometric.data import Data
from torch_scatter import scatter_max, scatter_mean, scatter_min, scatter_sum

from graphnet.models.components.layers import DynEdgeConv
from graphnet.models.gnn.gnn import GNN
from graphnet.models.utils import calculate_xyzt_homophily

GLOBAL_POOLINGS = {
    "min": scatter_min,
    "max": scatter_max,
    "sum": scatter_sum,
    "mean": scatter_mean,
}


class CrossAttentionLayer(torch.nn.Module):
    """Cross-attention layer for stereo telescope fusion."""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        """Initialize cross-attention layer.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.k_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.v_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim)
        
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(embed_dim)
        
    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """Apply cross-attention.
        
        Args:
            query: Query tensor [batch_size, seq_len_q, embed_dim]
            key: Key tensor [batch_size, seq_len_k, embed_dim]
            value: Value tensor [batch_size, seq_len_v, embed_dim]
            
        Returns:
            Attended features [batch_size, seq_len_q, embed_dim]
        """
        batch_size, seq_len_q, _ = query.size()
        seq_len_k = key.size(1)
        
        # Project to Q, K, V
        Q = self.q_proj(query).view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.embed_dim)
        
        # Output projection and residual connection
        output = self.out_proj(attended)
        output = self.layer_norm(output + query)  # Residual connection
        
        return output


class DynEdgeStereo(GNN):
    """Stereo DynEdge model for dual-telescope Cherenkov data."""
    
    def __init__(
        self,
        nb_inputs: int,
        *,
        nb_neighbours: int = 8,
        features_subset: Optional[Union[List[int], slice]] = None,
        dynedge_layer_sizes: Optional[List[Tuple[int, ...]]] = None,
        post_processing_layer_sizes: Optional[List[int]] = None,
        readout_layer_sizes: Optional[List[int]] = None,
        global_pooling_schemes: Optional[Union[str, List[str]]] = None,
        add_global_variables_after_pooling: bool = False,
        activation_layer: Optional[str] = None,
        add_norm_layer: bool = False,
        dynedge_dropout: Optional[float] = None,
        # Stereo-specific parameters
        cross_attention_heads: int = 8,
        cross_attention_layers: int = 2,
        cross_attention_dropout: float = 0.1,
        fusion_strategy: str = "multi_stage",  # "early", "late", "multi_stage"
        tel_id_feature_idx: int = 3,  # Index of telescope ID in features
        exclude_tel_id_from_processing: bool = True,
        add_stereo_global_features: bool = True,
    ):
        """Construct Stereo DynEdge model.
        
        Args:
            nb_inputs: Number of input features per node
            nb_neighbours: Number of neighbours for k-NN clustering
            features_subset: Subset of features for k-NN distance calculation
            dynedge_layer_sizes: Layer sizes for DynEdge convolutions
            post_processing_layer_sizes: Layer sizes for post-processing
            readout_layer_sizes: Layer sizes for readout
            global_pooling_schemes: Global pooling schemes to use
            add_global_variables_after_pooling: Whether to add global vars after pooling
            activation_layer: Activation function to use
            add_norm_layer: Whether to add normalization layers
            cross_attention_heads: Number of attention heads for cross-attention
            cross_attention_layers: Number of cross-attention layers
            cross_attention_dropout: Dropout rate for cross-attention
            fusion_strategy: When to apply cross-attention ("early", "late", "multi_stage")
            tel_id_feature_idx: Index of telescope ID feature
            exclude_tel_id_from_processing: Whether to exclude tel_id from processing
            add_stereo_global_features: Whether to add stereo-specific global features
        """
        # Set default parameters
        if features_subset is None:
            # Exclude tel_id from distance calculation (assuming it's at index 3)
            features_subset = [0, 1, 2]  # x_cam, y_cam, t
            
        if dynedge_layer_sizes is None:
            dynedge_layer_sizes = [
                (128, 256),
                (336, 256),
                (336, 256),
                (336, 256),
            ]
            
        if post_processing_layer_sizes is None:
            post_processing_layer_sizes = [336, 256]
            
        if readout_layer_sizes is None:
            readout_layer_sizes = [128]
            
        if global_pooling_schemes is None:
            global_pooling_schemes = ["min", "max", "mean", "sum"]
        elif isinstance(global_pooling_schemes, str):
            global_pooling_schemes = [global_pooling_schemes]
            
        # Store activation layer type (not the instance yet)
        self._activation_type = activation_layer
        
        # Store parameters
        self._nb_inputs = nb_inputs
        self._nb_neighbours = nb_neighbours
        self._features_subset = features_subset
        
        # Validate and convert dynedge_layer_sizes
        if isinstance(dynedge_layer_sizes, list):
            # Convert lists to tuples and validate
            converted_sizes = []
            for sizes in dynedge_layer_sizes:
                if isinstance(sizes, (list, tuple)):
                    converted_sizes.append(tuple(sizes))
                else:
                    raise ValueError(f"Invalid dynedge_layer_sizes format: {sizes}")
            self._dynedge_layer_sizes = converted_sizes
        else:
            self._dynedge_layer_sizes = dynedge_layer_sizes
            
        self._post_processing_layer_sizes = post_processing_layer_sizes
        self._readout_layer_sizes = readout_layer_sizes
        self._global_pooling_schemes = global_pooling_schemes
        self._add_global_variables_after_pooling = add_global_variables_after_pooling
        self._add_norm_layer = add_norm_layer
        self._dynedge_dropout = dynedge_dropout
        
        # Stereo-specific parameters
        self._cross_attention_heads = cross_attention_heads
        self._num_cross_attention_layers = cross_attention_layers  # FIXED: renamed to avoid collision
        self._cross_attention_dropout = cross_attention_dropout
        self._fusion_strategy = fusion_strategy
        self._tel_id_feature_idx = tel_id_feature_idx
        self._exclude_tel_id_from_processing = exclude_tel_id_from_processing
        self._add_stereo_global_features = add_stereo_global_features
        
        # Calculate effective input size (exclude tel_id if specified)
        self._effective_nb_inputs = nb_inputs - 1 if exclude_tel_id_from_processing else nb_inputs
        
        # Calculate global variables size for stereo model
        # Basic telescope globals: effective_nb_inputs (mean features) + 4 (homophily)
        telescope_globals_size = self._effective_nb_inputs + 4
        # Stereo globals: time_diff(1) + signal_ratio(1) + pointing_diff(2) = 4
        stereo_features_size = 4 if add_stereo_global_features else 0
        # Total: 2 telescopes + stereo features
        self._nb_global_variables = 2 * telescope_globals_size + stereo_features_size
            
        # Base class constructor
        super().__init__(nb_inputs, self._readout_layer_sizes[-1])
        
        # Create activation layer after parent constructor
        if self._activation_type is None or self._activation_type.lower() == "relu":
            self._activation = torch.nn.ReLU()
        elif self._activation_type.lower() == "gelu":
            self._activation = torch.nn.GELU()
        else:
            raise ValueError(f"Activation layer {self._activation_type} not supported.")
            
        self._construct_layers()
        
    def _construct_layers(self) -> None:
        """Construct all layers for the stereo model."""
        # Build DynEdge layers for each telescope
        self._construct_telescope_layers()
        
        # Build cross-attention layers
        self._construct_cross_attention_layers()
        
        # Build post-processing and readout layers
        self._construct_fusion_layers()
        
    def _construct_telescope_layers(self) -> None:
        """Construct DynEdge layers for individual telescopes."""
        # Input features (possibly excluding tel_id)
        nb_input_features = self._effective_nb_inputs
        if not self._add_global_variables_after_pooling:
            # For stereo model, we need to calculate the actual global variables size
            # This will be determined at runtime, so we'll use a placeholder for now
            # and fix it in the forward pass
            nb_input_features += self._nb_global_variables
            
        
        # Telescope 1 (M1) layers
        self._tel1_conv_layers = torch.nn.ModuleList()
        # Telescope 2 (M2) layers  
        self._tel2_conv_layers = torch.nn.ModuleList()
        
        nb_latent_features = nb_input_features
        for sizes in self._dynedge_layer_sizes:
            # Create identical layers for both telescopes
            tel1_layers = []
            tel2_layers = []
            
            layer_sizes = [nb_latent_features] + list(sizes)
            for ix, (nb_in, nb_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
                if ix == 0:
                    nb_in *= 2
                    
                # Telescope 1 layers
                tel1_layers.append(torch.nn.Linear(nb_in, nb_out))
                if self._add_norm_layer:
                    tel1_layers.append(torch.nn.LayerNorm(nb_out))
                tel1_layers.append(self._activation)
                if self._dynedge_dropout is not None:
                    tel1_layers.append(torch.nn.Dropout(self._dynedge_dropout))
                
                # Telescope 2 layers  
                tel2_layers.append(torch.nn.Linear(nb_in, nb_out))
                if self._add_norm_layer:
                    tel2_layers.append(torch.nn.LayerNorm(nb_out))
                tel2_layers.append(self._activation)
                if self._dynedge_dropout is not None:
                    tel2_layers.append(torch.nn.Dropout(self._dynedge_dropout))
                
            # Create DynEdge convolution layers
            tel1_conv = DynEdgeConv(
                torch.nn.Sequential(*tel1_layers),
                aggr="add",
                nb_neighbors=self._nb_neighbours,
                features_subset=self._features_subset,
            )
            tel2_conv = DynEdgeConv(
                torch.nn.Sequential(*tel2_layers),
                aggr="add", 
                nb_neighbors=self._nb_neighbours,
                features_subset=self._features_subset,
            )
            
            self._tel1_conv_layers.append(tel1_conv)
            self._tel2_conv_layers.append(tel2_conv)
            
            nb_latent_features = nb_out
            
    def _construct_cross_attention_layers(self) -> None:
        """Construct cross-attention layers."""
        if self._fusion_strategy == "early":
            # Cross-attention after each DynEdge layer
            self._cross_attention_layers = torch.nn.ModuleList()
            for layer_sizes in self._dynedge_layer_sizes:
                layer_output_dim = layer_sizes[-1]
                self._cross_attention_layers.append(
                    CrossAttentionLayer(
                        embed_dim=layer_output_dim,
                        num_heads=self._cross_attention_heads,
                        dropout=self._cross_attention_dropout
                    )
                )
        elif self._fusion_strategy in ["late", "multi_stage"]:
            # Cross-attention after all DynEdge layers
            final_layer_dim = self._dynedge_layer_sizes[-1][-1]
            self._late_cross_attention = torch.nn.ModuleList([
                CrossAttentionLayer(
                    embed_dim=final_layer_dim,
                    num_heads=self._cross_attention_heads,
                    dropout=self._cross_attention_dropout
                ) for _ in range(self._num_cross_attention_layers)
            ])
        elif self._fusion_strategy == "concatenation":
            # No cross-attention layers needed for simple concatenation
            pass
            
        if self._fusion_strategy == "multi_stage":
            # Additional early fusion layers
            self._early_cross_attention = torch.nn.ModuleList()
            for sizes in self._dynedge_layer_sizes:
                layer_output_dim = sizes[-1]
                self._early_cross_attention.append(
                    CrossAttentionLayer(
                        embed_dim=layer_output_dim,
                        num_heads=self._cross_attention_heads,
                        dropout=self._cross_attention_dropout
                    )
                )
                
    def _construct_fusion_layers(self) -> None:
        """Construct post-processing and readout layers."""
        # Calculate input size for fusion layers after global pooling
        nb_poolings = len(self._global_pooling_schemes)
        
        if self._fusion_strategy == "multi_stage":
            # Skip connections from all layers, pooled for each telescope
            single_telescope_features = (
                sum(sizes[-1] for sizes in self._dynedge_layer_sizes)  # All layer outputs
                + self._effective_nb_inputs  # Input features
            ) * nb_poolings  # After global pooling
            nb_latent_features = single_telescope_features * 2  # Both telescopes
        else:
            # Only final layer outputs, pooled for each telescope
            single_telescope_features = self._dynedge_layer_sizes[-1][-1] * nb_poolings
            nb_latent_features = single_telescope_features * 2  # Both telescopes
            
        # Post-processing layers
        post_processing_layers = []
        layer_sizes = [nb_latent_features] + list(self._post_processing_layer_sizes)
        for nb_in, nb_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            post_processing_layers.append(torch.nn.Linear(nb_in, nb_out))
            if self._add_norm_layer:
                post_processing_layers.append(torch.nn.LayerNorm(nb_out))
            post_processing_layers.append(self._activation)
            
        self._post_processing = torch.nn.Sequential(*post_processing_layers)
        
        # Readout layers
        nb_latent_features = self._post_processing_layer_sizes[-1]
        if self._add_global_variables_after_pooling:
            nb_latent_features += self._nb_global_variables
            
        readout_layers = []
        layer_sizes = [nb_latent_features] + list(self._readout_layer_sizes)
        for nb_in, nb_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            readout_layers.append(torch.nn.Linear(nb_in, nb_out))
            # add activation to all *except* the final linear
            if nb_out != self._readout_layer_sizes[-1]:
                readout_layers.append(self._activation)
            
        self._readout = torch.nn.Sequential(*readout_layers)
        
    def _split_by_telescope(self, data: Data) -> Tuple[Data, Data]:
        """Split data by telescope ID using vectorized operations for speed."""
        tel_id = data.x[:, self._tel_id_feature_idx]
        
        # Create masks for each telescope
        tel1_mask = (tel_id == 0)
        tel2_mask = (tel_id == 1)
        
        # Split node features (excluding tel_id if specified)
        if self._exclude_tel_id_from_processing:
            # Remove tel_id column
            features_to_keep = [i for i in range(data.x.shape[1]) if i != self._tel_id_feature_idx]
            x_clean = data.x[:, features_to_keep]
        else:
            x_clean = data.x
            
        tel1_x = x_clean[tel1_mask]
        tel2_x = x_clean[tel2_mask]
        
        # Get node indices for each telescope
        tel1_nodes = torch.where(tel1_mask)[0]
        tel2_nodes = torch.where(tel2_mask)[0]
        
        # VECTORIZED: Create mapping tensors instead of Python dictionaries
        device = data.x.device
        max_node_idx = data.x.shape[0]
        
        # Create inverse mapping tensors (old_idx -> new_idx)
        tel1_node_map = torch.full((max_node_idx,), -1, dtype=torch.long, device=device)
        tel2_node_map = torch.full((max_node_idx,), -1, dtype=torch.long, device=device)
        
        tel1_node_map[tel1_nodes] = torch.arange(len(tel1_nodes), device=device)
        tel2_node_map[tel2_nodes] = torch.arange(len(tel2_nodes), device=device) 
        
        # VECTORIZED: Split and remap edge indices using tensor operations
        def remap_edges_vectorized(edge_index, node_mask, node_map_tensor):
            # Find edges where both nodes belong to the telescope
            edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
            tel_edges = edge_index[:, edge_mask]
            
            if tel_edges.shape[1] > 0:
                # VECTORIZED: Remap all edges at once using tensor indexing
                tel_edges_remapped = torch.stack([
                    node_map_tensor[tel_edges[0]],
                    node_map_tensor[tel_edges[1]]
                ])
                return tel_edges_remapped
            else:
                return torch.empty((2, 0), dtype=torch.long, device=device)
                
        tel1_edge_index = remap_edges_vectorized(data.edge_index, tel1_mask, tel1_node_map)
        tel2_edge_index = remap_edges_vectorized(data.edge_index, tel2_mask, tel2_node_map)
        
        # Preserve original batch indices for each telescope
        tel1_batch = data.batch[tel1_mask]
        tel2_batch = data.batch[tel2_mask]
        
        # Calculate n_pulses per batch for each telescope
        from torch_scatter import scatter_add
        # Get unique batch indices to ensure proper tensor size
        max_batch_idx = data.batch.max().item() + 1
        tel1_n_pulses = scatter_add(
            torch.ones_like(tel1_batch), 
            tel1_batch, 
            dim=0,
            dim_size=max_batch_idx
        )
        tel2_n_pulses = scatter_add(
            torch.ones_like(tel2_batch), 
            tel2_batch, 
            dim=0,
            dim_size=max_batch_idx
        )
        
        # Create Data objects for each telescope
        tel1_data = Data(
            x=tel1_x,
            edge_index=tel1_edge_index,
            batch=tel1_batch,
            n_pulses=tel1_n_pulses
        )
        
        tel2_data = Data(
            x=tel2_x,
            edge_index=tel2_edge_index,
            batch=tel2_batch,
            n_pulses=tel2_n_pulses
        )
        
        return tel1_data, tel2_data
        
    def _calculate_stereo_global_variables(
        self,
        tel1_data: Data,
        tel2_data: Data,
        original_data: Data
    ) -> Tensor:
        """Calculate stereo-specific global variables."""
        # Get device from original data to ensure consistency
        device = original_data.x.device
        
        # Basic global variables for each telescope
        tel1_globals = self._calculate_telescope_global_variables(tel1_data, device)
        tel2_globals = self._calculate_telescope_global_variables(tel2_data, device)
        
        # Ensure both global variables are on the correct device
        tel1_globals = tel1_globals.to(device)
        tel2_globals = tel2_globals.to(device)
        
        # Stereo-specific features
        if self._add_stereo_global_features:
            # Time differences between telescopes
            tel1_mean_time = scatter_mean(tel1_data.x[:, 2], tel1_data.batch, dim=0)  # Assuming t is at index 2
            tel2_mean_time = scatter_mean(tel2_data.x[:, 2], tel2_data.batch, dim=0)
            time_diff = tel1_mean_time - tel2_mean_time
            
            # Signal ratios
            tel1_total_signal = scatter_sum(tel1_data.x[:, -3], tel1_data.batch, dim=0)  # Assuming signal is at index -3
            tel2_total_signal = scatter_sum(tel2_data.x[:, -3], tel2_data.batch, dim=0)
            signal_ratio = tel1_total_signal / (tel2_total_signal + 1e-8)
            
            # Telescope pointing differences (assuming they're in the last 2 features)
            tel1_pointing = scatter_mean(tel1_data.x[:, -2:], tel1_data.batch, dim=0)
            tel2_pointing = scatter_mean(tel2_data.x[:, -2:], tel2_data.batch, dim=0)
            pointing_diff = tel1_pointing - tel2_pointing
            
            stereo_features = torch.cat([
                time_diff.unsqueeze(1),
                signal_ratio.unsqueeze(1),
                pointing_diff
            ], dim=1)
            
            # Combine all global variables
            global_variables = torch.cat([
                tel1_globals,
                tel2_globals,
                stereo_features
            ], dim=1)
        else:
            global_variables = torch.cat([tel1_globals, tel2_globals], dim=1)
            
        return global_variables
        
    def _calculate_telescope_global_variables(self, data: Data, target_device: Optional[torch.device] = None) -> Tensor:
        """Calculate global variables for a single telescope."""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Handle empty telescope case
        if x is None or x.shape[0] == 0 or batch is None or batch.numel() == 0:
            # Return empty tensor with correct dimensions
            # Use target device if provided, otherwise try to infer
            if target_device is not None:
                device = target_device
                expected_features = 11  # Default: 7 features + 4 homophily terms
            elif x is not None and x.numel() > 0:
                device = x.device
                expected_features = x.shape[1] + 4
            elif batch is not None and batch.numel() > 0:
                device = batch.device
                expected_features = 11  # Default: 7 features + 4 homophily terms
            elif torch.cuda.is_available():
                device = torch.device('cuda:0')
                expected_features = 11
            else:
                device = torch.device('cpu')
                expected_features = 11
            # Return zeros with shape [0, expected_features]
            return torch.zeros(0, expected_features, device=device)
        
        # Ensure all tensors are on the same device
        device = x.device
        if edge_index is not None:
            edge_index = edge_index.to(device)
        if batch is not None:
            batch = batch.to(device)
        
        # Calculate mean features
        global_means = scatter_mean(x, batch, dim=0)
        
        # Calculate homophily (handle edge case of no edges)
        if edge_index is not None and edge_index.shape[1] > 0:
            h_x, h_y, h_z, h_t = calculate_xyzt_homophily(x, edge_index, batch)
            
            # Ensure homophily terms have correct shape [batch_size, 1]
            # The calculate_xyzt_homophily function may return different shapes
            if h_x.dim() == 1:
                h_x = h_x.unsqueeze(1)
            if h_y.dim() == 1:
                h_y = h_y.unsqueeze(1)
            if h_z.dim() == 1:
                h_z = h_z.unsqueeze(1)
            if h_t.dim() == 1:
                h_t = h_t.unsqueeze(1)
            
            # If they have more than 2 dimensions, squeeze appropriately
            if h_x.dim() > 2:
                h_x = h_x.squeeze()
                if h_x.dim() == 1:
                    h_x = h_x.unsqueeze(1)
            if h_y.dim() > 2:
                h_y = h_y.squeeze()
                if h_y.dim() == 1:
                    h_y = h_y.unsqueeze(1)
            if h_z.dim() > 2:
                h_z = h_z.squeeze()
                if h_z.dim() == 1:
                    h_z = h_z.unsqueeze(1)
            if h_t.dim() > 2:
                h_t = h_t.squeeze()
                if h_t.dim() == 1:
                    h_t = h_t.unsqueeze(1)
                    
        else:
            # No edges - create zero homophily
            # Safe batch size calculation
            if batch is not None and batch.numel() > 0:
                batch_size = batch.max().item() + 1
            else:
                batch_size = 0
            h_x = torch.zeros(batch_size, 1, device=device)
            h_y = torch.zeros(batch_size, 1, device=device)
            h_z = torch.zeros(batch_size, 1, device=device)
            h_t = torch.zeros(batch_size, 1, device=device)
        
        # Combine global variables
        # global_means: [batch_size, nb_features]
        # h_x, h_y, h_z, h_t: [batch_size, 1] each
        global_variables = torch.cat([
            global_means,    # [batch_size, nb_features]
            h_x, h_y, h_z, h_t  # [batch_size, 1] each
        ], dim=1)
        
        return global_variables
        
    def _global_pooling(self, x: Tensor, batch: LongTensor) -> Tensor:
        """Perform global pooling."""
        pooled = []
        for pooling_scheme in self._global_pooling_schemes:
            pooling_fn = GLOBAL_POOLINGS[pooling_scheme]
            pooled_x = pooling_fn(x, index=batch, dim=0)
            if isinstance(pooled_x, tuple) and len(pooled_x) == 2:
                pooled_x, _ = pooled_x
            pooled.append(pooled_x)
        return torch.cat(pooled, dim=1)
        
    def _apply_cross_attention(self, tel1_features: Tensor, tel2_features: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply bidirectional cross-attention between telescopes."""
        # Reshape for attention (add batch dimension if needed)
        if tel1_features.dim() == 2:
            tel1_features = tel1_features.unsqueeze(0)  # [1, seq_len, features]
        if tel2_features.dim() == 2:
            tel2_features = tel2_features.unsqueeze(0)  # [1, seq_len, features]
            
        # Bidirectional cross-attention
        tel1_attended = tel1_features
        tel2_attended = tel2_features
        
        # Apply cross-attention layers
        if hasattr(self, '_late_cross_attention'):
            for cross_attn_layer in self._late_cross_attention:
                # FIXED: Independent attention to avoid feedback loops
                tmp1 = cross_attn_layer(tel1_attended, tel2_attended, tel2_attended)  # M1 attends to M2
                tmp2 = cross_attn_layer(tel2_attended, tel1_attended, tel1_attended)  # M2 attends to M1
                tel1_attended, tel2_attended = tmp1, tmp2
                
        # Remove batch dimension
        tel1_attended = tel1_attended.squeeze(0)
        tel2_attended = tel2_attended.squeeze(0)
        
        return tel1_attended, tel2_attended
        
    def forward(self, data: Data) -> Tensor:
        """Apply learnable forward pass."""
        # Split data by telescope
        tel1_data, tel2_data = self._split_by_telescope(data)
        
        # Calculate global variables
        global_variables = self._calculate_stereo_global_variables(tel1_data, tel2_data, data)
        
        # Distribute global variables to nodes if needed
        if not self._add_global_variables_after_pooling:
            # Distribute to telescope 1 (per-sample global variables)
            tel1_global_distributed = global_variables[tel1_data.batch]
            tel1_data.x = torch.cat([tel1_data.x, tel1_global_distributed], dim=1)
            
            # Distribute to telescope 2 (per-sample global variables)
            tel2_global_distributed = global_variables[tel2_data.batch]
            tel2_data.x = torch.cat([tel2_data.x, tel2_global_distributed], dim=1)
            
        # Process each telescope with DynEdge layers
        tel1_skip_connections = [tel1_data.x]
        tel2_skip_connections = [tel2_data.x]
        
        tel1_x = tel1_data.x
        tel2_x = tel2_data.x
        tel1_edge_index = tel1_data.edge_index
        tel2_edge_index = tel2_data.edge_index
        
        for i, (tel1_conv, tel2_conv) in enumerate(zip(self._tel1_conv_layers, self._tel2_conv_layers)):
            # Apply DynEdge convolutions
            tel1_x, tel1_edge_index = tel1_conv(tel1_x, tel1_edge_index, tel1_data.batch)
            tel2_x, tel2_edge_index = tel2_conv(tel2_x, tel2_edge_index, tel2_data.batch)
            
            # Apply early cross-attention if multi-stage
            if self._fusion_strategy == "multi_stage" and hasattr(self, '_early_cross_attention'):
                tel1_x_attended, tel2_x_attended = self._apply_cross_attention(tel1_x, tel2_x)
                tel1_x = tel1_x_attended
                tel2_x = tel2_x_attended
                
            tel1_skip_connections.append(tel1_x)
            tel2_skip_connections.append(tel2_x)
            
        # Apply late cross-attention (skip for concatenation strategy)
        if self._fusion_strategy in ["late", "multi_stage"]:
            tel1_x, tel2_x = self._apply_cross_attention(tel1_x, tel2_x)
            
        # Combine skip connections
        if self._fusion_strategy == "multi_stage":
            tel1_combined = torch.cat(tel1_skip_connections, dim=1)
            tel2_combined = torch.cat(tel2_skip_connections, dim=1)
        else:
            tel1_combined = tel1_x
            tel2_combined = tel2_x
            
        # Global pooling for each telescope separately (since they have different numbers of nodes)
        tel1_pooled = self._global_pooling(tel1_combined, tel1_data.batch)
        tel2_pooled = self._global_pooling(tel2_combined, tel2_data.batch)
        
        # Concatenate telescope outputs after pooling
        combined_features = torch.cat([tel1_pooled, tel2_pooled], dim=1)
        
        # Post-processing
        x = self._post_processing(combined_features)
        
        # Add global variables after pooling if specified
        if self._add_global_variables_after_pooling:
            x = torch.cat([x, global_variables], dim=1)
            
        # Readout
        x = self._readout(x)
        
        return x
