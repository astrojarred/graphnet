"""MAGIC-specific node definitions for IceMix transformer architecture."""

from typing import List, Optional, Dict, Any
import torch
from torch_geometric.data import Data

from graphnet.models.graphs.nodes import NodeDefinition


class MAGICIceMixNodes(NodeDefinition):
    """MAGIC-specific node definition for IceMix transformer architecture.
    
    Handles MAGIC telescope data for transformer processing:
    - Random sampling to keep sequences manageable
    - Proper handling of dual-telescope structure
    - Telescope-aware preprocessing for transformer input
    """

    def __init__(
        self,
        input_feature_names: Optional[List[str]] = None,
        max_pulses: int = 512,
        tel_id_name: str = "tel_id",
        max_pulses_per_telescope: Optional[int] = None,
        telescope_balancing: bool = True,
        **kwargs: Any,
    ) -> None:
        """Construct `MAGICIceMixNodes`.

        Args:
            input_feature_names: Column names for input features. Expected MAGIC
                features: [x_cam, y_cam, t, tel_id, signal, telescope_phi, telescope_theta]
            max_pulses: Maximum number of pulses to keep in the entire event.
            tel_id_name: Name of the telescope ID column.
            max_pulses_per_telescope: Maximum pulses per individual telescope.
                If None, uses max_pulses // 2.
            telescope_balancing: Whether to balance pulses between telescopes.
        """
        if input_feature_names is None:
            input_feature_names = [
                "x_cam",
                "y_cam", 
                "t",
                "tel_id",
                "signal",
                "telescope_phi",
                "telescope_theta",
            ]

        super().__init__(input_feature_names=input_feature_names)

        # Validate telescope ID column
        if tel_id_name not in input_feature_names:
            raise ValueError(
                f"Telescope ID column '{tel_id_name}' not found in "
                f"input_feature_names {input_feature_names}"
            )

        self.input_feature_names = input_feature_names
        self._n_features = len(input_feature_names)  # Use private attribute to avoid conflict
        self.max_length = max_pulses
        self.tel_id_name = tel_id_name
        self.tel_id_index = input_feature_names.index(tel_id_name)
        self.telescope_balancing = telescope_balancing
        
        # Set per-telescope limits
        if max_pulses_per_telescope is None:
            self.max_pulses_per_telescope = max_pulses // 2
        else:
            self.max_pulses_per_telescope = max_pulses_per_telescope

        # Feature mapping for easy access
        self.feature_indexes = {
            feat: input_feature_names.index(feat) for feat in input_feature_names
        }

    def _define_output_feature_names(self, input_feature_names: List[str]) -> List[str]:
        """Return output feature names (same as input for MAGIC)."""
        return input_feature_names

    def _pulse_sampler(
        self, x: torch.Tensor, tel_id: int = None
    ) -> torch.Tensor:
        """Sample pulses from an event or telescope.
        
        Args:
            x: Input tensor with pulse data
            tel_id: Telescope ID to filter by (None for all telescopes)
            
        Returns:
            Indices of selected pulses
        """
        event_length = x.shape[0]
        
        # Filter by telescope if specified
        if tel_id is not None:
            tel_mask = x[:, self.tel_id_index] == tel_id
            tel_indices = torch.where(tel_mask)[0]
            
            if len(tel_indices) == 0:
                return torch.empty(0, dtype=torch.long)
                
            max_length = min(self.max_pulses_per_telescope, len(tel_indices))
            
            if len(tel_indices) <= max_length:
                return tel_indices
            else:
                # Random sampling within telescope
                perm = torch.randperm(len(tel_indices))[:max_length]
                return tel_indices[perm].sort().values
        else:
            # Sample from entire event
            max_length = min(self.max_length, event_length)
            
            if event_length <= max_length:
                return torch.arange(event_length)
            else:
                # Random sampling
                return torch.randperm(event_length)[:max_length].sort().values

    def _balanced_telescope_sampling(self, x: torch.Tensor) -> torch.Tensor:
        """Sample pulses with balancing between telescopes.
        
        Args:
            x: Input tensor with pulse data
            
        Returns:
            Indices of selected pulses from both telescopes
        """
        # Get telescope masks
        tel1_mask = x[:, self.tel_id_index] == 0
        tel2_mask = x[:, self.tel_id_index] == 1
        
        tel1_indices = torch.where(tel1_mask)[0]
        tel2_indices = torch.where(tel2_mask)[0]
        
        # Sample from each telescope
        tel1_sampled = self._pulse_sampler(x[tel1_indices], tel_id=None)
        tel2_sampled = self._pulse_sampler(x[tel2_indices], tel_id=None)
        
        # Map back to original indices
        if len(tel1_sampled) > 0:
            tel1_final = tel1_indices[tel1_sampled]
        else:
            tel1_final = torch.empty(0, dtype=torch.long)
            
        if len(tel2_sampled) > 0:
            tel2_final = tel2_indices[tel2_sampled]
        else:
            tel2_final = torch.empty(0, dtype=torch.long)
        
        # Combine and sort
        all_indices = torch.cat([tel1_final, tel2_final])
        
        # Ensure we don't exceed maximum
        if len(all_indices) > self.max_length:
            perm = torch.randperm(len(all_indices))[:self.max_length]
            all_indices = all_indices[perm].sort().values
        else:
            all_indices = all_indices.sort().values
            
        return all_indices

    def _construct_nodes(self, x: torch.Tensor) -> Data:
        """Construct nodes from raw MAGIC node features.
        
        Args:
            x: Raw pulse features with shape [num_pulses, n_features]
            
        Returns:
            Data object with sampled and processed nodes
        """
        event_length = x.shape[0]
        
        if event_length == 0:
            # Return empty graph
            return Data(x=torch.zeros(0, self.n_features))
        
        # Sample pulses
        if self.telescope_balancing:
            ids = self._balanced_telescope_sampling(x)
        else:
            ids = self._pulse_sampler(x)
        
        # Handle empty sampling
        if len(ids) == 0:
            return Data(x=torch.zeros(0, self.n_features))  # Use property here
        
        # Create output tensor
        sampled_length = len(ids)
        graph = torch.zeros([sampled_length, self._n_features])
        
        # Copy selected features
        for idx, feature in enumerate(self.input_feature_names):
            feature_idx = self.feature_indexes[feature]
            graph[:sampled_length, idx] = x[ids, feature_idx]
        
        return Data(x=graph)

    @property
    def n_features(self) -> int:
        """Return number of output features."""
        return len(self.input_feature_names)


class MAGICIceMixNodesSimple(NodeDefinition):
    """Simplified MAGIC node definition for IceMix - no telescope balancing.
    
    This is a simpler version that treats MAGIC data like IceCube data,
    useful for testing and comparison with the standard IceMix approach.
    """

    def __init__(
        self,
        input_feature_names: Optional[List[str]] = None,
        max_pulses: int = 512,
        **kwargs: Any,
    ) -> None:
        """Construct `MAGICIceMixNodesSimple`.

        Args:
            input_feature_names: Column names for input features.
            max_pulses: Maximum number of pulses to keep in the event.
        """
        if input_feature_names is None:
            input_feature_names = [
                "x_cam",
                "y_cam", 
                "t",
                "tel_id",
                "signal",
                "telescope_phi",
                "telescope_theta",
            ]

        super().__init__(input_feature_names=input_feature_names)

        self.input_feature_names = input_feature_names
        self._n_features = len(input_feature_names)  # Use private attribute
        self.max_length = max_pulses

    def _define_output_feature_names(self, input_feature_names: List[str]) -> List[str]:
        """Return output feature names (same as input)."""
        return input_feature_names

    def _construct_nodes(self, x: torch.Tensor) -> Data:
        """Construct nodes with simple random sampling.
        
        Args:
            x: Raw pulse features with shape [num_pulses, n_features]
            
        Returns:
            Data object with sampled nodes
        """
        event_length = x.shape[0]
        
        if event_length == 0:
            return Data(x=torch.zeros(0, self.n_features))  # Use property here
        
        # Simple random sampling
        if event_length <= self.max_length:
            ids = torch.arange(event_length)
        else:
            ids = torch.randperm(event_length)[:self.max_length].sort().values
        
        # Create output tensor
        sampled_length = len(ids)
        graph = torch.zeros([sampled_length, self.n_features])  # Use property here
        
        # Copy features
        graph[:sampled_length, :] = x[ids, :]
        
        return Data(x=graph)

    @property
    def n_features(self) -> int:
        """Return number of output features."""
        return len(self.input_feature_names)