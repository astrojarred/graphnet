"""Reconstruction-specific `Model` class(es)."""

import numpy as np
import torch
from torch import Tensor

from graphnet.models.task import StandardLearnedTask
from graphnet.utilities.maths import eps_like


class AzimuthReconstructionWithKappa(StandardLearnedTask):
    """Reconstructs azimuthal angle and associated kappa (1/var)."""

    # Requires two features: untransformed points in (x,y)-space.
    default_target_labels = ["azimuth"]
    default_prediction_labels = ["azimuth_pred", "azimuth_kappa"]
    nb_inputs = 2

    def _forward(self, x: Tensor) -> Tensor:
        # Transform outputs to angle and prepare prediction
        kappa = torch.linalg.vector_norm(x, dim=1) + eps_like(x)
        angle = torch.atan2(x[:, 1], x[:, 0])
        angle = torch.where(
            angle < 0, angle + 2 * np.pi, angle
        )  # atan(y,x) -> [-pi, pi]
        return torch.stack((angle, kappa), dim=1)


class AzimuthReconstruction(AzimuthReconstructionWithKappa):
    """Reconstructs azimuthal angle."""

    # Requires two features: untransformed points in (x,y)-space.
    default_target_labels = ["azimuth"]
    default_prediction_labels = ["azimuth_pred"]
    nb_inputs = 2

    def _forward(self, x: Tensor) -> Tensor:
        # Transform outputs to angle and prepare prediction
        res = super()._forward(x)
        angle = res[:, 0].unsqueeze(1)
        kappa = res[:, 1]
        sigma = torch.sqrt(1.0 / kappa)
        beta = 1e-3
        kl_loss = torch.mean(sigma**2 - torch.log(sigma) - 1)
        self._regularisation_loss += beta * kl_loss
        return angle


class DirectionReconstructionWithKappa(StandardLearnedTask):
    """Reconstructs direction with kappa from the 3D-vMF distribution."""

    # Requires three features: untransformed points in (x,y,z)-space.
    default_target_labels = [
        "direction"
    ]  # contains dir_x, dir_y, dir_z see https://github.com/graphnet-team/graphnet/blob/95309556cfd46a4046bc4bd7609888aab649e295/src/graphnet/training/labels.py#L29
    default_prediction_labels = [
        "dir_x_pred",
        "dir_y_pred",
        "dir_z_pred",
        "direction_kappa",
    ]
    nb_inputs = 3

    def _forward(self, x: Tensor) -> Tensor:
        # Transform outputs to angle and prepare prediction
        kappa = torch.linalg.vector_norm(x, dim=1) + eps_like(x)
        vec_x = x[:, 0] / kappa
        vec_y = x[:, 1] / kappa
        vec_z = x[:, 2] / kappa
        return torch.stack((vec_x, vec_y, vec_z, kappa), dim=1)


class ZenithReconstruction(StandardLearnedTask):
    """Reconstructs zenith angle."""

    # Requires two features: zenith angle itself.
    default_target_labels = ["zenith"]
    default_prediction_labels = ["zenith_pred"]
    nb_inputs = 1

    def _forward(self, x: Tensor) -> Tensor:
        # Transform outputs to angle and prepare prediction
        return torch.sigmoid(x[:, :1]) * np.pi


class ZenithReconstructionWithKappa(ZenithReconstruction):
    """Reconstructs zenith angle and associated kappa (1/var)."""

    # Requires one feature in addition to `ZenithReconstruction`: kappa (unceratinty; 1/variance).
    default_target_labels = ["zenith"]
    default_prediction_labels = ["zenith_pred", "zenith_kappa"]
    nb_inputs = 2

    def _forward(self, x: Tensor) -> Tensor:
        # Transform outputs to angle and prepare prediction
        angle = super()._forward(x[:, :1]).squeeze(1)
        kappa = torch.abs(x[:, 1]) + eps_like(x)
        return torch.stack((angle, kappa), dim=1)


class EnergyReconstruction(StandardLearnedTask):
    """Reconstructs energy using stable method."""

    # Requires one feature: untransformed energy
    default_target_labels = ["energy"]
    default_prediction_labels = ["energy_pred"]
    nb_inputs = 1

    def _forward(self, x: Tensor) -> Tensor:
        # Transform to positive energy domain avoiding `-inf` in `log10`
        # Transform, thereby preventing overflow and underflow error.
        return torch.nn.functional.softplus(x, beta=0.05) + eps_like(x)


class EnergyReconstructionWithPower(StandardLearnedTask):
    """Reconstructs energy."""

    # Requires one feature: untransformed energy
    default_target_labels = ["energy"]
    default_prediction_labels = ["energy_pred"]
    nb_inputs = 1

    def _forward(self, x: Tensor) -> Tensor:
        # Transform energy
        return torch.pow(10, x[:, 0] + 1.0).unsqueeze(1)


class EnergyTCReconstruction(StandardLearnedTask):
    """Reconstructs track and cascade energies using stable method."""

    # Requires two features: untransformed energy for track and cascade
    default_target_labels = ["energy_track", "energy_cascade"]
    default_prediction_labels = ["energy_track_pred", "energy_cascade_pred"]
    nb_inputs = 2

    def _forward(self, x: Tensor) -> Tensor:
        # Transform to positive energy domain avoiding `-inf` in `log10`
        # Transform, thereby preventing overflow and underflow error.
        x[:, 0] = torch.nn.functional.softplus(
            x[:, 0].clone(), beta=0.05
        ) + eps_like(x[:, 0].clone())
        x[:, 1] = torch.nn.functional.softplus(
            x[:, 1].clone(), beta=0.05
        ) + eps_like(x[:, 1].clone())
        return x


class EnergyReconstructionWithUncertainty(EnergyReconstruction):
    """Reconstructs energy and associated uncertainty (log(var))."""

    # Requires one feature in addition to `EnergyReconstruction`: log-variance (uncertainty).
    default_target_labels = ["energy"]
    default_prediction_labels = ["energy_pred", "energy_sigma"]
    nb_inputs = 2

    def _forward(self, x: Tensor) -> Tensor:
        # Transform energy
        energy = super()._forward(x[:, :1]).squeeze(1)
        log_var = x[:, 1]
        pred = torch.stack((energy, log_var), dim=1)
        return pred


class VertexReconstruction(StandardLearnedTask):
    """Reconstructs vertex position and time."""

    # Requires four features, x, y, z, and t.
    default_target_labels = ["vertex"]
    default_prediction_labels = [
        "position_x_pred",
        "position_y_pred",
        "position_z_pred",
        "interaction_time_pred",
    ]
    nb_inputs = 4

    def _forward(self, x: Tensor) -> Tensor:
        # Scale xyz to roughly the right order of magnitude, leave time
        x[:, 0] = x[:, 0] * 1e2
        x[:, 1] = x[:, 1] * 1e2
        x[:, 2] = x[:, 2] * 1e2

        return x


class PositionReconstruction(StandardLearnedTask):
    """Reconstructs vertex position."""

    # Requires three features, x, y, and z.
    default_target_labels = ["position"]
    default_prediction_labels = [
        "position_x_pred",
        "position_y_pred",
        "position_z_pred",
    ]
    nb_inputs = 3

    def _forward(self, x: Tensor) -> Tensor:
        # Scale to roughly the right order of magnitude
        x[:, 0] = x[:, 0] * 1e2
        x[:, 1] = x[:, 1] * 1e2
        x[:, 2] = x[:, 2] * 1e2

        return x


class TimeReconstruction(StandardLearnedTask):
    """Reconstructs time."""

    # Requires one feature, time.
    default_target_labels = ["interaction_time"]
    default_prediction_labels = ["interaction_time_pred"]
    nb_inputs = 1

    def _forward(self, x: Tensor) -> Tensor:
        # Leave as it is
        return x


class InelasticityReconstruction(StandardLearnedTask):
    """Reconstructs interaction inelasticity.

    That is, 1-(track energy / hadronic energy).
    """

    # Requires one features: inelasticity itself
    default_target_labels = ["inelasticity"]
    default_prediction_labels = ["inelasticity_pred"]
    nb_inputs = 1

    def _forward(self, x: Tensor) -> Tensor:
        # Transform output to unit range
        return torch.sigmoid(x)


class MAGICDirectionClassificationTask(StandardLearnedTask):
    """MAGIC direction reconstruction using quantile-binned classification.
    
    Based on IceCube Kaggle winning approaches. Converts continuous angle
    regression into classification over quantile-based bins.
    """
    
    default_target_labels = ["true_phi", "true_theta"]  # MAGIC truth labels
    default_prediction_labels = ["azimuth_pred", "zenith_pred"]
    
    def __init__(
        self,
        hidden_size: int,
        num_azimuth_bins: int = 48,
        num_zenith_bins: int = 48,
        **kwargs
    ):
        """Initialize classification task.
        
        Args:
            hidden_size: Size of input features from backbone
            num_azimuth_bins: Number of azimuth angle bins (0-2π)
            num_zenith_bins: Number of zenith angle bins (0-π)
        """
        self.num_azimuth_bins = num_azimuth_bins
        self.num_zenith_bins = num_zenith_bins
        
        # Set inference transform to convert logits to angles
        if 'transform_inference' not in kwargs:
            kwargs['transform_inference'] = self._transform_logits_to_angles
        
        # Parent will create: self._affine = Linear(hidden_size, self.nb_inputs)
        # where self.nb_inputs tells it how many outputs to create
        super().__init__(
            hidden_size=hidden_size,
            **kwargs
        )
        
        # Load quantile bins AFTER calling super() so buffers can be registered
        self._load_magic_directions()
    
    @property
    def nb_inputs(self) -> int:
        """Return number of logits the affine layer should output."""
        return self.num_azimuth_bins + self.num_zenith_bins
    
    def _load_magic_directions(self):
        """Load MAGIC direction data to compute quantile bins.
        
        This is THE critical component for IceCube-style performance.
        Quantile binning allocates model capacity efficiently.
        """
        import numpy as np
        
        # Try to load training data for quantile computation
        train_data_path = "/mnt/scratch/jarred/LMDB_75k_cleaned_nonstd/za05to35.lmdb"
        
        try:
            print(f"Loading training data from {train_data_path} for quantile binning...")
            
            # Import LMDB dataset tools
            from graphnet.data.dataset.lmdb.lmdb_dataset import LMDBDataset
            
            # Create temporary dataset to load truth data
            temp_dataset = LMDBDataset(
                path=train_data_path,
                pulsemaps=["total"],
                features=["x_cam", "y_cam", "t", "tel_id", "signal", "telescope_phi", "telescope_theta"],
                truth=["particle_id", "true_energy", "true_theta", "true_phi"],
                selection="event_id % 10 > 1",  # Training data only
                index_column="event_id",
                truth_table="truth"
            )
            
            print(f"Loading {len(temp_dataset)} training events for quantile computation...")
            
            # Extract all truth values for quantile computation
            all_azimuth = []
            all_zenith = []
            
            # Sample a subset for faster quantile computation (10k events should be enough)
            import random
            sample_size = min(10000, len(temp_dataset))
            sample_indices = random.sample(range(len(temp_dataset)), sample_size)
            
            for i, idx in enumerate(sample_indices):
                if i % 1000 == 0:
                    print(f"  Loading sample {i+1}/{sample_size}...")
                    
                try:
                    graph = temp_dataset[idx]
                    # Extract truth from graph object - check multiple possible attribute names
                    azimuth_val = None
                    zenith_val = None
                    
                    # Try different possible attribute names
                    for az_attr in ['true_phi', 'azimuth', 'phi']:
                        if hasattr(graph, az_attr):
                            azimuth_val = getattr(graph, az_attr)
                            break
                    
                    for zen_attr in ['true_theta', 'zenith', 'theta']:
                        if hasattr(graph, zen_attr):
                            zenith_val = getattr(graph, zen_attr)
                            break
                    
                    if azimuth_val is not None and zenith_val is not None:
                        # Convert to float and validate ranges
                        az_float = float(azimuth_val)
                        zen_float = float(zenith_val)
                        
                        # Validate ranges (azimuth: 0-2π, zenith: 0-π)
                        if 0 <= az_float <= 2*np.pi and 0 <= zen_float <= np.pi:
                            all_azimuth.append(az_float)
                            all_zenith.append(zen_float)
                        
                except Exception as e:
                    # Skip problematic samples
                    continue
            
            all_azimuth = np.array(all_azimuth)
            all_zenith = np.array(all_zenith)
            
            print(f"Loaded {len(all_azimuth)} direction samples")
            
            # Validate we have enough samples for quantile computation
            min_samples_needed = max(self.num_azimuth_bins, self.num_zenith_bins) * 10
            if len(all_azimuth) < min_samples_needed:
                raise ValueError(f"Need at least {min_samples_needed} samples for quantile binning, got {len(all_azimuth)}")
            
            print(f"Azimuth range: [{all_azimuth.min():.3f}, {all_azimuth.max():.3f}] rad")
            print(f"Zenith range: [{all_zenith.min():.3f}, {all_zenith.max():.3f}] rad")
            
            # Compute quantile-based bins (THE SECRET SAUCE!)
            azimuth_quantiles = np.linspace(0, 1, self.num_azimuth_bins + 1)
            zenith_quantiles = np.linspace(0, 1, self.num_zenith_bins + 1)
            
            self.azimuth_bins = np.quantile(all_azimuth, azimuth_quantiles)
            self.zenith_bins = np.quantile(all_zenith, zenith_quantiles)
            
            # Calculate bin centers from quantile edges
            self.azimuth_bin_centers = (self.azimuth_bins[:-1] + self.azimuth_bins[1:]) / 2
            self.zenith_bin_centers = (self.zenith_bins[:-1] + self.zenith_bins[1:]) / 2
            
            print("✅ Successfully computed quantile-based bins!")
            print(f"Azimuth bins: {self.num_azimuth_bins} bins from {self.azimuth_bins[0]:.3f} to {self.azimuth_bins[-1]:.3f} rad")
            print(f"Zenith bins: {self.num_zenith_bins} bins from {self.zenith_bins[0]:.3f} to {self.zenith_bins[-1]:.3f} rad")
            
        except Exception as e:
            print(f"⚠️  Could not load training data for quantiles: {e}")
            print("Falling back to uniform bins...")
            
            # Fallback to uniform bins
            self.azimuth_bins = np.linspace(0, 2*np.pi, self.num_azimuth_bins + 1)
            self.azimuth_bin_centers = (self.azimuth_bins[:-1] + self.azimuth_bins[1:]) / 2
            
            self.zenith_bins = np.linspace(0, np.pi, self.num_zenith_bins + 1)
            self.zenith_bin_centers = (self.zenith_bins[:-1] + self.zenith_bins[1:]) / 2
        
        # Register as PyTorch buffers so they move with model to GPU
        self.register_buffer('azimuth_bin_centers_tensor', 
                           torch.tensor(self.azimuth_bin_centers, dtype=torch.float32))
        self.register_buffer('zenith_bin_centers_tensor',
                           torch.tensor(self.zenith_bin_centers, dtype=torch.float32))
    
    def _forward(self, x: Tensor) -> Tensor:
        """Forward pass: return raw logits.
        
        For classification tasks, _forward should return raw logits so the loss 
        function can access them. The compute_loss() method will pass these 
        logits to CircularSmoothCrossEntropyLoss.
        
        Args:
            x: Output from affine layer [batch_size, num_azimuth_bins + num_zenith_bins]
            
        Returns:
            Raw logits [batch_size, num_azimuth_bins + num_zenith_bins]
        """
        # For classification, return raw logits (like BinaryClassificationTaskLogits)
        return x
    
    def predict_angles(self, x: Tensor) -> Tensor:
        """Convert logits to predicted angles using weighted bin centers.
        
        This method is for inference/evaluation, not training.
        
        Args:
            x: Input features [batch_size, hidden_size]
            
        Returns:
            Predicted angles [batch_size, 2] (azimuth, zenith)
        """
        # Get logits
        logits = self._forward(x)
        
        # Split azimuth and zenith logits
        azimuth_logits = logits[:, :self.num_azimuth_bins]
        zenith_logits = logits[:, self.num_azimuth_bins:]
        
        # Convert logits to probabilities
        azimuth_probs = torch.softmax(azimuth_logits, dim=1)
        zenith_probs = torch.softmax(zenith_logits, dim=1)
        
        # Compute weighted average of bin centers (expectation)
        azimuth_pred = torch.sum(azimuth_probs * self.azimuth_bin_centers_tensor, dim=1)
        zenith_pred = torch.sum(zenith_probs * self.zenith_bin_centers_tensor, dim=1)
        
        return torch.stack([azimuth_pred, zenith_pred], dim=1)
    
    def _transform_logits_to_angles(self, prediction: Tensor) -> Tensor:
        """Transform raw logits to angle predictions for inference.
        
        This is the key function that converts the 96 classification logits
        into 2 angle predictions during inference.
        
        Args:
            prediction: Raw logits [batch_size, num_azimuth_bins + num_zenith_bins]
            
        Returns:
            Predicted angles [batch_size, 2] (azimuth, zenith) in radians
        """
        # Split azimuth and zenith logits
        azimuth_logits = prediction[:, :self.num_azimuth_bins]
        zenith_logits = prediction[:, self.num_azimuth_bins:]
        
        # Convert logits to probabilities
        azimuth_probs = torch.softmax(azimuth_logits, dim=1)
        zenith_probs = torch.softmax(zenith_logits, dim=1)
        
        # Compute weighted average of bin centers (expectation)
        azimuth_pred = torch.sum(azimuth_probs * self.azimuth_bin_centers_tensor, dim=1)
        zenith_pred = torch.sum(zenith_probs * self.zenith_bin_centers_tensor, dim=1)
        
        return torch.stack([azimuth_pred, zenith_pred], dim=1)
