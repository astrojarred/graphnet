"""Classes for performing embedding of input data."""
import torch
import torch.nn as nn
from torch.functional import Tensor

from typing import Optional

from pytorch_lightning import LightningModule


class SinusoidalPosEmb(LightningModule):
    """Sinusoidal positional embeddings module.

    This module is from the kaggle competition 2nd place solution (see
    arXiv:2310.15674): It performs what is called Fourier encoding or it's used
    in the Attention is all you need arXiv:1706.03762. It can be seen as a soft
    digitization of the input data
    """

    def __init__(
        self,
        dim: int = 16,
        n_freq: int = 10000,
        scaled: bool = False,
    ):
        """Construct `SinusoidalPosEmb`.

        Args:
            dim: Embedding dimension.
            n_freq: Number of frequencies.
            scaled: Whether or not to scale the output.
        """
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim has to be even. Got: {dim}")
        self.scale = (
            nn.Parameter(torch.ones(1) * dim**-0.5) if scaled else 1.0
        )
        self.dim = dim
        self.n_freq = torch.Tensor([n_freq])

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        device = x.device
        half_dim = self.dim / 2
        emb = torch.log(self.n_freq.to(device=device)) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * (-emb))
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb * self.scale


class FourierEncoder(LightningModule):
    """Fourier encoder module.

    This module incorporates sinusoidal positional embeddings and auxiliary
    embeddings to process input sequences and produce meaningful
    representations. The module assumes that the input data is in the format of
    (x, y, z, time, charge, auxiliary), being the first four features
    mandatory.
    """

    def __init__(
        self,
        seq_length: int = 128,
        mlp_dim: Optional[int] = None,
        output_dim: int = 384,
        scaled: bool = False,
        n_features: int = 6,
    ):
        """Construct `FourierEncoder`.

        Args:
            seq_length: Dimensionality of the base sinusoidal positional
                embeddings.
            mlp_dim (Optional): Size of hidden, latent space of MLP. If not
                given, `mlp_dim` is set automatically as multiples of
                `seq_length` (in consistent with the 2nd place solution),
                depending on `n_features`.
            output_dim: Dimension of the output (I.e. number of columns).
            scaled: Whether or not to scale the embeddings.
            n_features: The number of features in the input data.
        """
        super().__init__()

        self.sin_emb = SinusoidalPosEmb(dim=seq_length, scaled=scaled)
        self.sin_emb2 = SinusoidalPosEmb(dim=seq_length // 2, scaled=scaled)

        if n_features < 4:
            raise ValueError(
                f"At least x, y, z and time of the DOM are required. Got only "
                f"{n_features} features."
            )
        elif n_features >= 6:
            self.aux_emb = nn.Embedding(2, seq_length // 2)
            hidden_dim = 6 * seq_length
        else:
            hidden_dim = int((n_features + 0.5) * seq_length)

        if mlp_dim is None:
            mlp_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.LayerNorm(mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, output_dim),
        )

        self.n_features = n_features

    def forward(
        self,
        x: Tensor,
        seq_length: Tensor,
    ) -> Tensor:
        """Forward pass."""
        length = torch.log10(seq_length.to(dtype=x.dtype))
        embeddings = [self.sin_emb(4096 * x[:, :, :3]).flatten(-2)]  # Position

        if self.n_features >= 5:
            embeddings.append(self.sin_emb(1024 * x[:, :, 4]))  # Charge

        embeddings.append(self.sin_emb(4096 * x[:, :, 3]))  # Time

        if self.n_features >= 6:
            embeddings.append(self.aux_emb(x[:, :, 5].long()))  # Auxiliary

        embeddings.append(
            self.sin_emb2(length).unsqueeze(1).expand(-1, max(seq_length), -1)
        )  # Length

        x = torch.cat(embeddings, -1)
        x = self.mlp(x)

        return x


class SpacetimeEncoder(LightningModule):
    """Spacetime encoder module."""

    def __init__(
        self,
        seq_length: int = 32,
    ):
        """Construct `SpacetimeEncoder`.

        This module calculates space-time interval between each pair of events
        and generates sinusoidal positional embeddings to be added to input
        sequences.

        Args:
            seq_length: Dimensionality of the sinusoidal positional embeddings.
        """
        super().__init__()
        self.sin_emb = SinusoidalPosEmb(dim=seq_length)
        self.projection = nn.Linear(seq_length, seq_length)

    def forward(
        self,
        x: Tensor,
        # Lmax: Optional[int] = None,
    ) -> Tensor:
        """Forward pass."""
        pos = x[:, :, :3]
        time = x[:, :, 3]
        spacetime_interval = (pos[:, :, None] - pos[:, None, :]).pow(2).sum(
            -1
        ) - ((time[:, :, None] - time[:, None, :]) * (3e4 / 500 * 3e-1)).pow(2)
        four_distance = torch.sign(spacetime_interval) * torch.sqrt(
            torch.abs(spacetime_interval)
        )
        sin_emb = self.sin_emb(1024 * four_distance.clip(-4, 4))
        rel_attn = self.projection(sin_emb)
        return rel_attn


class SpacetimeEncoderMAGIC(LightningModule):
    """Spacetime encoder for MAGIC's 2D camera + time geometry.
    
    MAGIC telescopes have a fundamentally different geometry than IceCube:
    - 2D camera coordinates (x_cam, y_cam) instead of 3D detector positions
    - Time is feature index 2, not 3
    - Features: [x_cam, y_cam, t, tel_id, signal, telescope_phi, telescope_theta]
    
    This encoder correctly calculates spacetime intervals for Cherenkov light
    propagation in MAGIC's stereo telescope system.
    """

    def __init__(
        self,
        seq_length: int = 32,
        time_scaling: float = 1.0,
    ):
        """Construct `SpacetimeEncoderMAGIC`.

        Args:
            seq_length: Dimensionality of the sinusoidal positional embeddings.
            time_scaling: Factor to scale time relative to spatial coordinates.
                Used to balance the importance of temporal vs spatial relationships.
        """
        super().__init__()
        self.sin_emb = SinusoidalPosEmb(dim=seq_length)
        self.projection = nn.Linear(seq_length, seq_length)
        self.time_scaling = time_scaling

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for MAGIC data.
        
        Args:
            x: Input tensor with shape [batch_size, seq_len, n_features]
               where features are [x_cam, y_cam, t, tel_id, signal, ...]
               
        Returns:
            Relative attention bias tensor for transformer attention
        """
        # MAGIC features: [x_cam, y_cam, t, tel_id, signal, telescope_phi, telescope_theta]
        pos = x[:, :, :2]   # Use features 0, 1 for camera coordinates (x_cam, y_cam)
        time = x[:, :, 2]   # Use feature 2 for time (t)
        
        # Calculate 2D+time spacetime interval for Cherenkov light
        # Spatial distance squared between camera pixels
        spatial_dist_sq = (pos[:, :, None] - pos[:, None, :]).pow(2).sum(-1)
        
        # Time distance squared (scaled to be comparable with spatial distances)
        time_dist_sq = ((time[:, :, None] - time[:, None, :]) * self.time_scaling).pow(2)
        
        # 2D+time spacetime interval: ds² = dx² + dy² - c²dt²
        # For Cherenkov light, particles traveling ~at speed of light should have ds² ≈ 0
        spacetime_interval = spatial_dist_sq - time_dist_sq
        
        # Calculate four-distance with proper sign handling
        four_distance = torch.sign(spacetime_interval) * torch.sqrt(
            torch.abs(spacetime_interval)
        )
        
        # Generate sinusoidal embeddings for the relative bias
        sin_emb = self.sin_emb(1024 * four_distance.clip(-4, 4))
        rel_attn = self.projection(sin_emb)
        
        return rel_attn


class FourierEncoderMAGIC(LightningModule):
    """Fourier encoder for MAGIC telescope data.
    
    MAGIC features (after tel_id exclusion): [x_cam, y_cam, t, signal, telescope_phi, telescope_theta]
    This encoder handles MAGIC's specific feature layout and doesn't assume auxiliary flags.
    """

    def __init__(
        self,
        seq_length: int = 128,
        mlp_dim: Optional[int] = None,
        output_dim: int = 384,
        scaled: bool = False,
        n_features: int = 6,  # After excluding tel_id: 6 features
    ):
        """Construct `FourierEncoderMAGIC`.

        Args:
            seq_length: Dimensionality of the base sinusoidal positional embeddings.
            mlp_dim: Size of hidden, latent space of MLP. If not given, set automatically.
            output_dim: Dimension of the output.
            scaled: Whether or not to scale the embeddings.
            n_features: Number of features in MAGIC data (after excluding tel_id).
        """
        super().__init__()

        self.sin_emb = SinusoidalPosEmb(dim=seq_length, scaled=scaled)
        self.sin_emb2 = SinusoidalPosEmb(dim=seq_length // 2, scaled=scaled)

        if n_features < 4:
            raise ValueError(
                f"At least x_cam, y_cam, t, and signal are required for MAGIC. "
                f"Got only {n_features} features."
            )

        # MAGIC doesn't have auxiliary flags, so we calculate hidden_dim differently
        # Features: [x_cam, y_cam, t, signal, telescope_phi, telescope_theta] = 6 features
        # Each feature gets sinusoidal encoding, plus length encoding
        hidden_dim = n_features * seq_length + seq_length // 2  # +length encoding

        if mlp_dim is None:
            mlp_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.LayerNorm(mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, output_dim),
        )

        self.n_features = n_features

    def forward(self, x: Tensor, seq_length: Tensor) -> Tensor:
        """Forward pass for MAGIC data.
        
        Args:
            x: Input tensor with MAGIC features [batch, seq_len, n_features]
               Features: [x_cam, y_cam, t, signal, telescope_phi, telescope_theta]
            seq_length: Sequence lengths per batch
            
        Returns:
            Fourier-encoded features
        """
        length = torch.log10(seq_length.to(dtype=x.dtype))
        embeddings = []

        # Encode each MAGIC feature with appropriate scaling
        if self.n_features >= 2:
            # Camera coordinates (x_cam, y_cam) - scale for camera size ~60cm
            embeddings.append(self.sin_emb(4096 * x[:, :, 0]))  # x_cam
            embeddings.append(self.sin_emb(4096 * x[:, :, 1]))  # y_cam

        if self.n_features >= 3:
            # Time - scale for nanosecond precision
            embeddings.append(self.sin_emb(4096 * x[:, :, 2]))  # t

        if self.n_features >= 4:
            # Signal (charge equivalent) - scale for dynamic range
            embeddings.append(self.sin_emb(1024 * x[:, :, 3]))  # signal

        if self.n_features >= 5:
            # Telescope pointing phi - scale for small angles
            embeddings.append(self.sin_emb(8192 * x[:, :, 4]))  # telescope_phi

        if self.n_features >= 6:
            # Telescope pointing theta - scale for small angles  
            embeddings.append(self.sin_emb(8192 * x[:, :, 5]))  # telescope_theta

        # Add sequence length encoding
        embeddings.append(
            self.sin_emb2(length).unsqueeze(1).expand(-1, max(seq_length), -1)
        )

        # Concatenate all embeddings
        x = torch.cat(embeddings, -1)
        x = self.mlp(x)

        return x
