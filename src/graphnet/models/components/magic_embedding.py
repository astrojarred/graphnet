from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
from torch import Tensor

from .embedding import SinusoidalPosEmb  # reuse existing implementation


class MagicFourierEncoder(LightningModule):
    """Fourier encoder that can handle an arbitrary number of (continuous) MAGIC features.

    The original `FourierEncoder` from the IceCube solution assumes a very
    specific column ordering (x, y, z, time, charge, aux) and treats the sixth
    column as a *binary* categorical feature.  For MAGIC we want to keep the
    full camera feature set

        [x_cam, y_cam, tel_id, t, signal, telescope_phi, telescope_theta]

    (7 columns) without any hard-coded categorical constraint.  This encoder
    simply applies a sinusoidal positional embedding to *every* feature and
    concatenates them; optional scaling factors per feature can be supplied
    but default to `1`.
    """

    def __init__(
        self,
        seq_length: int = 128,
        output_dim: int = 384,
        scaled: bool = False,
        n_features: int = 7,
        feature_scales: Tensor | None = None,
    ) -> None:
        """Construct `MagicFourierEncoder`.

        Args:
            seq_length: Base sinusoidal embedding dimension per feature.
            output_dim: Dimension of the encoded output that is fed to the
                transformer backbone.
            scaled: Whether to apply the `dim**-0.5` scaling inside
                `SinusoidalPosEmb`.
            n_features: Number of continuous input features.
            feature_scales: Optional 1-D tensor of length `n_features` giving
                per-feature multiplicative scales *before* the sinusoidal
                embedding.  If *None* a tensor of ones is used.
        """
        super().__init__()

        self.seq_length = seq_length
        self.n_features = n_features
        self.sin_emb = SinusoidalPosEmb(dim=seq_length, scaled=scaled)
        self.sin_emb2 = SinusoidalPosEmb(dim=seq_length // 2, scaled=scaled)

        if feature_scales is None:
            feature_scales = torch.ones(n_features)
        else:
            assert (
                len(feature_scales) == n_features
            ), "feature_scales must match n_features"
        # Register as buffer so it moves with the module but is not a parameter
        self.register_buffer("feature_scales", feature_scales.view(1, 1, -1))

        hidden_dim = n_features * seq_length + seq_length // 2  # +length emb

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor, seq_length: Tensor) -> Tensor:  # noqa: D401
        """Encode the pulse feature tensor.

        Args:
            x: Tensor of shape ``[batch, n_pulses, n_features]``.
            seq_length: 1-D integer tensor of shape ``[batch]`` containing the
                per-event number of pulses (used for the length embedding).
        Returns:
            Encoded tensor of shape ``[batch, n_pulses, output_dim]``.
        """
        # Scale features (broadcast over batch & pulse dimensions)
        scaled_x = x * self.feature_scales.to(dtype=x.dtype)

        # Sinusoidal embedding for each individual feature
        embeds = [
            self.sin_emb(scaled_x[:, :, i]) for i in range(self.n_features)
        ]
        x_emb = torch.cat(embeds, dim=-1)  # [B, N, n_features * seq_length]

        # Global length embedding (how many pulses in this event)
        length = torch.log10(seq_length.to(dtype=x.dtype))
        length_emb = (
            self.sin_emb2(length)
            .unsqueeze(1)
            .expand(-1, x_emb.size(1), -1)
        )

        x_emb = torch.cat([x_emb, length_emb], dim=-1)
        return self.mlp(x_emb) 
