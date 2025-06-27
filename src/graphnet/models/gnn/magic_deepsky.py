from __future__ import annotations

from typing import Any, Dict
from torch import Tensor

from .icemix import DeepIce
from graphnet.models.components.magic_embedding import MagicFourierEncoder


class MagicDeepSky(DeepIce):
    """DeepIce variant with a custom Fourier encoder suitable for MAGIC data."""

    def __init__(
        self,
        hidden_dim: int = 384,
        mlp_ratio: int = 4,
        seq_length: int = 192,
        depth: int = 12,
        head_size: int = 32,
        depth_rel: int = 4,
        n_rel: int = 1,
        scaled_emb: bool = False,
        include_dynedge: bool = False,
        dynedge_args: Dict[str, Any] | None = None,
        n_features: int = 7,
        feature_scales: Tensor | None = None,
    ) -> None:
        # Initialise the regular DeepIce backbone first (creates FourierEncoder)
        super().__init__(
            hidden_dim=hidden_dim,
            mlp_ratio=mlp_ratio,
            seq_length=seq_length,
            depth=depth,
            head_size=head_size,
            depth_rel=depth_rel,
            n_rel=n_rel,
            scaled_emb=scaled_emb,
            include_dynedge=include_dynedge,
            dynedge_args=dynedge_args or {},
            n_features=n_features,  # pass through although original enc will be replaced
        )

        # Replace the Fourier encoder with our MAGIC-specific one
        self.fourier_ext = MagicFourierEncoder(
            seq_length=seq_length,
            output_dim=hidden_dim // 2 if include_dynedge else hidden_dim,
            scaled=scaled_emb,
            n_features=n_features,
            feature_scales=feature_scales,
        )

    # Forward is inherited unchanged from DeepIce; we only changed the encoder. 
