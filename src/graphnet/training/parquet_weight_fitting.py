from __future__ import annotations

from typing import Callable, Dict, List, Union, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch_geometric.data import Data

__all__ = [
    "ParquetWeightFitter",
    "MAGICSystematicWeighter",
    "MAGICZenithEnergyWeighter",
]


class ParquetWeightFitter:
    """Base class for fitting weights from parquet files.

    This is similar to GraphNeT's WeightFitter but designed to work
    directly with parquet files instead of SQL databases.
    """

    def __init__(
        self,
        variable: str,
        bins: Union[int, np.ndarray] = 50,
        weight_name: Optional[str] = None,
    ):
        """Initialize the weight fitter.

        Args:
            variable: Name of the variable to reweight
            bins: Number of bins or bin edges for histogram
            weight_name: Name for the weight column (auto-generated if None)
        """
        self._variable = variable
        self._bins = bins
        self._weight_name = weight_name or self._generate_weight_name()

    def fit_weights_from_parquet(
        self,
        parquet_path: Union[str, Path],
        transform: Optional[Callable] = None,
        additional_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Fit weights from a parquet file.

        Args:
            parquet_path: Path to the parquet file
            transform: Optional transform to apply to the variable (e.g., np.log10)
            additional_columns: Additional columns to load from parquet

        Returns:
            DataFrame with weights added
        """
        # Load required columns
        columns_to_load = [self._variable]
        if additional_columns:
            columns_to_load.extend(additional_columns)

        df = pd.read_parquet(parquet_path, columns=columns_to_load)

        # Apply transform if provided
        if transform is not None:
            df[f"{self._variable}_transformed"] = transform(df[self._variable])
            variable_for_weights = f"{self._variable}_transformed"
        else:
            variable_for_weights = self._variable

        # Fit weights
        df = self._fit_weights(df, variable_for_weights)

        return df

    def _fit_weights(self, df: pd.DataFrame, variable: str) -> pd.DataFrame:
        """Fit per-event weights. To be implemented by subclasses."""
        raise NotImplementedError

    def _generate_weight_name(self) -> str:
        """Generate weight column name."""
        return f"{self._variable}_weight"


class UniformParquetWeighter(ParquetWeightFitter):
    """Uniform weighting for parquet files with optional capping.

    Args:
        variable: Column to re-weight.
        bins: Histogram bins (count or edges).
        max_weight: If given, clip weights at *max_weight × mean_weight* to
            avoid exploding factors in under-populated regions.
    """

    def __init__(self, variable: str, bins: Union[int, np.ndarray] = 50, *, max_weight: Optional[float] = None):
        super().__init__(variable=variable, bins=bins)
        self._max_weight = max_weight

    def _fit_weights(self, df: pd.DataFrame, variable: str) -> pd.DataFrame:  # noqa: D401 – keep GraphNeT style signature
        """Return dataframe with a new column of per-event weights."""

        # Histogram the variable
        bin_counts, bin_edges = np.histogram(df[variable], bins=self._bins)

        # Replace empty bins with a very small count ε to avoid div/0 → inf
        epsilon = 1e-6
        safe_counts = np.where(bin_counts == 0, epsilon, bin_counts)

        # Inverse histogram (flatten)
        bin_weights = 1.0 / safe_counts

        # Map every event to its bin weight
        ix = np.digitize(df[variable], bins=bin_edges, right=False) - 1
        ix = np.clip(ix, 0, len(bin_weights) - 1)
        sample_weights = bin_weights[ix]

        # Normalise so ⟨w⟩ = 1 (robust to zeros)
        mean_w = np.nanmean(sample_weights)
        if not np.isfinite(mean_w) or mean_w == 0:
            mean_w = 1.0
        sample_weights = sample_weights / mean_w

        # Optional cap of extreme weights
        if self._max_weight is not None:
            cap = self._max_weight
            sample_weights = np.minimum(sample_weights, cap)

        df[self._weight_name] = sample_weights
        return df

    def _generate_weight_name(self) -> str:  # noqa: D401
        return f"{self._variable}_uniform_weight"


class MAGICZenithEnergyWeighter:
    """Systematic weighter that only depends on zenith and energy."""

    def __init__(
        self,
        zenith_key: str = "telescope_theta",
        energy_key: str = "true_energy",
        zenith_bins: int = 30,
        energy_bins: int = 40,
        combined_weight_name: str = "systematic_weight",
        *,
        max_weight: Optional[float] = None,
    ):
        self.zenith_key = zenith_key
        self.energy_key = energy_key
        self.combined_weight_name = combined_weight_name

        self.zenith_weighter = UniformParquetWeighter(
            variable=zenith_key, bins=zenith_bins, max_weight=max_weight
        )
        self.energy_weighter = UniformParquetWeighter(
            variable=energy_key, bins=energy_bins, max_weight=max_weight
        )

        self._fitted_params: Optional[Dict[str, Dict[str, np.ndarray]]] = None

    def fit_systematic_weights_from_parquet(
        self,
        parquet_path: Union[str, Path],
        additional_columns: Optional[List[str]] = None,
        return_df: bool = False,
    ) -> pd.DataFrame | None:
        """Fit zenith+energy weights from parquet file."""
        required_cols = [self.zenith_key, self.energy_key]
        if additional_columns:
            required_cols.extend(additional_columns)

        df = pd.read_parquet(parquet_path, columns=required_cols)

        print(f"Loaded {len(df)} events from {parquet_path}")
        print(
            f"Zenith range: {np.rad2deg(df[self.zenith_key].min()):.1f}° - {np.rad2deg(df[self.zenith_key].max()):.1f}°"
        )
        print(
            f"Energy range: {df[self.energy_key].min():.2e} - {df[self.energy_key].max():.2e} TeV"
        )

        # Fit weights without re-loading the dataframe
        df = self.zenith_weighter._fit_weights(df, self.zenith_key)
        log_energy_col = "_log10_energy_temp"
        df[log_energy_col] = np.log10(df[self.energy_key])
        df = self.energy_weighter._fit_weights(df, log_energy_col)

        self._store_fitted_parameters(df)

        zenith_weights = df[self.zenith_weighter._weight_name]
        energy_weights = df[self.energy_weighter._weight_name]
        combined_weights = zenith_weights * energy_weights
        df[self.combined_weight_name] = combined_weights

        print("\nWeight statistics:")
        print(
            f"Zenith weights: {zenith_weights.min():.3f} - {zenith_weights.max():.3f} (mean: {zenith_weights.mean():.3f})"
        )
        print(
            f"Energy weights: {energy_weights.min():.3f} - {energy_weights.max():.3f} (mean: {energy_weights.mean():.3f})"
        )
        print(
            f"Combined weights: {combined_weights.min():.3f} - {combined_weights.max():.3f} (mean: {combined_weights.mean():.3f})"
        )

        if return_df:
            return df

    def _store_fitted_parameters(self, df: pd.DataFrame) -> None:
        zenith_values = df[self.zenith_key].values
        zenith_counts, zenith_edges = np.histogram(
            np.asarray(zenith_values), bins=self.zenith_weighter._bins
        )
        zenith_weights = 1.0 / np.where(zenith_counts == 0, np.nan, zenith_counts)
        zenith_weights = zenith_weights / np.nanmean(zenith_weights)

        energy_values = np.log10(df[self.energy_key].values)
        energy_counts, energy_edges = np.histogram(
            energy_values, bins=self.energy_weighter._bins
        )
        energy_weights = 1.0 / np.where(energy_counts == 0, np.nan, energy_counts)
        energy_weights = energy_weights / np.nanmean(energy_weights)

        self._fitted_params = {
            self.zenith_key: {"edges": zenith_edges, "weights": zenith_weights},
            self.energy_key: {"edges": energy_edges, "weights": energy_weights},
        }

    def get_weights_for_batch(self, batch_data: Data) -> Tensor:
        if self._fitted_params is None:
            print(
                "Warning: Systematic weighter not fitted yet. Call fit_systematic_weights_from_parquet first."
            )
            batch_size = len(batch_data[self.zenith_key])
            return torch.ones(batch_size, device=batch_data[self.zenith_key].device)

        try:
            zenith = batch_data[self.zenith_key]
            energy = batch_data[self.energy_key]
        except KeyError as e:
            print(f"KeyError: {self.zenith_key}, {self.energy_key} not found in batch_data")
            print(f"batch_data keys: {batch_data.keys()}")
            raise e

        device = zenith.device
        zenith_np = zenith.detach().cpu().numpy()
        energy_np = energy.detach().cpu().numpy()

        zenith_weights = self._apply_fitted_weights(zenith_np, self.zenith_key)
        log_energy = np.log10(energy_np)
        energy_weights = self._apply_fitted_weights(log_energy, self.energy_key)

        combined_weights = zenith_weights * energy_weights
        combined_weights = combined_weights / np.mean(combined_weights)

        if self.zenith_weighter._max_weight is not None:
            cap = self.zenith_weighter._max_weight
            combined_weights = np.minimum(combined_weights, cap)

        return torch.from_numpy(combined_weights).float().to(device)

    def _apply_fitted_weights(
        self, values: np.ndarray, variable_key: str
    ) -> np.ndarray:
        if self._fitted_params is None:
            print(
                "WARNING: Systematic weighter not fitted yet. Call fit_systematic_weights_from_parquet first."
            )
            return np.ones_like(values)

        params = self._fitted_params[variable_key]
        bin_edges = params["edges"]
        bin_weights = params["weights"]

        ix = np.digitize(values, bins=bin_edges, right=False) - 1
        ix = np.clip(ix, 0, len(bin_weights) - 1)

        sample_weights = bin_weights[ix]
        sample_weights = np.where(~np.isfinite(sample_weights), 0.0, sample_weights)

        return sample_weights

class MAGICSystematicWeighter:
    """Combined weight fitter for MAGIC systematic corrections.

    Handles zenith angle, energy (log10), and pointing offset corrections
    with uniform weighting for all three variables.
    """

    def __init__(
        self,
        zenith_key: str = "telescope_theta",
        energy_key: str = "true_energy",
        offset_key: str = "distance_from_center",
        zenith_bins: int = 30,
        energy_bins: int = 40,
        offset_bins: int = 25,
        combined_weight_name: str = "systematic_weight",
        *,
        max_weight: Optional[float] = None,
    ):
        """Initialize the combined systematic weighter.

        Args:
            zenith_key: Column name for zenith angle (radians)
            energy_key: Column name for true energy
            offset_key: Column name for pointing offset
            zenith_bins: Number of bins for zenith angle
            energy_bins: Number of bins for log10(energy)
            offset_bins: Number of bins for pointing offset
            combined_weight_name: Name for the final combined weight
        """
        self.zenith_key = zenith_key
        self.energy_key = energy_key
        self.offset_key = offset_key
        self.combined_weight_name = combined_weight_name

        # Individual weight fitters (share the same max_weight cap)
        self.zenith_weighter = UniformParquetWeighter(
            variable=zenith_key, bins=zenith_bins, max_weight=max_weight
        )

        self.energy_weighter = UniformParquetWeighter(
            variable=energy_key, bins=energy_bins, max_weight=max_weight
        )

        self.offset_weighter = UniformParquetWeighter(
            variable=offset_key, bins=offset_bins, max_weight=max_weight
        )

        # Store fitted parameters for on-the-fly computation
        self._fitted_params = None

    def fit_systematic_weights_from_parquet(
        self,
        parquet_path: Union[str, Path],
        additional_columns: Optional[List[str]] = None,
        return_df: bool = False,
    ) -> pd.DataFrame | None:
        """Fit combined systematic weights from parquet file.

        Args:
            parquet_path: Path to the parquet file
            additional_columns: Additional columns to load

        Returns:
            DataFrame with all individual weights and combined weight
        """
        # Load required columns
        required_cols = [self.zenith_key, self.energy_key, self.offset_key]
        if additional_columns:
            required_cols.extend(additional_columns)

        df = pd.read_parquet(parquet_path, columns=required_cols)

        print(f"Loaded {len(df)} events from {parquet_path}")
        print(
            f"Zenith range: {np.rad2deg(df[self.zenith_key].min()):.1f}° - {np.rad2deg(df[self.zenith_key].max()):.1f}°"
        )
        print(
            f"Energy range: {df[self.energy_key].min():.2e} - {df[self.energy_key].max():.2e} TeV"
        )
        print(
            f"Offset range: {df[self.offset_key].min():.3f} - {df[self.offset_key].max():.3f}"
        )

        # --- Fit individual weights on the SAME DataFrame (no re-read) ----

        # Zenith (use the raw radian value)
        df = self.zenith_weighter._fit_weights(df, self.zenith_key)

        # Energy (operate in log10 space but keep original column intact)
        log_energy_col = "_log10_energy_temp"
        df[log_energy_col] = np.log10(df[self.energy_key])
        df = self.energy_weighter._fit_weights(df, log_energy_col)

        # Offset (already in suitable units)
        df = self.offset_weighter._fit_weights(df, self.offset_key)

        # ------------------------------------------------------------------

        # Store fitted parameters for on-the-fly weight computation
        self._store_fitted_parameters(df)

        # Combine weights
        zenith_weights = df[self.zenith_weighter._weight_name]
        energy_weights = df[self.energy_weighter._weight_name]
        offset_weights = df[self.offset_weighter._weight_name]

        # Combined weight is product of individual weights
        combined_weights = zenith_weights * energy_weights * offset_weights

        # Do NOT normalise here – keep raw product. Normalisation will be
        # applied once per batch in `get_weights_for_batch` to avoid the
        # earlier double-normalisation bug.

        df[self.combined_weight_name] = combined_weights

        # Print statistics
        print("\nWeight statistics:")
        print(
            f"Zenith weights: {zenith_weights.min():.3f} - {zenith_weights.max():.3f} (mean: {zenith_weights.mean():.3f})"
        )
        print(
            f"Energy weights: {energy_weights.min():.3f} - {energy_weights.max():.3f} (mean: {energy_weights.mean():.3f})"
        )
        print(
            f"Offset weights: {offset_weights.min():.3f} - {offset_weights.max():.3f} (mean: {offset_weights.mean():.3f})"
        )
        print(
            f"Combined weights: {combined_weights.min():.3f} - {combined_weights.max():.3f} (mean: {combined_weights.mean():.3f})"
        )

        if return_df:
            return df

    def _store_fitted_parameters(self, df: pd.DataFrame):
        """Store fitted parameters for efficient on-the-fly weight computation."""
        # Store parameters for each variable

        # Zenith (degrees)
        zenith_values = df[self.zenith_key].values
        zenith_counts, zenith_edges = np.histogram(
            np.asarray(zenith_values), bins=self.zenith_weighter._bins
        )
        zenith_weights = 1.0 / np.where(zenith_counts == 0, np.nan, zenith_counts)
        zenith_weights = zenith_weights / np.nanmean(zenith_weights)

        # Energy (log10)
        energy_values = np.log10(df[self.energy_key].values)
        energy_counts, energy_edges = np.histogram(
            energy_values, bins=self.energy_weighter._bins
        )
        energy_weights = 1.0 / np.where(energy_counts == 0, np.nan, energy_counts)
        energy_weights = energy_weights / np.nanmean(energy_weights)

        # Offset
        offset_values = df[self.offset_key].values
        offset_counts, offset_edges = np.histogram(
            np.asarray(offset_values), bins=self.offset_weighter._bins
        )
        offset_weights = 1.0 / np.where(offset_counts == 0, np.nan, offset_counts)
        offset_weights = offset_weights / np.nanmean(offset_weights)

        # Store all parameters
        self._fitted_params = {
            "telescope_theta": {"edges": zenith_edges, "weights": zenith_weights},
            "true_energy": {"edges": energy_edges, "weights": energy_weights},
            "distance_from_center": {"edges": offset_edges, "weights": offset_weights},
        }

    def get_weights_for_batch(self, batch_data: Data) -> Tensor:
        """Compute systematic weights for a training batch on-the-fly.

        Args:
            batch_data: Dictionary containing batch data with zenith, energy, offset tensors

        Returns:
            Tensor of combined systematic weights for the batch
        """
        if self._fitted_params is None:
            print(
                "Warning: Systematic weighter not fitted yet. Call fit_systematic_weights_from_parquet first."
            )
            batch_size = len(batch_data[self.zenith_key])
            return torch.ones(batch_size, device=batch_data[self.zenith_key].device)

        # Extract variables
        try:
            zenith = batch_data[self.zenith_key]
            energy = batch_data[self.energy_key]
            offset = batch_data[self.offset_key]
        except KeyError as e:
            print(f"KeyError: {self.zenith_key}, {self.energy_key}, {self.offset_key} not found in batch_data")
            print(f"batch_data keys: {batch_data.keys()}")
            raise e

        device = zenith.device

        # Convert to numpy for weight computation
        zenith_np = zenith.detach().cpu().numpy()
        energy_np = energy.detach().cpu().numpy()
        offset_np = offset.detach().cpu().numpy()

        # Compute individual weights
        zenith_deg = np.rad2deg(zenith_np)
        zenith_weights = self._apply_fitted_weights(zenith_deg, "telescope_theta")

        log_energy = np.log10(energy_np)
        energy_weights = self._apply_fitted_weights(log_energy, "true_energy")

        offset_weights = self._apply_fitted_weights(offset_np, "distance_from_center")

        # Combine weights
        combined_weights = zenith_weights * energy_weights * offset_weights

        # Single normalisation pass (mean→1)
        combined_weights = combined_weights / np.mean(combined_weights)

        # Final optional cap inherited from individual weighters -------------
        if self.zenith_weighter._max_weight is not None:
            cap = self.zenith_weighter._max_weight
            combined_weights = np.minimum(combined_weights, cap)

        # Convert back to tensor
        weights_tensor = torch.from_numpy(combined_weights).float().to(device)

        return weights_tensor

    def _apply_fitted_weights(
        self, values: np.ndarray, variable_type: str
    ) -> np.ndarray:
        """Apply fitted weights to values using stored parameters."""
        if self._fitted_params is None:
            print(
                "WARNING: Systematic weighter not fitted yet. Call fit_systematic_weights_from_parquet first."
            )
            return np.ones_like(values)

        params = self._fitted_params[variable_type]
        bin_edges = params["edges"]
        bin_weights = params["weights"]

        # Assign weights to each sample
        ix = np.digitize(values, bins=bin_edges, right=False) - 1
        ix = np.clip(ix, 0, len(bin_weights) - 1)

        sample_weights = bin_weights[ix]
        sample_weights = np.where(~np.isfinite(sample_weights), 0.0, sample_weights)

        return sample_weights
