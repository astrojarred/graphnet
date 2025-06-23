"""Collection of loss functions.

All loss functions inherit from `LossFunction` which ensures a common syntax,
handles per-event weights, etc.
"""

from abc import abstractmethod
from typing import Any, Optional, Union, List, Dict

import numpy as np
import scipy.special
import torch
from torch import Tensor
from torch import nn
from torch.nn.functional import (
    one_hot,
    cross_entropy,
    binary_cross_entropy,
    softplus,
)
import torch.nn.functional as F

from graphnet.models.model import Model
from graphnet.utilities.decorators import final


class LossFunction(Model):
    """Base class for loss functions in `graphnet`."""

    def __init__(self, **kwargs: Any) -> None:
        """Construct `LossFunction`, saving model config."""
        super().__init__(**kwargs)

    @final
    def forward(  # type: ignore[override]
        self,
        prediction: Tensor,
        target: Tensor,
        weights: Optional[Tensor] = None,
        return_elements: bool = False,
    ) -> Tensor:
        """Forward pass for all loss functions.

        Args:
            prediction: Tensor containing predictions. Shape [N,P]
            target: Tensor containing targets. Shape [N,T]
            return_elements: Whether elementwise loss terms should be returned.
                The alternative is to return the averaged loss across examples.

        Returns:
            Loss, either averaged to a scalar (if `return_elements = False`) or
            elementwise terms with shape [N,] (if `return_elements = True`).
        """
        elements = self._forward(prediction, target)
        if weights is not None:
            elements = elements * weights
        assert elements.size(dim=0) == target.size(
            dim=0
        ), "`_forward` should return elementwise loss terms."

        return elements if return_elements else torch.mean(elements)

    @abstractmethod
    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Syntax like `.forward`, for implentation in inheriting classes."""


class MSELoss(LossFunction):
    """Mean squared error loss."""

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Implement loss calculation."""
        # Check(s)
        assert prediction.dim() == 2
        if target.dim() != prediction.dim():
            target = target.squeeze(1)
        assert prediction.size() == target.size()

        elements = torch.mean((prediction - target) ** 2, dim=-1)
        return elements


class RMSELoss(MSELoss):
    """Root mean squared error loss."""

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Implement loss calculation."""
        # Check(s)
        elements = super()._forward(prediction, target)
        elements = torch.sqrt(elements)
        return elements


class LogCoshLoss(LossFunction):
    """Log-cosh loss function.

    Acts like x^2 for small x; and like |x| for large x.
    """

    @classmethod
    def _log_cosh(cls, x: Tensor) -> Tensor:  # pylint: disable=invalid-name
        """Numerically stable version on log(cosh(x)).

        Used to avoid `inf` for even moderately large differences.
        See [https://github.com/keras-team/keras/blob/v2.6.0/keras/losses.py#L1580-L1617]
        """
        return x + softplus(-2.0 * x) - np.log(2.0)

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Implement loss calculation."""
        diff = prediction - target
        elements = self._log_cosh(diff)
        return elements


class CrossEntropyLoss(LossFunction):
    """Compute cross-entropy loss for classification tasks.

    Predictions are an [N, num_class]-matrix of logits (i.e., non-softmax'ed
    probabilities), and targets are an [N,1]-matrix with integer values in
    (0, num_classes - 1).
    """

    def __init__(
        self,
        options: Union[int, List[Any], Dict[Any, int]],
        *args: Any,
        **kwargs: Any,
    ):
        """Construct CrossEntropyLoss."""
        # Base class constructor
        super().__init__(*args, **kwargs)

        # Member variables
        self._options = options
        self._nb_classes: int
        if isinstance(self._options, int):
            assert self._options in [torch.int32, torch.int64]
            assert (
                self._options >= 2
            ), f"Minimum of two classes required. Got {self._options}."
            self._nb_classes = options  # type: ignore
        elif isinstance(self._options, list):
            self._nb_classes = len(self._options)  # type: ignore
        elif isinstance(self._options, dict):
            self._nb_classes = len(
                np.unique(list(self._options.values()))
            )  # type: ignore
        else:
            raise ValueError(
                f"Class options of type {type(self._options)} not supported"
            )

        self._loss = nn.CrossEntropyLoss(reduction="none")

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Transform outputs to angle and prepare prediction."""
        if isinstance(self._options, int):
            # Integer number of classes: Targets are expected to be in
            # (0, nb_classes - 1).

            # Target integers are positive
            assert torch.all(target >= 0)

            # Target integers are consistent with the expected number of class.
            assert torch.all(target < self._options)

            assert target.dtype in [torch.int32, torch.int64]
            target_integer = target

        elif isinstance(self._options, list):
            # List of classes: Mapping target classes in list onto
            # (0, nb_classes - 1). Example:
            #    Given options: [1, 12, 13, ...]
            #    Yields: [1, 13, 12] -> [0, 2, 1, ...]
            target_integer = torch.tensor(
                [self._options.index(value) for value in target]
            )

        elif isinstance(self._options, dict):
            # Dictionary of classes: Mapping target classes in dict onto
            # (0, nb_classes - 1). Example:
            #     Given options: {1: 0, -1: 0, 12: 1, -12: 1, ...}
            #     Yields: [1, -1, -12, ...] -> [0, 0, 1, ...]
            target_integer = torch.tensor(
                [self._options[int(value)] for value in target]
            )

        else:
            assert False, "Shouldn't reach here."

        target_one_hot: Tensor = one_hot(target_integer, self._nb_classes).to(
            prediction.device
        )

        return self._loss(prediction.float(), target_one_hot.float())


class BinaryCrossEntropyLoss(LossFunction):
    """Compute binary cross entropy loss.

    Predictions are vector probabilities (i.e., values between 0 and 1), and
    targets should be 0 and 1.
    """

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        return binary_cross_entropy(
            prediction.float(), target.float(), reduction="none"
        )


class BCEWithLogitsLoss(LossFunction):
    """Binary cross entropy loss with logits.
    
    This loss combines a Sigmoid layer and the BinaryCrossEntropyLoss in one 
    single class. It is safer than using a plain Sigmoid followed by 
    BinaryCrossEntropyLoss as it is numerically more stable and supports 
    automatic mixed precision (AMP).
    
    Predictions are raw logits (any real value), and targets should be 0 and 1.
    """

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Calculate BCE with logits loss.
        
        Args:
            prediction: Raw logits from model
            target: Binary targets (0 or 1)
            
        Returns:
            Loss values
        """
        return torch.nn.functional.binary_cross_entropy_with_logits(
            prediction.float(), target.float(), reduction="none"
        )


class FocalBCEWithLogitsLoss(LossFunction):
    """Focal Binary Cross Entropy Loss with Logits.
    
    Focal loss addresses class imbalance by down-weighting easy examples and 
    focusing on hard examples. This combines focal loss with BCE and logits 
    for numerical stability and AMP support.
    
    Reference: https://arxiv.org/abs/1708.02002
    
    Args:
        alpha: Weighting factor for rare class (typically 0.25 for binary)
        gamma: Focusing parameter (typically 2.0)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, **kwargs: Any) -> None:
        """Initialize Focal BCE with Logits Loss.
        
        Args:
            alpha: Weighting factor for positive class (default: 0.25)
            gamma: Focusing parameter (default: 2.0)
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Calculate Focal BCE with logits loss.
        
        Args:
            prediction: Raw logits from model
            target: Binary targets (0 or 1)
            
        Returns:
            Loss values
        """
        # Convert logits to probabilities
        p = torch.sigmoid(prediction.float())
        target = target.float()
        
        # Calculate BCE loss without reduction
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            prediction.float(), target, reduction="none"
        )
        
        # Calculate p_t
        p_t = p * target + (1 - p) * (1 - target)
        
        # Calculate alpha_t
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        # Calculate focal weight
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        # Apply focal weight to BCE loss
        focal_loss = focal_weight * bce_loss
        
        return focal_loss


class BCEWithLogitsAndLabelSmoothingLoss(LossFunction):
    """Binary Cross Entropy Loss with Logits and Label Smoothing.
    
    Label smoothing is a regularization technique that prevents the model from
    becoming overconfident by using soft targets instead of hard targets.
    
    Args:
        smoothing: Label smoothing factor (typically 0.05-0.1)
    """
    
    def __init__(self, smoothing: float = 0.05, **kwargs: Any) -> None:
        """Initialize BCE with Logits and Label Smoothing Loss.
        
        Args:
            smoothing: Label smoothing factor (default: 0.05)
        """
        super().__init__(**kwargs)
        self.smoothing = smoothing

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Calculate BCE with logits and label smoothing loss.
        
        Args:
            prediction: Raw logits from model
            target: Binary targets (0 or 1)
            
        Returns:
            Loss values
        """
        target = target.float()
        
        # Apply label smoothing
        # For binary classification: smooth_target = (1-smoothing)*target + smoothing*0.5
        smooth_target = (1 - self.smoothing) * target + self.smoothing * 0.5
        
        # Calculate BCE loss with smoothed targets
        return torch.nn.functional.binary_cross_entropy_with_logits(
            prediction.float(), smooth_target, reduction="none"
        )


class LogCMK(torch.autograd.Function):
    """MIT License.

    Copyright (c) 2019 Max Ryabinin

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    _____________________

    From [https://github.com/mryab/vmf_loss/blob/master/losses.py]
    Modified to use modified Bessel function instead of exponentially scaled ditto
    (i.e. `.ive` -> `.iv`) as indiciated in [1812.04616] in spite of suggestion in
    Sec. 8.2 of this paper. The change has been validated through comparison with
    exact calculations for `m=2` and `m=3` and found to yield the correct results.
    """

    @staticmethod
    def forward(
        ctx: Any, m: int, kappa: Tensor
    ) -> Tensor:  # pylint: disable=invalid-name,arguments-differ
        """Forward pass."""
        dtype = kappa.dtype
        ctx.save_for_backward(kappa)
        ctx.m = m
        ctx.dtype = dtype
        kappa = kappa.double()
        iv = torch.from_numpy(
            scipy.special.iv(m / 2.0 - 1, kappa.cpu().numpy())
        ).to(kappa.device)
        return (
            (m / 2.0 - 1) * torch.log(kappa)
            - torch.log(iv)
            - (m / 2) * np.log(2 * np.pi)
        ).type(dtype)

    @staticmethod
    def backward(
        ctx: Any, grad_output: Tensor
    ) -> Tensor:  # pylint: disable=invalid-name,arguments-differ
        """Backward pass."""
        kappa = ctx.saved_tensors[0]
        m = ctx.m
        dtype = ctx.dtype
        kappa = kappa.double().cpu().numpy()
        grads = -(
            (scipy.special.iv(m / 2.0, kappa))
            / (scipy.special.iv(m / 2.0 - 1, kappa))
        )
        return (
            None,
            grad_output
            * torch.from_numpy(grads).to(grad_output.device).type(dtype),
        )


class VonMisesFisherLoss(LossFunction):
    """General class for calculating von Mises-Fisher loss.

    Requires implementation for specific dimension `m` in which the target and
    prediction vectors need to be prepared.
    """

    @classmethod
    def log_cmk_exact(
        cls, m: int, kappa: Tensor
    ) -> Tensor:  # pylint: disable=invalid-name
        """Calculate $log C_{m}(k)$ term in von Mises-Fisher loss exactly."""
        return LogCMK.apply(m, kappa)

    @classmethod
    def log_cmk_approx(
        cls, m: int, kappa: Tensor
    ) -> Tensor:  # pylint: disable=invalid-name
        """Calculate $log C_{m}(k)$ term in von Mises-Fisher loss approx.

        [https://arxiv.org/abs/1812.04616] Sec. 8.2 with additional minus sign.
        """
        v = m / 2.0 - 0.5
        a = torch.sqrt((v + 1) ** 2 + kappa**2)
        b = v - 1
        return -a + b * torch.log(b + a)

    @classmethod
    def log_cmk(
        cls, m: int, kappa: Tensor, kappa_switch: float = 100.0
    ) -> Tensor:  # pylint: disable=invalid-name
        """Calculate $log C_{m}(k)$ term in von Mises-Fisher loss.

        Since `log_cmk_exact` is diverges for `kappa` >~ 700 (using float64
        precision), and since `log_cmk_approx` is unaccurate for small `kappa`,
        this method automatically switches between the two at `kappa_switch`,
        ensuring continuity at this point.
        """
        kappa_switch = torch.tensor([kappa_switch]).to(kappa.device)
        mask_exact = kappa < kappa_switch

        # Ensure continuity at `kappa_switch`
        offset = cls.log_cmk_approx(m, kappa_switch) - cls.log_cmk_exact(
            m, kappa_switch
        )
        ret = cls.log_cmk_approx(m, kappa) - offset
        ret[mask_exact] = cls.log_cmk_exact(m, kappa[mask_exact])
        return ret

    def _evaluate(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Calculate von Mises-Fisher loss for a vector in D dimensons.

        This loss utilises the von Mises-Fisher distribution, which is a
        probability distribution on the (D - 1) sphere in D-dimensional space.

        Args:
            prediction: Predicted vector, of shape [batch_size, D].
            target: Target unit vector, of shape [batch_size, D].

        Returns:
            Elementwise von Mises-Fisher loss terms.
        """
        # Check(s)
        assert prediction.dim() == 2
        assert target.dim() == 2
        assert prediction.size() == target.size()

        # Computing loss
        m = target.size()[1]
        k = torch.norm(prediction, dim=1)
        dotprod = torch.sum(prediction * target, dim=1)
        elements = -self.log_cmk(m, k) - dotprod
        return elements

    @abstractmethod
    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        raise NotImplementedError


class VonMisesFisher2DLoss(VonMisesFisherLoss):
    """von Mises-Fisher loss function vectors in the 2D plane."""

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Calculate von Mises-Fisher loss for an angle in the 2D plane.

        Args:
            prediction: Output of the model. Must have shape [N, 2] where 0th
                column is a prediction of `angle` and 1st column is an estimate
                of `kappa`.
            target: Target tensor, extracted from graph object.

        Returns:
            loss: Elementwise von Mises-Fisher loss terms. Shape [N,]
        """
        # Check(s)
        assert prediction.dim() == 2 and prediction.size()[1] == 2
        assert target.dim() == 2
        assert prediction.size()[0] == target.size()[0]

        # Formatting target
        angle_true = target[:, 0]
        t = torch.stack(
            [
                torch.cos(angle_true),
                torch.sin(angle_true),
            ],
            dim=1,
        )

        # Formatting prediction
        angle_pred = prediction[:, 0]
        kappa = prediction[:, 1]
        p = kappa.unsqueeze(1) * torch.stack(
            [
                torch.cos(angle_pred),
                torch.sin(angle_pred),
            ],
            dim=1,
        )

        return self._evaluate(p, t)


class EuclideanDistanceLoss(LossFunction):
    """Mean squared error in three dimensions."""

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Calculate 3D Euclidean distance between predicted and target.

        Args:
            prediction: Output of the model. Must have shape [N, 3]
            target: Target tensor, extracted from graph object.

        Returns:
            Elementwise von Mises-Fisher loss terms. Shape [N,]
        """
        return torch.sqrt(
            (prediction[:, 0] - target[:, 0]) ** 2
            + (prediction[:, 1] - target[:, 1]) ** 2
            + (prediction[:, 2] - target[:, 2]) ** 2
        )


class VonMisesFisher3DLoss(VonMisesFisherLoss):
    """von Mises-Fisher loss function vectors in the 3D plane."""

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Calculate von Mises-Fisher loss for a direction in the 3D.

        Args:
            prediction: Output of the model. Must have shape [N, 4] where
                columns 0, 1, 2 are predictions of `direction` and last column
                is an estimate of `kappa`.
            target: Target tensor, extracted from graph object.

        Returns:
            Elementwise von Mises-Fisher loss terms. Shape [N,]
        """
        target = target.reshape(-1, 3)
        # Check(s)
        assert prediction.dim() == 2 and prediction.size()[1] == 4
        assert target.dim() == 2
        assert prediction.size()[0] == target.size()[0]

        kappa = prediction[:, 3]
        p = kappa.unsqueeze(1) * prediction[:, [0, 1, 2]]
        return self._evaluate(p, target)


class EnsembleLoss(LossFunction):
    """Chain multiple loss functions together."""

    def __init__(
        self,
        loss_functions: List[LossFunction],
        loss_factors: List[float] = None,
        prediction_keys: Optional[List[List[int]]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Chain multiple loss functions together.

            Optionally apply a weight to each loss function contribution.

            E.g. Loss = RMSE*0.5 + LogCoshLoss*1.5

        Args:
            loss_functions: A list of loss functions to use.
                Each loss function contributes a term to the overall loss.
            loss_factors: An optional list of factors that will be mulitplied
            to each loss function contribution. Must be ordered according
            to `loss_functions`. If not given, the weights default to 1.
            prediction_keys: An optional list of lists of indices for which
                prediction columns to use for each loss function. If not
                given, all columns are used for all loss functions.
        """
        if loss_factors is None:
            # add weight of 1 - i.e no discrimination
            loss_factors = np.repeat(1, len(loss_functions)).tolist()

        assert len(loss_functions) == len(loss_factors)
        self._factors = loss_factors
        self._loss_functions = loss_functions

        if prediction_keys is not None:
            self._prediction_keys: Optional[List[List[int]]] = prediction_keys
        else:
            self._prediction_keys = None
        super().__init__(*args, **kwargs)

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Calculate loss using multiple loss functions.

        Args:
            prediction: Output of the model.
            target: Target tensor, extracted from graph object.

        Returns:
            Elementwise loss terms. Shape [N,]
        """
        if self._prediction_keys is None:
            prediction_keys = [list(range(prediction.size(1)))] * len(
                self._loss_functions
            )
        else:
            prediction_keys = self._prediction_keys
        for k, (loss_function, prediction_key) in enumerate(
            zip(self._loss_functions, prediction_keys)
        ):
            if k == 0:
                elements = self._factors[k] * loss_function._forward(
                    prediction=prediction[:, prediction_key], target=target
                )
            else:
                elements += self._factors[k] * loss_function._forward(
                    prediction=prediction[:, prediction_key], target=target
                )
        return elements


class RMSEVonMisesFisher3DLoss(EnsembleLoss):
    """Combine the VonMisesFisher3DLoss with RMSELoss."""

    def __init__(self, vmfs_factor: float = 0.05) -> None:
        """VonMisesFisher3DLoss with a RMSE penality term.

            The VonMisesFisher3DLoss will be weighted with `vmfs_factor`.

        Args:
            vmfs_factor: A factor applied to the VonMisesFisher3DLoss term.
            Defaults ot 0.05.
        """
        super().__init__(
            loss_functions=[RMSELoss(), VonMisesFisher3DLoss()],
            loss_factors=[1, vmfs_factor],
            prediction_keys=[[0, 1, 2], [0, 1, 2, 3]],
        )


class VonMisesFisher3DCosineLoss(LossFunction):
    """Combined von Mises-Fisher 3D and Cosine Similarity loss.
    
    This loss function combines VonMisesFisher3DLoss with a cosine similarity
    term, as used in the winning IceCube Kaggle competition solution.
    The combination provides better angular resolution by leveraging both
    the probabilistic von Mises-Fisher distribution and direct angular
    similarity optimization.
    
    Loss = VonMisesFisher3DLoss + (1 - CosineSimilarity)
    
    Reference: IceCube Neutrinos in Deep Ice Kaggle competition (8th place solution)
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the combined loss function."""
        super().__init__(**kwargs)
        self.vmf_loss = VonMisesFisher3DLoss()
        self.cosine_sim = nn.CosineSimilarity(dim=1)

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Calculate combined von Mises-Fisher and cosine similarity loss.

        Args:
            prediction: Output of the model. Must have shape [N, 4] where
                columns 0, 1, 2 are predictions of direction (x, y, z) and 
                column 3 is the concentration parameter kappa.
            target: Target tensor with shape [N, 3] containing unit direction vectors.

        Returns:
            Elementwise loss terms. Shape [N,]
        """
        # Ensure target is properly shaped
        target = target.reshape(-1, 3)
        
        # Check shapes
        assert prediction.dim() == 2 and prediction.size()[1] == 4
        assert target.dim() == 2 and target.size()[1] == 3
        assert prediction.size()[0] == target.size()[0]

        # Calculate von Mises-Fisher loss (using the existing implementation)
        vmf_loss_elements = self.vmf_loss._forward(prediction, target)

        # Calculate cosine similarity loss
        # Extract only the direction components (first 3 columns) 
        pred_direction = prediction[:, :3]
        
        # Calculate cosine similarity between prediction and target
        # Both vectors should already be normalized for direction tasks
        cosine_similarity = self.cosine_sim(pred_direction, target)
        
        # Convert to loss term: (1 - cosine_similarity) 
        # Note: Using elementwise operation, not mean() like in datasaurus
        # since LossFunction base class handles averaging
        cosine_loss_elements = 1.0 - cosine_similarity

        # Combine losses (elementwise)
        combined_loss = vmf_loss_elements + cosine_loss_elements

        return combined_loss


class CircularSmoothCrossEntropyLoss(LossFunction):
    """Circular-aware smooth cross-entropy loss for angular quantities.
    
    Based on IceCube Kaggle winning solutions. Implements smooth label 
    distributions that respect the circular nature of azimuth angles
    and the linear nature of zenith angles.
    
    For azimuth: Circular smoothing where bin 0 neighbors bin N-1
    For zenith: Linear smoothing with boundary handling
    
    Reference: IceCube Neutrinos in Deep Ice Kaggle (3rd and 5th place solutions)
    """
    
    def __init__(
        self,
        num_azimuth_bins: int = 48,
        num_zenith_bins: int = 48,
        azimuth_smoothing_strength: float = 7.2,
        zenith_smoothing_strength: float = 40.0,
        **kwargs: Any
    ):
        """Initialize circular smooth cross-entropy loss.
        
        Args:
            num_azimuth_bins: Number of azimuth bins (0-2π)
            num_zenith_bins: Number of zenith bins (0-π)
            azimuth_smoothing_strength: Smoothing parameter for azimuth (higher = sharper)
            zenith_smoothing_strength: Smoothing parameter for zenith (higher = sharper)
        """
        super().__init__(**kwargs)
        self.num_azimuth_bins = num_azimuth_bins
        self.num_zenith_bins = num_zenith_bins
        self.azimuth_smoothing_strength = azimuth_smoothing_strength
        self.zenith_smoothing_strength = zenith_smoothing_strength
        
        # Initialize bin centers
        self._initialize_bins()
        
    def _initialize_bins(self):
        """Initialize bin centers and precompute smoothing matrices."""
        import numpy as np
        
        # Azimuth bin centers (0 to 2π)
        azimuth_centers = np.linspace(0, 2*np.pi, self.num_azimuth_bins, endpoint=False)
        azimuth_centers += np.pi / self.num_azimuth_bins  # Shift to bin centers
        
        # Zenith bin centers (0 to π)
        zenith_centers = np.linspace(0, np.pi, self.num_zenith_bins, endpoint=False)
        zenith_centers += np.pi / (2 * self.num_zenith_bins)  # Shift to bin centers
        
        # Register as buffers so they move with model to GPU
        self.register_buffer('azimuth_centers', torch.tensor(azimuth_centers, dtype=torch.float32))
        self.register_buffer('zenith_centers', torch.tensor(zenith_centers, dtype=torch.float32))
        
    def _create_smooth_targets(self, true_angles: Tensor, angle_type: str) -> Tensor:
        """Create smooth target distributions for given angles.
        
        Args:
            true_angles: True angles [batch_size]
            angle_type: 'azimuth' or 'zenith'
            
        Returns:
            Smooth target distributions [batch_size, num_bins]
        """
        batch_size = true_angles.size(0)
        
        if angle_type == 'azimuth':
            bin_centers = self.azimuth_centers
            num_bins = self.num_azimuth_bins
            smoothing_strength = self.azimuth_smoothing_strength
            
            # Create circular distance matrix
            # For azimuth: use cosine distance for circular topology
            # Distance = 1 + cos(angle_diff)
            angle_diff = true_angles.unsqueeze(1) - bin_centers.unsqueeze(0)  # [batch, bins]
            smooth_targets = torch.softmax(
                (1.0 + torch.cos(angle_diff)) ** smoothing_strength, 
                dim=1
            )
            
        elif angle_type == 'zenith':
            bin_centers = self.zenith_centers
            num_bins = self.num_zenith_bins
            smoothing_strength = self.zenith_smoothing_strength
            
            # For zenith: use linear distance
            # Distance = π - |angle_diff|
            angle_diff = torch.abs(true_angles.unsqueeze(1) - bin_centers.unsqueeze(0))  # [batch, bins]
            smooth_targets = torch.softmax(
                (np.pi - angle_diff) * smoothing_strength,
                dim=1
            )
            
        else:
            raise ValueError(f"Unknown angle_type: {angle_type}")
            
        return smooth_targets
    
    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Calculate circular smooth cross-entropy loss.
        
        Args:
            prediction: Model logits [batch_size, num_azimuth_bins + num_zenith_bins]
            target: True angles [batch_size, 2] (azimuth=true_phi, zenith=true_theta)
            
        Returns:
            Elementwise loss terms [batch_size]
        """
        # Validate input shapes
        assert prediction.dim() == 2, f"Expected 2D prediction tensor, got {prediction.dim()}D"
        assert target.dim() == 2 and target.size(1) == 2, f"Expected [B,2] target tensor, got {target.shape}"
        assert prediction.size(1) == (self.num_azimuth_bins + self.num_zenith_bins), \
            f"Expected {self.num_azimuth_bins + self.num_zenith_bins} logits, got {prediction.size(1)}"
        
        # Split prediction logits
        azimuth_logits = prediction[:, :self.num_azimuth_bins]
        zenith_logits = prediction[:, self.num_azimuth_bins:]
        
        # Split target angles - task gives us [true_phi, true_theta]
        true_azimuth = target[:, 0]  # true_phi (azimuth)
        true_zenith = target[:, 1]   # true_theta (zenith)
        
        # Create smooth target distributions
        azimuth_smooth_targets = self._create_smooth_targets(true_azimuth, 'azimuth')
        zenith_smooth_targets = self._create_smooth_targets(true_zenith, 'zenith')
        
        # Calculate KL divergence losses
        # KL(smooth_targets || predicted_probs) = - sum(targets * log(probs))
        azimuth_log_probs = torch.log_softmax(azimuth_logits, dim=1)
        zenith_log_probs = torch.log_softmax(zenith_logits, dim=1)
        
        # Element-wise multiplication and sum over bin dimension
        azimuth_loss = -torch.sum(azimuth_smooth_targets * azimuth_log_probs, dim=1)
        zenith_loss = -torch.sum(zenith_smooth_targets * zenith_log_probs, dim=1)
        
        # Return the sum of losses for each event in the batch
        return azimuth_loss + zenith_loss


class MAGICVMFLoss(LossFunction):
    """Von Mises-Fisher loss for MAGIC directional data.
    
    Based on IceCube competition winners. The VMF distribution is the spherical
    analogue of the Gaussian distribution, perfect for directional data.
    Optimized for MAGIC telescope direction reconstruction.
    """
    
    def __init__(self, prediction_kappa_index: int = 3, **kwargs: Any):
        """Initialize MAGIC VMF loss.
        
        Args:
            prediction_kappa_index: Index of kappa in prediction tensor.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self.prediction_kappa_index = prediction_kappa_index
        
    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Calculate VMF loss for MAGIC direction data.
        
        Args:
            prediction: [batch, 4] with (dir_x, dir_y, dir_z, kappa)
            target: [batch, 3] true direction unit vectors
            
        Returns:
            Elementwise loss terms [batch,]
        """
        # Extract predicted direction and concentration
        pred_direction = prediction[:, :3]
        pred_direction = F.normalize(pred_direction, p=2, dim=1)
        kappa = prediction[:, self.prediction_kappa_index]
        
        # Ensure target is normalized
        target = F.normalize(target, p=2, dim=1)
        
        # VMF log likelihood: log(C(kappa)) + kappa * cos(angle)
        # where cos(angle) = dot product of unit vectors
        cos_angle = (pred_direction * target).sum(dim=1)
        
        # VMF log likelihood for 3D case (corrected approximation)
        # Full VMF normalizing constant: C_3(κ) = κ / (4π(eκ - e-κ))
        # For large κ: log C_3(κ) ≈ log(κ) - log(4π) - κ
        # For small κ: use simpler approximation
        
        # Clamp kappa to reasonable range to prevent explosion
        kappa = torch.clamp(kappa, min=1e-6, max=500.0)
        
        # Log normalizing constant (corrected)
        log_4pi = torch.log(torch.tensor(4 * torch.pi, device=kappa.device))
        log_c_kappa = torch.log(kappa + 1e-8) - log_4pi - kappa
        
        # VMF log likelihood: log(C(κ)) + κ * <μ, x>
        log_likelihood = log_c_kappa + kappa * cos_angle
        
        # Return NEGATIVE log likelihood for minimization
        nll = -log_likelihood
            
        return nll


class LocallySmoothedCrossEntropyLoss(LossFunction):
    """Locally smoothed cross-entropy for angular classification.
    
    From IceCube 3rd place: smooth labels across neighboring angular bins
    to account for uncertainty in bin boundaries.
    """
    
    def __init__(self, num_bins: int, smoothing_sigma: float = 0.5, **kwargs: Any):
        """Initialize locally smoothed cross-entropy loss.
        
        Args:
            num_bins: Number of angular bins.
            smoothing_sigma: Standard deviation for Gaussian smoothing.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self.num_bins = num_bins
        self.smoothing_sigma = smoothing_sigma
        
    def _forward(self, logits: Tensor, target_bins: Tensor) -> Tensor:
        """Calculate locally smoothed cross-entropy loss.
        
        Args:
            logits: [batch, num_bins] classification logits
            target_bins: [batch] target bin indices
            
        Returns:
            Elementwise loss terms [batch,]
        """
        batch_size = logits.shape[0]
        
        # Create smoothed labels
        bin_centers = torch.arange(self.num_bins, device=logits.device).float()
        target_bins_expanded = target_bins.unsqueeze(1).float()
        
        # Gaussian smoothing around true bin
        distances = (bin_centers - target_bins_expanded) ** 2
        smooth_labels = torch.exp(-distances / (2 * self.smoothing_sigma ** 2))
        smooth_labels = smooth_labels / smooth_labels.sum(dim=1, keepdim=True)
        
        # Cross entropy with smooth labels
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(smooth_labels * log_probs).sum(dim=1)
            
        return loss


class MAGICFocalLoss(LossFunction):
    """Focal loss for handling class imbalance in MAGIC angular bins.
    
    Particularly useful for fine-grained angular classification where most
    events fall in central bins near the pointing direction.
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, **kwargs: Any):
        """Initialize focal loss.
        
        Args:
            alpha: Weighting factor for rare classes.
            gamma: Focusing parameter (higher gamma focuses more on hard examples).
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        
    def _forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Calculate focal loss.
        
        Args:
            logits: [batch, num_classes] classification logits
            targets: [batch] target class indices
            
        Returns:
            Elementwise loss terms [batch,]
        """
        # Get class probabilities
        p = F.softmax(logits, dim=1)
        
        # Get probability of true class
        batch_size = logits.shape[0]
        p_true = p[torch.arange(batch_size), targets.long()]
        
        # Focal loss
        focal_weight = (1 - p_true) ** self.gamma
        loss = -self.alpha * focal_weight * torch.log(p_true + 1e-8)
            
        return loss


class CombinedVMFClassificationLoss(LossFunction):
    """Combined loss for MAGIC hybrid models.
    
    Balances VMF regression loss with classification loss, with dynamic weighting
    based on prediction confidence.
    """
    
    def __init__(
        self,
        vmf_weight: float = 0.7,
        classification_weight: float = 0.3,
        dynamic_weighting: bool = True,
        num_classification_outputs: int = 136,
        **kwargs: Any
    ):
        """Initialize combined loss.
        
        Args:
            vmf_weight: Weight for VMF regression loss.
            classification_weight: Weight for classification loss.
            dynamic_weighting: Whether to use dynamic weighting based on confidence.
            num_classification_outputs: Number of classification outputs.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self.vmf_loss = MAGICVMFLoss()
        self.classification_loss = LocallySmoothedCrossEntropyLoss(num_classification_outputs)
        self.vmf_weight = vmf_weight
        self.classification_weight = classification_weight
        self.dynamic_weighting = dynamic_weighting
        self.num_classification_outputs = num_classification_outputs
        
    def _forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Calculate combined VMF and classification loss.
        
        Args:
            predictions: Combined predictions [batch, 4 + num_bins + extras]
            targets: [batch, 4] with (dir_x, dir_y, dir_z, angular_bin)
            
        Returns:
            Elementwise loss terms [batch,]
        """
        # Split predictions
        vmf_pred = predictions[:, :4]  # direction + kappa
        classification_logits = predictions[:, 4:4+self.num_classification_outputs]
        
        # Split targets
        true_direction = targets[:, :3]
        true_bins = targets[:, 3].long()
        
        # Compute individual losses
        vmf_loss = self.vmf_loss._forward(vmf_pred, true_direction)
        class_loss = self.classification_loss._forward(classification_logits, true_bins)
        
        # Dynamic weighting based on prediction confidence
        if self.dynamic_weighting:
            # Use kappa as confidence measure
            kappa = vmf_pred[:, 3]
            confidence = torch.sigmoid(kappa / 10)  # Scale kappa to reasonable range
            avg_confidence = confidence.mean()
            
            # Higher confidence -> more weight on VMF
            vmf_w = self.vmf_weight * (0.5 + avg_confidence)
            class_w = self.classification_weight * (1.5 - avg_confidence)
            
            # Normalize weights
            total_w = vmf_w + class_w
            vmf_w = vmf_w / total_w
            class_w = class_w / total_w
        else:
            vmf_w = self.vmf_weight
            class_w = self.classification_weight
            
        return vmf_w * vmf_loss + class_w * class_loss
