"""Physics task-specific modules to be used as model "read-outs"."""

from .task import (
    Task,
    IdentityTask,
    StandardLearnedTask,
    StandardFlowTask,
)

from .reconstruction import (
    MAGICDirectionClassificationTask,
    DirectionReconstructionWithKappa,
    AzimuthReconstructionWithKappa,
    ZenithReconstructionWithKappa,
    EnergyReconstruction,
)

from .magic_reconstruction import (
    MAGICDirectionReconstructionVMF,
    MAGICDirectionClassification,
    MAGICHybridDirectionTask,
    MAGICAngularResolution,
)
