"""GNN-specific modules, for performing the main learnable operations."""

from .convnet import ConvNet
from .dynedge import DynEdge
from .dynedge_jinst import DynEdgeJINST
from .dynedge_kaggle_tito import DynEdgeTITO
from .dynedge_stereo import DynEdgeStereo
from .dynedge_stereo_dir import DynEdgeStereoDir
from .RNN_tito import RNN_TITO
from .icemix import DeepIce
from .particlenet import ParticleNeT
from .magic_transformer import MAGICTransformer
from .magic_direction_classifier import MAGICDirectionClassifier
from .magic_hybrid_model import MAGICHybridModel
