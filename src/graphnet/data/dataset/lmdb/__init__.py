"""LMDB-based dataset implementation for GraphNeT."""

from graphnet.utilities.imports import has_torch_package

if has_torch_package():
    try:
        import lmdb
        from .lmdb_dataset import LMDBDataset
    except ImportError:
        LMDBDataset = None

del has_torch_package 
