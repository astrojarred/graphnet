"""Unit tests for LMDBDataset."""

import io
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

try:
    import lmdb
    LMDB_AVAILABLE = True
except ImportError:
    LMDB_AVAILABLE = False

from graphnet.utilities.imports import has_torch_package

if has_torch_package() and LMDB_AVAILABLE:
    from graphnet.data.dataset.lmdb import LMDBDataset
    from graphnet.models.graphs import KNNGraph
    from graphnet.models.detector.prometheus import ORCA150


@pytest.mark.skipif(not LMDB_AVAILABLE, reason="LMDB not available")
@pytest.mark.skipif(not has_torch_package(), reason="PyTorch not available")
class TestLMDBDataset:
    """Test cases for LMDBDataset."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.lmdb")
        self.num_events = 10
        
        # Create test data
        self._create_test_data()
        
        # Set up graph definition
        detector = ORCA150()
        self.graph_definition = KNNGraph(
            detector=detector,
            nb_nearest_neighbours=4,  # 4 nearest neighbors for faster testing
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_data(self):
        """Create test LMDB database."""
        env = lmdb.open(
            self.db_path,
            map_size=int(1e8),  # 100MB
            subdir=True,
            readonly=False,
        )
        
        with env.begin(write=True) as txn:
            for event_id in range(self.num_events):
                # Create pulse data
                num_pulses = 20  # Fixed number for testing
                pulse_data = {
                    'sensor_pos_x': np.random.uniform(-100, 100, num_pulses),
                    'sensor_pos_y': np.random.uniform(-100, 100, num_pulses),
                    'sensor_pos_z': np.random.uniform(-100, 100, num_pulses),
                    't': np.random.uniform(0, 100, num_pulses),
                }
                
                # Store pulse data
                key = f"photons:{event_id}".encode('utf-8')
                buffer = io.BytesIO()
                torch.save(pulse_data, buffer)
                txn.put(key, buffer.getvalue())
                
                # Create truth data
                truth_data = {
                    'event_no': event_id,
                    'energy': float(10.0 + event_id),  # Predictable values for testing
                    'azimuth': float(event_id * 0.1),
                    'zenith': float(event_id * 0.05),
                    'pid': 14,  # Muon neutrino
                }
                
                # Store truth data
                key = f"truth:{event_id}".encode('utf-8')
                buffer = io.BytesIO()
                torch.save(truth_data, buffer)
                txn.put(key, buffer.getvalue())
        
        env.close()
    
    def test_dataset_creation(self):
        """Test that dataset can be created."""
        dataset = LMDBDataset(
            path=self.db_path,
            pulsemaps=["photons"],
            features=["sensor_pos_x", "sensor_pos_y", "sensor_pos_z", "t"],
            truth=["energy", "azimuth", "zenith", "pid"],
            graph_definition=self.graph_definition,
        )
        
        assert len(dataset) == self.num_events
    
    def test_dataset_getitem(self):
        """Test that individual items can be retrieved."""
        dataset = LMDBDataset(
            path=self.db_path,
            pulsemaps=["photons"],
            features=["sensor_pos_x", "sensor_pos_y", "sensor_pos_z", "t"],
            truth=["energy", "azimuth", "zenith", "pid"],
            graph_definition=self.graph_definition,
        )
        
        # Get first event
        graph = dataset[0]
        
        # Check that it's a valid graph
        assert hasattr(graph, 'x')  # Node features
        assert hasattr(graph, 'edge_index')  # Edges
        assert hasattr(graph, 'energy')  # Truth values
        assert hasattr(graph, 'event_no')  # Event number
        
        # Check shapes
        assert graph.x.shape[1] == 4  # 4 features
        assert graph.x.shape[0] == 20  # 20 pulses
        
        # Check truth values match what we stored
        assert graph.energy.item() == 10.0  # First event has energy 10.0
        assert graph.event_no.item() == 0   # First event has event_no 0
    
    def test_query_table(self):
        """Test the query_table method directly."""
        dataset = LMDBDataset(
            path=self.db_path,
            pulsemaps=["photons"],
            features=["sensor_pos_x", "sensor_pos_y", "sensor_pos_z", "t"],
            truth=["energy", "azimuth", "zenith", "pid"],
            graph_definition=self.graph_definition,
        )
        
        # Query truth table
        truth_data = dataset.query_table(
            table="truth",
            columns=["energy", "event_no"],
            sequential_index=0
        )
        
        assert truth_data.shape == (1, 2)  # One row, two columns requested
        
        # Query pulse table
        pulse_data = dataset.query_table(
            table="photons",
            columns=["sensor_pos_x", "sensor_pos_y"],
            sequential_index=0
        )
        
        assert pulse_data.shape == (20, 2)  # 20 pulses, 2 features
    
    def test_missing_table(self):
        """Test handling of missing tables."""
        dataset = LMDBDataset(
            path=self.db_path,
            pulsemaps=["photons"],
            features=["sensor_pos_x", "sensor_pos_y", "sensor_pos_z", "t"],
            truth=["energy", "azimuth", "zenith", "pid"],
            graph_definition=self.graph_definition,
        )
        
        with pytest.raises(IndexError):
            dataset.query_table(
                table="nonexistent_table",
                columns=["dummy_column"],
                sequential_index=0
            )
    
    def test_multiprocessing_pickling(self):
        """Test that dataset can be pickled/unpickled for multiprocessing."""
        dataset = LMDBDataset(
            path=self.db_path,
            pulsemaps=["photons"],
            features=["sensor_pos_x", "sensor_pos_y", "sensor_pos_z", "t"],
            truth=["energy", "azimuth", "zenith", "pid"],
            graph_definition=self.graph_definition,
        )
        
        import pickle
        
        # Test pickling
        pickled = pickle.dumps(dataset)
        unpickled = pickle.loads(pickled)
        
        # Test that unpickled dataset works
        graph = unpickled[0]
        assert hasattr(graph, 'x')
        assert hasattr(graph, 'energy') 
