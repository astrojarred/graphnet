# MAGIC Baseline Multi-task GNN

A simple, standalone script for training Graph Neural Networks on MAGIC telescope data. No config files needed!

## What it does

This script trains a **single model** that simultaneously performs:

1. **🔍 Binary Classification**: Gamma rays vs protons
2. **⚡ Energy Reconstruction**: Predict energy of gamma ray events  
3. **🎯 Direction Reconstruction**: Predict arrival direction as unified (theta, phi) vector

## Quick Start

```bash
# Basic training
python magic_baseline.py --path /path/to/your/magic_data.lmdb --max-epochs 20

# With GPU acceleration
python magic_baseline.py --path /path/to/data.lmdb --gpus 0 1 --max-epochs 50

# With Weights & Biases logging
python magic_baseline.py --path /path/to/data.lmdb --wandb --wandb-project my-magic-project
```

## Key Features

- **📦 Self-contained**: No external config files required
- **🚀 Simple**: ~300 lines of clean, readable code
- **🎯 Multi-task**: Trains all 3 tasks simultaneously
- **📊 Auto-evaluation**: Generates performance metrics and plots
- **🔧 Configurable**: Command-line arguments for all key parameters

## Architecture

- **Graph**: K-nearest neighbors (8 neighbors) in X-Y-T space
- **Backbone**: DynEdge GNN with global pooling
- **Tasks**: MAGIC-specific reconstruction tasks (adapted from GraphNet's neutrino tasks)
  - `BinaryClassificationTask`: Gamma vs proton classification
  - `MAGICEnergyReconstruction`: Primary gamma-ray energy (10 GeV - 100 TeV)
  - `MAGICDirectionReconstruction`: Unified zenith+azimuth with VonMisesFisher2DLoss
- **Features**: 7D node features (x_cam, y_cam, t, tel_id, signal, telescope_phi, telescope_theta)

## CLI Arguments

### Data & Output
- `--path`: Path to LMDB dataset (required)
- `--output-dir`: Output directory (default: ./magic_baseline_results)

### Training Control
- `--max-epochs`: Maximum epochs (default: 50)
- `--early-stopping-patience`: Early stopping patience (default: 10)
- `--batch-size`: Batch size (default: 128)
- `--num-workers`: Data loading workers (default: 10)
- `--gpus`: GPU IDs to use (e.g., `--gpus 0 1`)
- `--output-dir`: Where to save results (default: ./magic_baseline_results)

### Resume & Checkpoints
- `--resume-from-checkpoint`: Resume training from checkpoint (.ckpt file)
- `--wandb-run-id`: Resume specific W&B run

### Evaluation Only
- `--eval-only`: Skip training, only run evaluation
- `--checkpoint`: Model checkpoint for evaluation (.pth file, required with --eval-only)
- `--use-test-data`: Use test split instead of validation for evaluation

### Advanced
- `--learning-rate`: Learning rate (default: 1e-3)

### Logging
- `--wandb`: Enable Weights & Biases logging
- `--wandb-project`: W&B project name

## Output Files

After training, you'll get:

```
magic_baseline_results/
├── model.pth              # Complete model
├── state_dict.pth         # Model weights only
├── model_config.yml       # Model configuration
├── results.csv            # Predictions on validation set
├── roc_curve.png          # Classification performance
└── energy_correlation.png # Energy reconstruction quality
```

## Example Output

```
==================================================
MAGIC BASELINE GNN RESULTS
==================================================
📊 CLASSIFICATION:
   • AUC: 0.8542
   • Accuracy: 0.7823
   • Total events: 15000

⚡ ENERGY (Gamma events only):
   • MAE: 0.1234
   • Relative bias: 2.15%
   • Gamma events: 7500

🎯 DIRECTION (Gamma events only):
   • Mean angular error: 0.68°
   • 68% containment: 0.45°

📁 Plots saved: roc_curve.png, energy_correlation.png
==================================================
```

## Customization

The script is designed to be easily modifiable. Key areas for customization:

1. **Graph connectivity**: Modify `nb_nearest_neighbours` in `create_model()`
2. **Architecture**: Change DynEdge parameters or try different backbones
3. **Loss functions**: Experiment with different loss functions for each task
4. **Features**: Modify the feature set in the data loading section
5. **Evaluation**: Add more sophisticated evaluation metrics

## Performance Tips

- **Batch size**: Start with 32, increase if you have large GPUs
- **Learning rate**: 1e-3 is usually good, try 3e-4 for more stable training
- **Epochs**: Start with 20-50 epochs, increase if training is improving
- **Early stopping**: Use patience of 5-10 to avoid overfitting

## Troubleshooting

**Out of memory?**
- Reduce `--batch-size`
- Use fewer GPUs
- Check your data size

**Poor performance?**
- Increase `--max-epochs`
- Try different `--learning-rate` (e.g., 3e-4, 1e-4)
- Check your data quality and labels

**Slow training?**
- Increase `--num-workers` (but not more than CPU cores)
- Use multiple GPUs: `--gpus 0 1 2 3`
- Increase `--batch-size` if memory allows

## Important Data Preparation Note

⚠️ **Critical**: Your data preparation step must create a `gamma_mask` column:

```python
# In your data preparation transform
gamma_mask = (particle_id == 0).float()  # 1.0 for gamma, 0.0 for proton
```

This ensures energy and direction reconstruction losses only apply to gamma events (not protons), which is scientifically correct and improves training.

## MAGIC-Specific Adaptations

⚠️ **Important**: This script uses **MAGIC-adapted reconstruction tasks** instead of GraphNet's default neutrino tasks:

### **Why the adaptations?**
- **GraphNet was designed for neutrino telescopes** with different energy ranges and coordinate systems
- **MAGIC gamma-ray telescopes** have specific physics requirements:
  - **Energy**: Primary particle energy (not deposited energy)
  - **Zenith range**: Limited to 5-35° from vertical (not full 0-180°)
  - **Coordinate system**: Telescope-centric rather than detector-centric

### **What's different?**
1. **`MAGICEnergyReconstruction`**: Optimized for gamma-ray primary energy reconstruction
2. **`MAGICDirectionReconstruction`**: **Unified direction task** that properly uses `VonMisesFisher2DLoss`
   - Predicts both zenith and azimuth simultaneously as a 2D direction vector
   - Handles spherical topology correctly (unlike separate θ/φ predictions)
   - More physically meaningful than independent angle predictions

These adaptations ensure the model learns physically meaningful ranges and relationships specific to gamma-ray astronomy.

### **Critical: Unified Direction Reconstruction**

⚠️ **Important Physics Note**: Direction is treated as a **single 2D vector** on the celestial sphere, not as separate zenith/azimuth angles:

- **Why**: The `VonMisesFisher2DLoss` requires both θ and φ to compute proper angular distances
- **Benefit**: Handles spherical topology correctly (e.g., azimuth wraparound at 0°/360°)
- **Result**: More accurate direction reconstruction than independent angle predictions

## Next Steps

This baseline gives you a working multi-task model. To improve performance:

### 🔧 **Immediate Improvements** (Easy to debug)
1. **Hyperparameter tuning**: Learning rate, batch size, architecture size
2. **Loss weighting**: Adjust task importance (e.g., `--task-weights 1.0 2.0 1.0 1.0`)
3. **Training duration**: More epochs with proper early stopping
4. **Feature scaling**: Experiment with different normalization strategies

### 🚀 **Advanced Improvements** (For later iterations)
1. **Inter-telescope connections**: Add temporal proximity links between telescopes
2. **Edge features**: Add distance/time differences as edge attributes
3. **3D direction vectors**: Predict direction as unit vector instead of separate θ/φ
4. **Uncertainty estimation**: Add prediction confidence measures
5. **Physics-informed losses**: Shower-plane coordinate transformations

### 🧪 **Research Directions** (Long-term)
1. **Attention mechanisms**: Focus on most relevant pixels/times
2. **Graph transformers**: Replace GNN backbone with transformer
3. **Multi-scale architectures**: Different resolutions for different tasks
4. **Ensemble methods**: Combine multiple specialized models

**Recommendation**: Start with immediate improvements first. Only move to advanced improvements once you have a well-debugged baseline and understand what's working/not working.

Happy training! 🚀 

## Usage Examples

### Basic Training
```bash
python magic_baseline.py --path data.lmdb --max-epochs 30
```

### With GPU and W&B Logging
```bash
python magic_baseline.py --path data.lmdb --max-epochs 50 --gpus 0 1 --wandb
```

### Resume Training from Checkpoint
```bash
python magic_baseline.py --path data.lmdb --resume-from-checkpoint ./results/checkpoints/epoch=25.ckpt
```

### Evaluation Only (No Training)
```bash
# Evaluate on validation set
python magic_baseline.py --path data.lmdb --eval-only --checkpoint ./results/model.pth

# Evaluate on test set  
python magic_baseline.py --path data.lmdb --eval-only --checkpoint ./results/model.pth --use-test-data
```

### Multiple GPUs with Early Stopping
```bash
python magic_baseline.py --path data.lmdb --gpus 0 1 --early-stopping-patience 5
``` 
 