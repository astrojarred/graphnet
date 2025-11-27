#!/usr/bin/env python3
"""
Standalone script to regenerate plots from existing CSV results.
"""
import sys
import pandas as pd
from pathlib import Path

# Add parent directory to path to import evaluation functions
sys.path.insert(0, str(Path(__file__).parent))

from evaluate_energy_reco import compute_energy_metrics, plot_energy_results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Regenerate plots from CSV results")
    parser.add_argument("--csv", required=True, help="Path to CSV file with results")
    parser.add_argument("--output-dir", default=None, help="Output directory for plot (default: same as CSV)")
    parser.add_argument("--energy-unit", default="GeV", help="Energy unit for plots")
    parser.add_argument("--energy-pred-key", default="energy_pred", help="Column name for predicted energy")
    parser.add_argument("--energy-true-key", default="true_energy", help="Column name for true energy")
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = csv_path.parent
    
    print(f"Loading results from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df):,} events")
    print(f"Columns: {list(df.columns)}")
    
    # Compute metrics
    print("Computing metrics...")
    metrics = compute_energy_metrics(
        df,
        energy_pred_key=args.energy_pred_key,
        energy_true_key=args.energy_true_key,
        energy_unit=args.energy_unit
    )
    
    # Generate plot
    print(f"Generating plot in: {output_dir}")
    plot_energy_results(
        df,
        metrics,
        output_dir,
        show_plots=False,
        save_plots=True,
        energy_pred_key=args.energy_pred_key,
        energy_true_key=args.energy_true_key,
        energy_unit=args.energy_unit
    )
    
    print(f"✓ Plot saved to {output_dir / 'energy_reco_eval.png'}")

if __name__ == "__main__":
    main()


