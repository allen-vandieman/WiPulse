#!/usr/bin/env python3

# ----------------------------------------------------------------------
# Usage: python main.py csi_capture_*.csv [--filter] [--plot-only]
# 
# Main application that orchestrates:
# 1. Data loading (read_csv_csi.py)
# 2. Filtering (filters.py)
# 3. Plotting (plot_csv.py)
# ----------------------------------------------------------------------

import sys
import argparse
from pathlib import Path

# Import modules
from read_csv_csi import CSIDataLoader
from filters import CSIFilter, FilterConfig
from plot_csv import CSIPlotter, create_all_plots


class CSIProcessor:
    """Main application orchestrator for CSI processing"""
    
    def __init__(self):
        self.loader = None
        self.filter_obj = None
        self.plotter = None
        self.csi_data = None
        self.metadata = None
        self.filtered_data = None
        self.problematic_mask = None
        
    def load_data(self, csv_file):
        """Load CSI data from CSV file"""
        print("\n" + "="*60)
        print("STEP 1: Loading CSI Data")
        print("="*60)
        
        self.loader = CSIDataLoader(csv_file)
        self.csi_data, self.metadata = self.loader.load()
        
        print(f"\nLoaded {self.metadata['n_packets']} packets")
        print(f"with {self.metadata['n_subcarriers']} subcarriers each")
        print(f"Duration: {self.metadata['duration']:.3f} seconds")
        print(f"Rate: {self.metadata['rate']:.2f} Hz")
        
        return self.csi_data, self.metadata
    
    def apply_filters(self, config=None, verbose=True):
        """Apply filters to the loaded data"""
        if self.csi_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        print("\n" + "="*60)
        print("STEP 2: Applying Filters")
        print("="*60)
        
        if config is None:
            config = FilterConfig()
        
        self.filter_obj = CSIFilter(config)
        self.filtered_data = self.filter_obj.apply_all_filters(self.csi_data, verbose)
        self.problematic_mask = self.filter_obj.problematic_mask
        
        return self.filtered_data, self.problematic_mask
    
    def create_plots(self, output_dir='graphs', prefix='', 
                    create_comparison=True, create_filtered=True):
        """Generate all plots"""
        if self.csi_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        print("\n" + "="*60)
        print("STEP 3: Generating Plots")
        print("="*60)
        
        self.plotter = CSIPlotter(output_dir)
        
        # Original plots
        print("\nGenerating original data plots...")
        create_all_plots(self.csi_data, self.metadata, output_dir, prefix)
        
        # Filtered plots if available
        if self.filtered_data is not None:
            if create_comparison:
                print("\nGenerating comparison plots...")
                self.plotter.plot_filter_comparison(
                    self.csi_data, self.filtered_data, 
                    prefix='filter_'
                )
            
            if create_filtered:
                print("\nGenerating filtered data plots...")
                create_all_plots(
                    self.filtered_data, self.metadata, 
                    output_dir, 'filtered_'
                )
            
            if self.problematic_mask is not None and np.any(self.problematic_mask):
                print("\nGenerating problematic subcarrier plots...")
                self.plotter.plot_problematic_subcarriers(
                    self.csi_data, self.filtered_data, 
                    self.problematic_mask, prefix='filter_'
                )
        
        return self.plotter
    
    def save_filtered_data(self, output_file=None):
        """Save filtered data to CSV"""
        if self.filtered_data is None:
            raise ValueError("No filtered data available. Run apply_filters() first.")
        
        if self.loader is None:
            raise ValueError("No loader available. Call load_data() first.")
        
        return self.loader.save_filtered_csv(self.filtered_data, output_file)
    
    def run_full_pipeline(self, csv_file, output_dir='graphs', 
                         filter_config=None, save_filtered=True):
        """Run the complete processing pipeline"""
        # Load data
        self.load_data(csv_file)
        
        # Apply filters
        self.apply_filters(filter_config)
        
        # Generate plots
        self.create_plots(output_dir)
        
        # Save filtered data
        if save_filtered and self.filtered_data is not None:
            self.save_filtered_data()
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE!")
        print("="*60)
        print(f"Output directory: {output_dir}/")
        if save_filtered and self.filtered_data is not None:
            filtered_file = csv_file.replace('.csv', '_filtered.csv')
            print(f"Filtered data: {filtered_file}")
    
    def run_analysis_only(self, csv_file, output_dir='graphs'):
        """Run only analysis (load + plot), no filtering"""
        self.load_data(csv_file)
        self.create_plots(output_dir, create_comparison=False, create_filtered=False)


def main():
    parser = argparse.ArgumentParser(
        description='CSI Data Processing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline (load, filter, plot)
  python main.py csi_capture_20240101_120000.csv
  
  # Analysis only (no filtering)
  python main.py csi_capture_*.csv --analysis-only
  
  # Plot only (load existing processed data)
  python main.py csi_capture_*.csv --plot-only
  
  # Custom output directory
  python main.py csi_capture_*.csv --output-dir my_results
        """
    )
    
    parser.add_argument('csv_file', help='CSI CSV file to process')
    parser.add_argument('--analysis-only', action='store_true',
                       help='Only load and plot, do not filter')
    parser.add_argument('--plot-only', action='store_true',
                       help='Only generate plots from loaded data (for debugging)')
    parser.add_argument('--output-dir', default='graphs',
                       help='Output directory for plots (default: graphs)')
    parser.add_argument('--no-save-filtered', action='store_true',
                       help='Do not save filtered CSV data')
    parser.add_argument('--notch-q', type=float, default=30.0,
                       help='Quality factor for notch filter (default: 30.0)')
    parser.add_argument('--hampel-window', type=int, default=11,
                       help='Window size for Hampel filter (default: 11)')
    parser.add_argument('--hampel-threshold', type=float, default=3.0,
                       help='Threshold for Hampel filter (default: 3.0)')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.csv_file).exists():
        print(f"Error: File '{args.csv_file}' not found!")
        sys.exit(1)
    
    try:
        processor = CSIProcessor()
        
        if args.plot_only:
            # For plot-only mode, load data and generate plots
            processor.load_data(args.csv_file)
            processor.create_plots(args.output_dir, 
                                 create_comparison=False, 
                                 create_filtered=False)
            print(f"\nPlots saved to '{args.output_dir}/'")
            
        elif args.analysis_only:
            # Analysis only: load + plot, no filtering
            processor.run_analysis_only(args.csv_file, args.output_dir)
            print(f"\nAnalysis complete. Plots saved to '{args.output_dir}/'")
            
        else:
            # Full pipeline with filtering
            # Configure filters
            config = FilterConfig()
            config.notch_Q = args.notch_q
            config.hampel_window_size = args.hampel_window
            config.hampel_threshold = args.hampel_threshold
            
            # Run full pipeline
            processor.run_full_pipeline(
                args.csv_file, 
                args.output_dir,
                filter_config=config,
                save_filtered=not args.no_save_filtered
            )
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()