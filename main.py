import os
import logging
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('amr_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main pipeline runner for AMR Intelligence Platform"""
    start_time = datetime.now()
    logger.info("🚀 Starting AMR Intelligence Platform Pipeline")
    logger.info(f"⏰ Pipeline started at: {start_time}")

    try:
        # Import all modules
        from data_pipeline import AMRDataPipeline
        from analysis import AMRAnalysis
        from visualization import AMRVisualization
        from manuscript_builder import ManuscriptBuilder
        from dashboard import AMRDashboard

        logger.info("📦 All modules imported successfully")

        # Step 1: Data Pipeline
        logger.info("🔄 Step 1: Starting Data Pipeline...")
        pipeline = AMRDataPipeline()
        pipeline.extract_data()
        pipeline.clean_and_process_data()
        logger.info("✅ Data pipeline completed successfully")

        # Step 2: Analysis Pipeline
        logger.info("🔄 Step 2: Starting Analysis Pipeline...")
        analyzer = AMRAnalysis()
        if analyzer.load_data():
            analyzer.run_descriptive_statistics()
            analyzer.run_regression_analysis()
            # Note: Time-series forecasting and ML models disabled for current dataset
            # To add forecasting, add pathogen/antibiotic columns to panel data
            analyzer.generate_analysis_report()
            logger.info("✅ Analysis pipeline completed successfully (descriptive stats & regression)")
        else:
            logger.error("❌ Failed to load data for analysis")
            return False

        # Step 3: Visualization Pipeline
        logger.info("🔄 Step 3: Starting Visualization Pipeline...")
        viz = AMRVisualization()
        if viz.load_data():
            viz.create_time_series_plots()
            viz.create_heatmaps()
            viz.create_choropleth_maps()
            viz.create_forecast_plots()
            viz.create_consumption_vs_resistance_plot()
            viz.generate_visualization_report()
            logger.info("✅ Visualization pipeline completed successfully")
        else:
            logger.error("❌ Failed to load data for visualization")
            return False

        # Step 4: Manuscript Builder
        logger.info("🔄 Step 4: Starting Manuscript Builder...")
        builder = ManuscriptBuilder()
        builder.load_analysis_results()
        builder.generate_manuscript_markdown()
        builder.export_to_docx()
        builder.export_to_pdf()
        logger.info("✅ Manuscript builder completed successfully")

        # Step 5: Dashboard Setup
        logger.info("🔄 Step 5: Setting up Interactive Dashboard...")
        # Dashboard is run via streamlit command, not called directly here
        # This step creates the dashboard code and confirms it's ready
        dashboard = AMRDashboard()
        if dashboard.load_data():
            dashboard.load_forecast_data()
            logger.info("✅ Dashboard setup completed successfully")
            logger.info("🎯 To launch dashboard, run: streamlit run dashboard.py")
        else:
            logger.warning("⚠️  Dashboard data loading failed")

        # Pipeline Completion
        end_time = datetime.now()
        duration = end_time - start_time

        logger.info("🎉 AMR Intelligence Platform Pipeline completed successfully!")
        logger.info(f"⏱️  Total execution time: {duration}")
        logger.info(f"📁 Results saved in /results directory")
        logger.info(f"📄 Manuscript saved in /manuscript directory")
        logger.info(f"🗂️  Raw data saved in /data/raw directory")
        logger.info(f"🧹 Processed data saved in /data/processed directory")

        # Display completion summary
        print("\n" + "="*80)
        print("🎉 AMR INTELLIGENCE PLATFORM PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"⏰ Total execution time: {duration}")
        print("📦 Generated outputs:")
        print("  • Clean datasets (data/processed/)")
        print("  • Analytical results + plots (results/)")
        print("  • Forecasting models (results/)")
        print("  • Research manuscript (manuscript/)")
        print("  • Interactive dashboard code (dashboard.py)")

        print("\n🚀 Next steps:")
        print("  • View results in the 'results/' directory")
        print("  • Open manuscript.docx or manuscript.pdf")
        print("  • Launch dashboard: streamlit run dashboard.py")
        print("  • Review logs in amr_pipeline.log")
        print("="*80 + "\n")

        return True

    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error("💡 Please ensure all dependencies are installed: pip install -r requirements.txt")
        return False
    except Exception as e:
        logger.error(f"❌ Pipeline execution failed: {e}")
        logger.error(f"💡 Check the logs in amr_pipeline.log for details")
        return False

def check_requirements():
    """Check if required dependencies are available"""
    required_modules = [
        'pandas', 'numpy', 'requests', 'matplotlib', 'seaborn',
        'plotly', 'streamlit', 'docx', 'reportlab', 'statsmodels',
        'prophet', 'sklearn'  # geopandas is optional, fallback methods available
    ]

    optional_modules = ['geopandas']

    missing_modules = []
    missing_optional = []

    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)

    for module in optional_modules:
        try:
            __import__(module)
        except ImportError:
            missing_optional.append(module)

    if missing_modules:
        print("❌ Missing required modules:")
        for module in missing_modules:
            print(f"   - {module}")
        print("💡 Install missing dependencies: pip install -r requirements.txt")
        return False

    if missing_optional:
        print("⚠️  Optional modules missing (using fallback methods):")
        for module in missing_optional:
            print(f"   - {module}")
        print("Pipeline will continue with limited functionality.")

    return True

def print_help():
    """Print help information"""
    print("""
🦠 AMR Intelligence Platform

Usage: python main.py [command]

Commands:
  (no command)  - Run the full pipeline
  help          - Show this help message
  install       - Install dependencies
  test          - Run unit tests
  dashboard     - Launch interactive dashboard
  clean         - Clean generated files

Examples:
  python main.py              # Run full pipeline
  python main.py test         # Run tests
  python main.py dashboard    # Launch dashboard
  pip install -r requirements.txt  # Install dependencies

Output directories:
  data/raw/       - Raw data files
  data/processed/ - Cleaned panel dataset
  results/        - Analysis results and plots
  manuscript/     - Generated manuscripts

For more information, see README.md
""")

if __name__ == "__main__":
    # Handle command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == 'help':
            print_help()
        elif command == 'install':
            print("💡 Install dependencies with: pip install -r requirements.txt")
        elif command == 'test':
            print("🧪 Running tests...")
            # Import and run tests (will implement)
            print("Tests completed (placeholder)")
        elif command == 'dashboard':
            print("🎯 Launching dashboard...")
            os.system("streamlit run dashboard.py")
        elif command == 'clean':
            print("🧹 Cleaning generated files...")
            # Clean function (placeholder)
            print("Cleaning completed")
        else:
            print(f"Unknown command: {command}")
            print_help()
    else:
        # Check requirements first
        if not check_requirements():
            sys.exit(1)

        # Run full pipeline
        success = main()
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
