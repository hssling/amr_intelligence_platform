import unittest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import AMRAnalysis

class TestAMRAnalysis(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.results_dir = os.path.join(self.temp_dir, 'results')

        # Create sample data
        self.sample_data = pd.DataFrame({
            'Country': ['USA', 'USA', 'GBR', 'GBR', 'FRA', 'FRA'] * 3,
            'Year': [2019, 2020, 2021] * 6,
            'Pathogen': ['Escherichia coli'] * 18,
            'Antibiotic': ['Ciprofloxacin'] * 18,
            'ResistanceRate': np.random.uniform(0.1, 0.8, 18),
            'ConsumptionRate': np.random.uniform(10, 50, 18),
            'GDP': np.random.uniform(30000, 70000, 18),
            'Sanitation': np.random.uniform(85, 100, 18)
        })

        # Save sample data
        os.makedirs(os.path.join(self.temp_dir, 'data', 'processed'), exist_ok=True)
        self.processed_file = os.path.join(self.temp_dir, 'data', 'processed', 'amr_panel_data.csv')
        self.sample_data.to_csv(self.processed_file, index=False)

        self.analysis = AMRAnalysis(self.processed_file, self.results_dir)

    def tearDown(self):
        """Tear down test fixtures after each test method."""
        shutil.rmtree(self.temp_dir)

    def test_load_data(self):
        """Test data loading."""
        result = self.analysis.load_data()
        self.assertTrue(result)
        self.assertIsNotNone(self.analysis.data)
        self.assertEqual(len(self.analysis.data), 18)

        # Check required columns
        required_cols = ['Country', 'Year', 'Pathogen', 'Antibiotic', 'ResistanceRate', 'ConsumptionRate', 'GDP', 'Sanitation']
        for col in required_cols:
            self.assertIn(col, self.analysis.data.columns)

    def test_run_descriptive_statistics(self):
        """Test descriptive statistics generation."""
        self.analysis.data = self.sample_data.copy()
        self.analysis.run_descriptive_statistics()

        # Check if output files were created
        expected_files = [
            'descriptive_statistics.csv',
            'country_statistics.csv',
            'pathogen_statistics.csv',
            'antibiotic_statistics.csv'
        ]

        for filename in expected_files:
            filepath = os.path.join(self.results_dir, filename)
            self.assertTrue(os.path.exists(filepath), f"File {filename} was not created")

            # Verify file contains data
            df = pd.read_csv(filepath)
            self.assertFalse(df.empty, f"File {filename} is empty")

    def test_run_time_series_forecasting_insufficient_data(self):
        """Test time series forecasting with insufficient data."""
        # Create analysis with very limited data
        small_data = self.sample_data.head(5).copy()
        small_data['Year'] = [2019, 2020, 2021, 2022, 2023]  # Single pathogen-antibiotic combo
        small_data['Pathogen'] = ['E. coli'] * 5
        small_data['Antibiotic'] = ['Ciprofloxacin'] * 5

        self.analysis.data = small_data
        # Should not raise error, just log warning
        try:
            self.analysis.run_time_series_forecasting()
        except Exception as e:
            self.fail(f"Time series forecasting raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()
