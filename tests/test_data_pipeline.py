import unittest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline import AMRDataPipeline

class TestAMRDataPipeline(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.raw_dir = os.path.join(self.temp_dir, 'raw')
        self.processed_dir = os.path.join(self.temp_dir, 'processed')
        os.makedirs(self.raw_dir)
        os.makedirs(self.processed_dir)

        self.pipeline = AMRDataPipeline(self.raw_dir, self.processed_dir)

    def tearDown(self):
        """Tear down test fixtures after each test method."""
        shutil.rmtree(self.temp_dir)

    def test_init(self):
        """Test pipeline initialization."""
        self.assertEqual(self.pipeline.raw_dir, self.raw_dir)
        self.assertEqual(self.pipeline.processed_dir, self.processed_dir)
        self.assertTrue(os.path.exists(self.raw_dir))
        self.assertTrue(os.path.exists(self.processed_dir))

    def test_create_panel_dataframe(self):
        """Test panel dataframe creation."""
        # Create sample GLASS data
        glass_data = pd.DataFrame({
            'country': ['USA', 'USA', 'GBR', 'GBR'],
            'year': [2020, 2021, 2020, 2021],
            'pathogen': ['E. coli', 'E. coli', 'E. coli', 'E. coli'],
            'antibiotic': ['Ciprofloxacin', 'Ciprofloxacin', 'Ciprofloxacin', 'Ciprofloxacin'],
            'resistance_rate': [0.1, 0.15, 0.2, 0.18]
        })

        # Create sample consumption data
        consumption_data = pd.DataFrame({
            'country': ['USA', 'USA', 'GBR', 'GBR'],
            'year': [2020, 2021, 2020, 2021],
            'consumption_rate': [15.5, 16.2, 12.8, 13.1]
        })

        wb_data = {}

        panel_df = self.pipeline._create_panel_dataframe(glass_data, consumption_data, wb_data)

        # Check basic structure
        required_cols = ['Country', 'Year', 'Pathogen', 'Antibiotic', 'ResistanceRate', 'ConsumptionRate']
        for col in required_cols:
            self.assertIn(col, panel_df.columns)

        # Check data integrity
        self.assertEqual(len(panel_df), 4)
        self.assertTrue((panel_df['Country'] == 'USA').any())
        self.assertTrue((panel_df['Country'] == 'GBR').any())

    def test_handle_missing_values(self):
        """Test missing value imputation."""
        # Create test dataframe with NaN values
        test_df = pd.DataFrame({
            'Country': ['USA', 'GBR', 'FRA'],
            'Year': [2020, 2021, 2022],
            'Pathogen': ['E. coli'] * 3,
            'Antibiotic': ['Ciprofloxacin'] * 3,
            'ResistanceRate': [0.1, np.nan, 0.3],
            'ConsumptionRate': [15.0, 20.0, np.nan],
            'GDP': [60000, np.nan, 45000],
            'Sanitation': [95, 98, np.nan]
        })

        result_df = self.pipeline._handle_missing_values(test_df)

        # Check that no NaN values remain
        self.assertFalse(result_df.isnull().any().any())

        # Check that ResistanceRate values are reasonable
        self.assertTrue(all(result_df['ResistanceRate'] >= 0))
        self.assertTrue(all(result_df['ResistanceRate'] <= 1))

    def test_generate_sample_glass_data(self):
        """Test sample GLASS data generation."""
        self.pipeline._generate_sample_glass_data()

        # Check if sample file was created
        sample_file = os.path.join(self.raw_dir, 'who_glass_data.json')
        self.assertTrue(os.path.exists(sample_file))

        # Load and verify structure
        sample_data = pd.read_json(sample_file)
        required_cols = ['country', 'year', 'pathogen', 'antibiotic', 'resistance_rate']
        for col in required_cols:
            self.assertIn(col, sample_data.columns)

        # Check reasonable value ranges
        self.assertTrue(all(sample_data['resistance_rate'] >= 0))
        self.assertTrue(all(sample_data['resistance_rate'] <= 1))

    def test_generate_sample_consumption_data(self):
        """Test sample consumption data generation."""
        self.pipeline._generate_sample_consumption_data()

        # Check if sample file was created
        sample_file = os.path.join(self.raw_dir, 'resistancemap_data.json')
        self.assertTrue(os.path.exists(sample_file))

        # Load and verify structure
        sample_data = pd.read_json(sample_file)
        required_cols = ['country', 'year', 'consumption_rate']
        for col in required_cols:
            self.assertIn(col, sample_data.columns)

        # Check reasonable value ranges (10-50 DDD per 1000 inhabitants/day is reasonable)
        self.assertTrue(all(sample_data['consumption_rate'] >= 10))
        self.assertTrue(all(sample_data['consumption_rate'] <= 50))

if __name__ == '__main__':
    unittest.main()
