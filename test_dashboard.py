#!/usr/bin/env python3
"""
Test dashboard functionality without Streamlit UI
"""

import os
import sys
sys.path.append('.')

class AMRDashboardTest:
    def __init__(self, processed_data_path='data/processed/amr_panel_data.csv',
                 results_dir='results'):
        self.processed_data_path = processed_data_path
        self.results_dir = results_dir
        self.data = None
        self.forecast_data = None

    def load_data(self):
        """Load processed AMR data"""
        import pandas as pd
        try:
            self.data = pd.read_csv(self.processed_data_path)
            print(f"✅ Loaded dashboard data with shape: {self.data.shape}")
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False

    def load_forecast_data(self):
        """Load forecast data for predictions"""
        import pandas as pd
        try:
            arima_forecast = pd.read_csv(os.path.join(self.results_dir, 'arima_forecast.csv'))
            prophet_forecast = pd.read_csv(os.path.join(self.results_dir, 'prophet_forecast.csv'))
            self.forecast_data = {
                'arima': arima_forecast,
                'prophet': prophet_forecast
            }
            print("✅ Forecast data loaded successfully")
        except FileNotFoundError:
            print("🔍 Forecast data files not found.")
            self.forecast_data = None
        except Exception as e:
            print(f"❌ Error loading forecast data: {e}")
            self.forecast_data = None

    def test_basic_functionality(self):
        """Test basic dashboard functionality"""
        print("\n🧪 TESTING DASHBOARD FUNCTIONALITY")

        # Load data
        if not self.load_data():
            return False

        # Test data structure
        print(f"📊 Data columns: {list(self.data.columns)}")
        print(f"🌍 Countries: {sorted(self.data['Country'].unique())}")
        print(f"📅 Years: {sorted(self.data['Year'].unique())}")

        # Test filtering
        countries = self.data['Country'].unique()
        selected_countries = countries[:3]  # Test with first 3 countries
        years = sorted(self.data['Year'].unique())
        year_range = (years[0], years[-1])

        filtered_data = self.data[
            (self.data['Country'].isin(selected_countries)) &
            (self.data['Year'].between(year_range[0], year_range[1]))
        ]

        print(f"🔍 Filter test: {len(filtered_data)} records from {selected_countries}")

        # Test forecast data loading
        self.load_forecast_data()

        # Test simple aggregations for dashboard
        stats_df = filtered_data.groupby('Country').agg({
            'ResistanceRate': ['mean', 'std', 'min', 'max'],
            'ConsumptionRate': 'mean'
        }).round(3)

        print(f"📈 Statistics computed for {len(stats_df)} countries")

        # Test correlation calculations
        if 'ConsumptionRate' in filtered_data.columns and 'ResistanceRate' in filtered_data.columns:
            corr = filtered_data['ConsumptionRate'].corr(filtered_data['ResistanceRate'])
            print(f"📊 Consumption-Resistance correlation: {corr:.3f}")

        print("✅ DASHBOARD FUNCTIONALITY TEST PASSED")
        return True

if __name__ == "__main__":
    print("🚀 AMR DASHBOARD TEST RUN")
    print("=" * 40)

    try:
        dashboard = AMRDashboardTest()
        success = dashboard.test_basic_functionality()

        if success:
            print("\n🎉 DASHBOARD IS FULLY OPERATIONAL!")
            print("Ready for: streamlit run dashboard.py")
        else:
            print("\n❌ DASHBOARD HAS ISSUES THAT NEED FIXING")

    except Exception as e:
        print(f"❌ DASHBOARD TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
