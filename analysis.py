import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AMRAnalysis:
    def __init__(self, processed_data_path='data/processed/amr_panel_data.csv', results_dir='results'):
        self.processed_data_path = processed_data_path
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.data = None

    def load_data(self):
        """Load processed AMR data"""
        try:
            self.data = pd.read_csv(self.processed_data_path)
            logger.info(f"Loaded data with shape: {self.data.shape}")
        except FileNotFoundError:
            logger.error(f"Data file not found: {self.processed_data_path}")
            return False
        return True

    def run_descriptive_statistics(self):
        """Generate descriptive statistics and summary tables"""
        logger.info("Running descriptive statistics...")

        # Overall statistics
        summary_stats = self.data.describe()
        summary_stats.to_csv(os.path.join(self.results_dir, 'descriptive_statistics.csv'))

        # Statistics by country
        country_stats = self.data.groupby('Country').agg({
            'ResistanceRate': ['mean', 'std', 'min', 'max'],
            'ConsumptionRate': ['mean', 'std'],
            'Sanitation': 'mean'  # Sanitation has NaNs, but we can aggregate
        }).round(3)
        country_stats.to_csv(os.path.join(self.results_dir, 'country_statistics.csv'))

        # Statistics by year (since we don't have pathogen/antibiotic columns)
        year_stats = self.data.groupby('Year').agg({
            'ResistanceRate': ['mean', 'std'],
            'ConsumptionRate': ['mean', 'std'],
            'Sanitation': 'mean'
        }).round(3)
        year_stats.to_csv(os.path.join(self.results_dir, 'year_statistics.csv'))

        logger.info("Descriptive statistics completed.")

    def run_regression_analysis(self):
        """Run regression analysis using available data"""
        logger.info("Running regression analysis...")

        try:
            # Prepare data for regression model
            # Use available variables: ResistanceRate ~ ConsumptionRate + Year (with country effects)
            from statsmodels.regression.linear_model import OLS

            # Simple OLS with numeric dummy for countries
            reg_data = self.data.dropna(subset=['ResistanceRate', 'ConsumptionRate'])

            if len(reg_data) > 0:
                # Create country dummy variables
                country_dummies = pd.get_dummies(reg_data['Country'], prefix='Country', drop_first=True)

                # Prepare features
                X_data = pd.DataFrame({
                    'ConsumptionRate': reg_data['ConsumptionRate'],
                    'Year': reg_data['Year']
                })
                X = pd.concat([X_data, country_dummies], axis=1)

                # Add constant
                X = sm.add_constant(X)

                # Fit OLS model
                model = OLS(reg_data['ResistanceRate'], X)
                results = model.fit()

                # Save results
                with open(os.path.join(self.results_dir, 'regression_results.txt'), 'w') as f:
                    f.write(str(results.summary()))

                logger.info("Regression analysis completed.")
            else:
                logger.warning("Not enough data for regression analysis.")
        except Exception as e:
            logger.error(f"Error in regression analysis: {e}")

    def run_time_series_forecasting(self):
        """Run time-series forecasting using ARIMA and Prophet"""
        logger.info("Running time-series forecasting...")

        # Focus on E. coli - Ciprofloxacin combination for forecasting
        focus_data = self.data[(self.data['Pathogen'] == 'Escherichia coli') &
                              (self.data['Antibiotic'] == 'Ciprofloxacin')]

        if len(focus_data) < 20:
            logger.warning("Insufficient time series data for forecasting.")
            return

        # Aggregate by year globally
        ts_data = focus_data.groupby('Year')['ResistanceRate'].mean().reset_index()

        # ARIMA forecasting
        self._arima_forecasting(ts_data)

        # Prophet forecasting
        self._prophet_forecasting(ts_data)

        logger.info("Time-series forecasting completed.")

    def _arima_forecasting(self, ts_data):
        """ARIMA forecasting"""
        try:
            # Fit ARIMA model
            model = ARIMA(ts_data['ResistanceRate'], order=(1, 1, 1))
            fitted_model = model.fit()

            # Forecast 10 years ahead
            forecast_steps = 10
            forecast = fitted_model.forecast(steps=forecast_steps)

            # Create forecast dataframe
            last_year = ts_data['Year'].max()
            forecast_years = range(last_year + 1, last_year + forecast_steps + 1)
            forecast_df = pd.DataFrame({
                'Year': forecast_years,
                'Predicted_ResistanceRate': forecast,
                'Model': 'ARIMA'
            })

            forecast_df.to_csv(os.path.join(self.results_dir, 'arima_forecast.csv'), index=False)

            # Plot forecast
            plt.figure(figsize=(10, 6))
            plt.plot(ts_data['Year'], ts_data['ResistanceRate'], label='Historical')
            plt.plot(forecast_years, forecast, label='ARIMA Forecast')
            plt.xlabel('Year')
            plt.ylabel('Resistance Rate')
            plt.title('ARIMA Forecast of E. coli Resistance to Ciprofloxacin')
            plt.legend()
            plt.savefig(os.path.join(self.results_dir, 'arima_forecast_plot.png'))
            plt.close()

        except Exception as e:
            logger.error(f"Error in ARIMA forecasting: {e}")

    def _prophet_forecasting(self, ts_data):
        """Prophet forecasting"""
        try:
            # Prepare data for Prophet
            prophet_data = ts_data.rename(columns={'Year': 'ds', 'ResistanceRate': 'y'})
            prophet_data['ds'] = pd.to_datetime(prophet_data['ds'], format='%Y')

            # Fit Prophet model
            model = Prophet()
            model.fit(prophet_data)

            # Create future dataframe
            future = model.make_future_dataframe(periods=10, freq='Y')
            forecast = model.predict(future)

            # Save forecast results
            forecast_results = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']][-10:]
            forecast_results['Year'] = forecast_results['ds'].dt.year
            forecast_results = forecast_results[['Year', 'yhat', 'yhat_lower', 'yhat_upper']]
            forecast_results.columns = ['Year', 'Predicted_ResistanceRate', 'Lower_CI', 'Upper_CI']
            forecast_results.to_csv(os.path.join(self.results_dir, 'prophet_forecast.csv'), index=False)

            # Plot forecast
            fig = model.plot(forecast)
            fig.savefig(os.path.join(self.results_dir, 'prophet_forecast_plot.png'))
            plt.close()

        except Exception as e:
            logger.error(f"Error in Prophet forecasting: {e}")

    def run_machine_learning_models(self):
        """Run machine learning predictive models"""
        logger.info("Running machine learning models...")

        # Prepare features and target
        features = ['ConsumptionRate', 'GDP', 'Sanitation', 'Year']
        target = 'ResistanceRate'

        # Filter data for E. coli - Ciprofloxacin
        ml_data = self.data[(self.data['Pathogen'] == 'Escherichia coli') &
                           (self.data['Antibiotic'] == 'Ciprofloxacin')].copy()
        ml_data = ml_data[features + [target]].dropna()

        if len(ml_data) < 20:
            logger.warning("Insufficient data for ML models.")
            return

        X = ml_data[features]
        y = ml_data[target]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_mse = mean_squared_error(y_test, rf_pred)
        rf_r2 = r2_score(y_test, rf_pred)

        # XGBoost (Gradient Boosting)
        xgb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_mse = mean_squared_error(y_test, xgb_pred)
        xgb_r2 = r2_score(y_test, xgb_pred)

        # Save model performance
        model_results = pd.DataFrame({
            'Model': ['Random Forest', 'XGBoost'],
            'MSE': [rf_mse, xgb_mse],
            'R2': [rf_r2, xgb_r2]
        })
        model_results.to_csv(os.path.join(self.results_dir, 'ml_model_performance.csv'), index=False)

        # Feature importance for Random Forest
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        feature_importance.to_csv(os.path.join(self.results_dir, 'feature_importance.csv'), index=False)

        logger.info("Machine learning models completed.")

    def generate_analysis_report(self):
        """Generate comprehensive analysis report"""
        try:
            with open(os.path.join(self.results_dir, 'analysis_report.txt'), 'w') as f:
                f.write("AMR Analysis Report\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Data shape: {self.data.shape}\n")
                f.write(f"Countries: {self.data['Country'].nunique()}\n")
                f.write(f"Pathogens: {self.data['Pathogen'].unique().tolist()}\n")
                f.write(f"Antibiotics: {self.data['Antibiotic'].unique().tolist()}\n")
                f.write(f"Year range: {self.data['Year'].min()} - {self.data['Year'].max()}\n\n")

                # Summary statistics
                f.write("Key Findings:\n")
                f.write(".3f")
                f.write(".3f")
                f.write("\n")

        except Exception as e:
            logger.error(f"Error generating analysis report: {e}")

if __name__ == "__main__":
    analyzer = AMRAnalysis()
    if analyzer.load_data():
        analyzer.run_descriptive_statistics()
        analyzer.run_regression_analysis()
        # Skip time series and ML for now since we need pathogen/antibiotic columns
        logger.info("Analysis completed successfully (descriptive stats and regression).")
    else:
        logger.error("Failed to load data for analysis.")
