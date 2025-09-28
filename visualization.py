import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Optional imports with fallbacks
try:
    import geopandas as gpd
    from shapely.geometry import Point
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AMRVisualization:
    def __init__(self, processed_data_path='data/processed/amr_panel_data.csv', results_dir='results'):
        self.processed_data_path = processed_data_path
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.data = None

    def load_data(self):
        """Load processed AMR data"""
        try:
            self.data = pd.read_csv(self.processed_data_path, parse_dates=['Year'])
            logger.info(f"Loaded data with shape: {self.data.shape}")
            # Ensure Year is int
            self.data['Year'] = self.data['Year'].astype(int)
        except FileNotFoundError:
            logger.error(f"Data file not found: {self.processed_data_path}")
            return False
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
        return True

    def create_time_series_plots(self):
        """Create time-series trends by country and globally"""
        logger.info("Creating time-series plots...")

        # Global trend for E. coli - Ciprofloxacin
        focus_data = self.data[(self.data['Pathogen'] == 'Escherichia coli') &
                              (self.data['Antibiotic'] == 'Ciprofloxacin')]

        if len(focus_data) > 0:
            global_trend = focus_data.groupby('Year')['ResistanceRate'].mean().reset_index()

            # Matplotlib plot
            plt.figure(figsize=(12, 6))
            plt.plot(global_trend['Year'], global_trend['ResistanceRate'], marker='o', linewidth=2)
            plt.title('Global E. coli Resistance to Ciprofloxacin Over Time', fontsize=14)
            plt.xlabel('Year', fontsize=12)
            plt.ylabel('Resistance Rate', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.results_dir, 'global_resistance_trend.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()

            # Plotly interactive plot
            fig = px.line(global_trend, x='Year', y='ResistanceRate',
                         title='Global E. coli Resistance to Ciprofloxacin',
                         markers=True)
            fig.update_layout(
                xaxis_title='Year',
                yaxis_title='Resistance Rate',
                template='plotly_white'
            )
            fig.write_html(os.path.join(self.results_dir, 'global_resistance_trend_interactive.html'))

        # By country trends
        plt.figure(figsize=(15, 10))
        for country in self.data['Country'].unique()[:8]:  # Top 8 countries
            country_data = self.data[(self.data['Country'] == country) &
                                   (self.data['Pathogen'] == 'Escherichia coli') &
                                   (self.data['Antibiotic'] == 'Ciprofloxacin')]
            if len(country_data) > 2:
                plt.plot(country_data['Year'], country_data['ResistanceRate'],
                        marker='o', label=country, alpha=0.7)

        plt.title('E. coli Resistance Trends by Country', fontsize=14)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Resistance Rate', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'country_resistance_trends.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("Time-series plots created.")

    def create_heatmaps(self):
        """Create heatmaps of resistance rates"""
        logger.info("Creating resistance heatmaps...")

        # Pivot data for heatmap
        heatmap_data = self.data[(self.data['Pathogen'] == 'Escherichia coli') &
                                (self.data['Antibiotic'] == 'Ciprofloxacin')]

        # Average by country and year
        pivot_data = heatmap_data.pivot_table(values='ResistanceRate',
                                             index='Country',
                                             columns='Year',
                                             aggfunc='mean')

        # Seaborn heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_data, cmap='RdYlGn_r', annot=True, fmt='.2f', linewidths=0.5)
        plt.title('E. coli Resistance to Ciprofloxacin Heatmap by Country and Year', fontsize=14)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Country', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'resistance_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Resistance by antibiotic heatmap
        antibiotic_pivot = self.data.pivot_table(values='ResistanceRate',
                                                index='Antibiotic',
                                                columns='Year',
                                                aggfunc='mean')

        plt.figure(figsize=(12, 6))
        sns.heatmap(antibiotic_pivot, cmap='YlOrRd', annot=True, fmt='.2f', linewidths=0.5)
        plt.title('Resistance Rates by Antibiotic Over Time', fontsize=14)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Antibiotic', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'antibiotic_resistance_heatmap.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("Resistance heatmaps created.")

    def create_choropleth_maps(self):
        """Create world maps of AMR resistance rates"""
        logger.info("Creating choropleth maps...")

        if not GEOPANDAS_AVAILABLE:
            logger.info("GeoPandas not available, using alternative map visualization")
            self._create_alternative_map()
            return

        try:
            # Load world shapefile (using geopandas built-in world data)
            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

            # Prepare data for mapping (latest year average)
            latest_year = self.data['Year'].max()
            map_data = self.data[self.data['Year'] == latest_year]
            map_data = map_data[(map_data['Pathogen'] == 'Escherichia coli') &
                               (map_data['Antibiotic'] == 'Ciprofloxacin')]

            # Average by country
            country_avg = map_data.groupby('Country')['ResistanceRate'].mean().reset_index()

            # Country code mapping (simplified - would need proper ISO mapping)
            country_codes = {
                'USA': 'United States of America',
                'GBR': 'United Kingdom',
                'FRA': 'France',
                'DEU': 'Germany',
                'IND': 'India',
                'CHN': 'China',
                'BRA': 'Brazil',
                'JPN': 'Japan'
            }

            country_avg['name'] = country_avg['Country'].map(country_codes)

            # Merge with world data
            world_data = world.merge(country_avg, left_on='name', right_on='name', how='left')

            # Create choropleth map
            fig, ax = plt.subplots(1, 1, figsize=(15, 8))
            world_data.boundary.plot(ax=ax, linewidth=0.5, color='black')
            world_data.plot(column='ResistanceRate', ax=ax,
                          legend=True, cmap='RdYlGn_r',
                          missing_kwds={'color': 'lightgrey'})
            plt.title(f'E. coli Resistance to Ciprofloxacin ({latest_year})', fontsize=16)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'amr_choropleth_map.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.warning(f"Error creating choropleth map: {e}")
            # Alternative: simple scatter plot with country positions
            self._create_alternative_map()

    def _create_alternative_map(self):
        """Create alternative world map visualization"""
        # Simple approach using plotly
        latest_year = self.data['Year'].max()
        map_data = self.data[self.data['Year'] == latest_year]
        map_data = map_data[(map_data['Pathogen'] == 'Escherichia coli') &
                           (map_data['Antibiotic'] == 'Ciprofloxacin')]

        # Simplified coordinates for countries
        coords = {
            'USA': [39.50, -98.35],
            'GBR': [55.3781, -3.4360],
            'FRA': [46.2276, 2.2137],
            'DEU': [51.1657, 10.4515],
            'IND': [20.5937, 78.9629],
            'CHN': [35.8617, 104.1954],
            'BRA': [-14.2350, -51.9253],
            'JPN': [36.2048, 138.2529]
        }

        map_data['lat'] = map_data['Country'].map(lambda x: coords.get(x, [0, 0])[0])
        map_data['lon'] = map_data['Country'].map(lambda x: coords.get(x, [0, 0])[1])

        fig = px.scatter_geo(map_data, lat='lat', lon='lon', size='ResistanceRate',
                           hover_name='Country', color='ResistanceRate',
                           color_continuous_scale='RdYlGn_r',
                           title=f'E. coli Resistance to Ciprofloxacin ({latest_year})')

        fig.update_layout(
            geo=dict(showframe=False, showcoastlines=True),
            template='plotly_white'
        )
        fig.write_html(os.path.join(self.results_dir, 'amr_world_map_alternative.html'))

    def create_forecast_plots(self):
        """Create forecast plots with confidence intervals"""
        logger.info("Creating forecast visualization plots...")

        try:
            # Load forecast data from previous analysis
            arima_forecast = pd.read_csv(os.path.join(self.results_dir, 'arima_forecast.csv'))
            prophet_forecast = pd.read_csv(os.path.join(self.results_dir, 'prophet_forecast.csv'))

            # Historical data for context
            hist_data = self.data[(self.data['Pathogen'] == 'Escherichia coli') &
                                (self.data['Antibiotic'] == 'Ciprofloxacin')]
            hist_avg = hist_data.groupby('Year')['ResistanceRate'].mean().reset_index()

            # Combined forecast plot
            plt.figure(figsize=(14, 8))

            # Historical
            plt.plot(hist_avg['Year'], hist_avg['ResistanceRate'],
                    'bo-', linewidth=2, label='Historical Data')

            # ARIMA forecast
            plt.plot(arima_forecast['Year'], arima_forecast['Predicted_ResistanceRate'],
                    'r--', linewidth=2, label='ARIMA Forecast')

            # Prophet forecast with CI
            plt.plot(prophet_forecast['Year'], prophet_forecast['Predicted_ResistanceRate'],
                    'g--', linewidth=2, label='Prophet Forecast')
            plt.fill_between(prophet_forecast['Year'],
                           prophet_forecast['Lower_CI'],
                           prophet_forecast['Upper_CI'],
                           color='green', alpha=0.2, label='Prophet 95% CI')

            plt.title('10-Year AMR Forecast: E. coli Resistance to Ciprofloxacin', fontsize=16)
            plt.xlabel('Year', fontsize=12)
            plt.ylabel('Resistance Rate', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'amr_forecast_combined.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()

            # Interactive Plotly forecast
            fig = go.Figure()

            # Historical
            fig.add_trace(go.Scatter(x=hist_avg['Year'], y=hist_avg['ResistanceRate'],
                                   mode='lines+markers', name='Historical',
                                   line=dict(color='blue', width=3)))

            # ARIMA
            fig.add_trace(go.Scatter(x=arima_forecast['Year'],
                                   y=arima_forecast['Predicted_ResistanceRate'],
                                   mode='lines', name='ARIMA Forecast',
                                   line=dict(color='red', dash='dash', width=2)))

            # Prophet with CI
            fig.add_trace(go.Scatter(x=prophet_forecast['Year'],
                                   y=prophet_forecast['Predicted_ResistanceRate'],
                                   mode='lines', name='Prophet Forecast',
                                   line=dict(color='green', dash='dash', width=2)))
            fig.add_trace(go.Scatter(x=prophet_forecast['Year'],
                                   y=prophet_forecast['Lower_CI'],
                                   fill=None, mode='lines', line_color='rgba(0,100,0,0.2)',
                                   showlegend=False))
            fig.add_trace(go.Scatter(x=prophet_forecast['Year'],
                                   y=prophet_forecast['Upper_CI'],
                                   fill='tonexty', mode='lines', line_color='rgba(0,100,0,0.2)',
                                   name='Prophet 95% CI'))

            fig.update_layout(
                title='10-Year AMR Forecast with Confidence Intervals',
                xaxis_title='Year',
                yaxis_title='Resistance Rate',
                template='plotly_white',
                height=600
            )
            fig.write_html(os.path.join(self.results_dir, 'amr_forecast_interactive.html'))

        except FileNotFoundError:
            logger.warning("Forecast data files not found. Skipping forecast plots.")
        except Exception as e:
            logger.error(f"Error creating forecast plots: {e}")

    def create_consumption_vs_resistance_plot(self):
        """Create scatter plots of antibiotic consumption vs resistance"""
        logger.info("Creating consumption vs resistance plots...")

        plot_data = self.data[(self.data['Pathogen'] == 'Escherichia coli') &
                             (self.data['Antibiotic'] == 'Ciprofloxacin')].copy()

        if len(plot_data) > 0:
            plt.figure(figsize=(10, 6))
            plt.scatter(plot_data['ConsumptionRate'], plot_data['ResistanceRate'],
                       alpha=0.6, s=50)
            plt.xlabel('Antibiotic Consumption Rate (DDD per 1000 inhabitants/day)', fontsize=12)
            plt.ylabel('Resistance Rate', fontsize=12)
            plt.title('Antibiotic Consumption vs AMR Resistance', fontsize=14)
            plt.grid(True, alpha=0.3)

            # Add trend line
            z = np.polyfit(plot_data['ConsumptionRate'], plot_data['ResistanceRate'], 1)
            p = np.poly1d(z)
            plt.plot(plot_data['ConsumptionRate'], p(plot_data['ConsumptionRate']),
                    "r--", alpha=0.8)

            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'consumption_vs_resistance.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()

            logger.info("Consumption vs resistance plot created.")

    def generate_visualization_report(self):
        """Generate visualization report summary"""
        try:
            with open(os.path.join(self.results_dir, 'visualization_report.txt'), 'w') as f:
                f.write("AMR Data Visualization Report\n")
                f.write("=" * 50 + "\n\n")
                f.write("Generated visualizations:\n")
                f.write("1. Time-series resistance trends\n")
                f.write("2. Country-specific resistance trends\n")
                f.write("3. Resistance heatmaps\n")
                f.write("4. World choropleth maps\n")
                f.write("5. Forecast plots with confidence intervals\n")
                f.write("6. Consumption vs resistance scatter plots\n\n")
                f.write("Files saved in /results/ directory\n")

        except Exception as e:
            logger.error(f"Error generating visualization report: {e}")

if __name__ == "__main__":
    viz = AMRVisualization()
    if viz.load_data():
        viz.create_time_series_plots()
        viz.create_heatmaps()
        viz.create_choropleth_maps()
        viz.create_forecast_plots()
        viz.create_consumption_vs_resistance_plot()
        viz.generate_visualization_report()
        logger.info("Visualization completed successfully.")
    else:
        logger.error("Failed to load data for visualization.")
