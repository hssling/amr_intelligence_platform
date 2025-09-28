import os
import logging
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Try to import statsmodels for trendlines
try:
    import statsmodels.api as sm
    STATS_MODELS_AVAILABLE = True
except ImportError:
    STATS_MODELS_AVAILABLE = False
    print("Warning: statsmodels not available. Trendlines will be disabled.")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AMRDashboard:
    def __init__(self, processed_data_path='data/processed/amr_panel_data.csv',
                 results_dir='results'):
        self.processed_data_path = processed_data_path
        self.results_dir = results_dir
        self.data = None
        self.forecast_data = None

    def load_data(self):
        """Load processed AMR data"""
        try:
            self.data = pd.read_csv(self.processed_data_path)
            logger.info(f"Loaded dashboard data with shape: {self.data.shape}")
        except FileNotFoundError:
            logger.error(f"Data file not found: {self.processed_data_path}")
            st.error(f"Data file not found: {self.processed_data_path}")
            return False
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            st.error(f"Error loading data: {e}")
            return False
        return True

    def load_forecast_data(self):
        """Load forecast data for predictions"""
        try:
            arima_forecast = pd.read_csv(os.path.join(self.results_dir, 'arima_forecast.csv'))
            prophet_forecast = pd.read_csv(os.path.join(self.results_dir, 'prophet_forecast.csv'))
            self.forecast_data = {
                'arima': arima_forecast,
                'prophet': prophet_forecast
            }
        except FileNotFoundError:
            logger.warning("Forecast data files not found.")
            self.forecast_data = None
        except Exception as e:
            logger.error(f"Error loading forecast data: {e}")
            self.forecast_data = None

    def run_dashboard(self):
        """Run the Streamlit dashboard"""
        st.set_page_config(
            page_title="AMR Intelligence Dashboard",
            page_icon="🦠",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        st.title("🦠 Antimicrobial Resistance Intelligence Platform")
        st.markdown("""
        **Research Question:** How does antibiotic consumption across countries correlate with
        antimicrobial resistance trends, and can we predict resistance prevalence for the next 10 years?

        **Data Source:** World Bank health indicators + WHO GLASS AMR patterns
        """)

        # Load data
        if not self.load_data():
            return

        self.load_forecast_data()

        # Sidebar filters
        st.sidebar.header("📊 Data Filters")

        # Country filter
        countries = self.data['Country'].unique()
        selected_countries = st.sidebar.multiselect(
            "Select Countries:",
            countries,
            default=countries[:6] if len(countries) > 6 else countries
        )

        # Year range filter
        years = sorted(self.data['Year'].unique())
        year_range = st.sidebar.slider(
            "Select Year Range:",
            min_value=int(min(years)),
            max_value=int(max(years)),
            value=(int(min(years)), int(max(years)))
        )

        # Filter data based on selections
        filtered_data = self.data[
            (self.data['Country'].isin(selected_countries)) &
            (self.data['Year'].between(year_range[0], year_range[1]))
        ]

        if filtered_data.empty:
            st.warning("No data available for the selected filters. Please adjust your selection.")
            return

        # Create dashboard tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Resistance Trends",
            "🌍 Global Overview",
            "🔮 Forecasting",
            "📊 Correlations"
        ])

        with tab1:
            self.create_trends_tab(filtered_data)

        with tab2:
            self.create_global_overview_tab(filtered_data)

        with tab3:
            self.create_forecasting_tab()

        with tab4:
            self.create_correlations_tab(filtered_data)

        # Footer
        st.markdown("---")
        st.markdown(f"""
        **Dashboard updated:** Data reflects information up to {max(years)}

        *For research purposes only. Contact [research team] for raw data access.*
        """)

    def create_trends_tab(self, data):
        """Create resistance trends visualization tab"""
        st.header("Resistance Trends: Antibiotic Consumption vs AMR")

        col1, col2 = st.columns(2)

        with col1:
            # Line chart for each country
            fig = px.line(
                data,
                x='Year',
                y='ResistanceRate',
                color='Country',
                markers=True,
                title='Resistance Rates by Country Over Time'
            )
            fig.update_layout(
                xaxis_title='Year',
                yaxis_title='Resistance Rate',
                legend_title='Country'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Box plot showing distribution by year
            fig = px.box(
                data,
                x='Year',
                y='ResistanceRate',
                title='Resistance Rate Distribution by Year'
            )
            fig.update_layout(
                xaxis_title='Year',
                yaxis_title='Resistance Rate'
            )
            st.plotly_chart(fig, use_container_width=True)

        # Statistics table
        st.subheader("📋 Summary Statistics")

        stats_df = data.groupby('Country').agg({
            'ResistanceRate': ['mean', 'std', 'min', 'max'],
            'ConsumptionRate': 'mean'
        }).round(3)

        stats_df.columns = ['Mean Resistance', 'Std Resistance', 'Min Resistance', 'Max Resistance', 'Mean Consumption']
        stats_df = stats_df.reset_index()

        st.dataframe(stats_df, use_container_width=True)

    def create_global_overview_tab(self, data):
        """Create global overview tab with maps and comparisons"""
        st.header("Global Overview: AMR Rates by Country")

        # Latest year data
        latest_year = data['Year'].max()
        latest_data = data[data['Year'] == latest_year]

        col1, col2 = st.columns([2, 1])

        with col1:
            # Create simple bubble map
            latest_data = latest_data.copy()
            # Coordinates for our AMR dataset countries
            coord_map = {
                'USA': (-98.35, 39.50),    # United States
                'GBR': (-3.4360, 55.3781), # United Kingdom
                'FRA': (2.2137, 46.2276),  # France
                'DEU': (10.4515, 51.1657), # Germany
                'ITA': (12.5674, 41.8719), # Italy
                'ESP': (-3.7492, 40.4637)  # Spain
            }

            latest_data['lon'] = latest_data['Country'].map(lambda x: coord_map.get(x, (0, 0))[0])
            latest_data['lat'] = latest_data['Country'].map(lambda x: coord_map.get(x, (0, 0))[1])

            fig = px.scatter_geo(
                latest_data,
                lat='lat',
                lon='lon',
                size='ResistanceRate',
                color='ConsumptionRate',
                hover_name='Country',
                hover_data=['ResistanceRate', 'ConsumptionRate'],
                title=f'Global AMR Overview ({latest_year})',
                color_continuous_scale='RdYlGn_r'
            )
            fig.update_layout(
                geo=dict(showframe=False, showcoastlines=True)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("🏆 Top Countries by Resistance Rate")

            top_countries = latest_data.nlargest(5, 'ResistanceRate')[
                ['Country', 'ResistanceRate', 'ConsumptionRate']
            ].round(3)

            for idx, row in top_countries.iterrows():
                st.metric(
                    label=row['Country'],
                    value=f"{row['ResistanceRate']:.1%}",
                    delta=f"Consumption: {row['ConsumptionRate']:.1f}"
                )

            st.subheader("📊 Key Metrics")

            avg_resistance = latest_data['ResistanceRate'].mean()
            avg_consumption = latest_data['ConsumptionRate'].mean()

            st.metric("Global Average Resistance Rate", f"{avg_resistance:.1%}")
            st.metric("Global Average Consumption Rate", f"{avg_consumption:.1f} DDD/1000/day")

    def create_forecasting_tab(self):
        """Create forecasting dashboard tab"""
        st.header("🔮 10-Year Forecasting: AMR Trends")

        if self.forecast_data is None:
            st.info("🔍 Forecast data is being generated using ML models...")
            st.warning("Advanced forecasting results will be available after running ML analysis pipeline.")
            st.markdown("**Alternative: Run Intervention Scenarios**")

            # Show simple trend projection based on current data
            st.subheader("📈 Simple Trend Projection (Next 5 Years)")

            # Calculate trend from historical data
            hist_data = self.data.groupby('Year')['ResistanceRate'].mean().reset_index()

            # Simple linear trend
            from sklearn.linear_model import LinearRegression
            import numpy as np

            X = hist_data['Year'].values.reshape(-1, 1)
            y = hist_data['ResistanceRate'].values

            model = LinearRegression()
            model.fit(X, y)

            # Future years
            future_years = np.array(range(2023, 2028)).reshape(-1, 1)
            future_predictions = model.predict(future_years)

            # Create projection chart
            fig = go.Figure()

            # Historical data
            fig.add_trace(go.Scatter(
                x=hist_data['Year'],
                y=hist_data['ResistanceRate'],
                mode='lines+markers',
                name='Historical Data',
                line=dict(color='blue', width=3)
            ))

            # Linear trend projection
            future_years_flat = future_years.flatten()
            fig.add_trace(go.Scatter(
                x=future_years_flat,
                y=future_predictions,
                mode='lines+markers',
                name='Linear Trend Projection',
                line=dict(color='orange', dash='dash', width=3)
            ))

            fig.update_layout(
                title='AMR Resistance Linear Trend Projection (2023-2027)',
                xaxis_title='Year',
                yaxis_title='Resistance Rate',
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Projection summary
            st.subheader("📊 Projected Resistance Rate (2027)")
            projected_2027 = future_predictions[-1]
            st.metric("Linear Projection 2027", f"{projected_2027:.1%}")
            st.caption("*Simple linear extrapolation from 2018-2022 trends*")

            # Policy intervention slider
            st.subheader("🎯 Policy Intervention Scenario Modeling")

            reduction_pct = st.slider("Antibiotic Consumption Reduction (%):",
                                     min_value=0, max_value=50, value=20, step=5)

            reduction_factor = (100 - reduction_pct) / 100
            projected_intervention = projected_2027 * reduction_factor

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Baseline Projection 2027", f"{projected_2027:.1%}")
            with col2:
                st.metric("With Intervention", f"{projected_intervention:.1%}",
                         delta=f"-{reduction_pct}%")

            effectiveness = (projected_2027 - projected_intervention) / projected_2027 * 100
            st.success(f"**Intervention Effectiveness:** {effectiveness:.1f}% resistance reduction")
            st.caption("Policy modeling: Antibiotic consumption reduction → proportional resistance decrease")

            return

        col1, col2 = st.columns(2)

        with col1:
            # ARIMA Forecast
            st.subheader("ARIMA Model Forecast")

            arima_data = self.forecast_data['arima']

            fig = go.Figure()

            # Historical data
            hist_data = self.data.groupby('Year')['ResistanceRate'].mean().reset_index()

            fig.add_trace(go.Scatter(
                x=hist_data['Year'],
                y=hist_data['ResistanceRate'],
                mode='lines+markers',
                name='Historical Data',
                line=dict(color='blue', width=3)
            ))

            # ARIMA forecast
            fig.add_trace(go.Scatter(
                x=arima_data['Year'],
                y=arima_data['Predicted_ResistanceRate'],
                mode='lines+markers',
                name='ARIMA Forecast',
                line=dict(color='red', dash='dash', width=2)
            ))

            fig.update_layout(
                title='ARIMA Resistance Forecast',
                xaxis_title='Year',
                yaxis_title='Resistance Rate',
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Prophet Forecast
            st.subheader("Prophet Model Forecast")

            prophet_data = self.forecast_data['prophet']

            fig = go.Figure()

            # Historical data
            fig.add_trace(go.Scatter(
                x=hist_data['Year'],
                y=hist_data['ResistanceRate'],
                mode='lines+markers',
                name='Historical Data',
                line=dict(color='blue', width=3)
            ))

            # Prophet forecast with confidence interval
            fig.add_trace(go.Scatter(
                x=prophet_data['Year'],
                y=prophet_data['Predicted_ResistanceRate'],
                mode='lines+markers',
                name='Prophet Forecast',
                line=dict(color='green', dash='dash', width=2)
            ))

            # Confidence interval
            fig.add_trace(go.Scatter(
                x=prophet_data['Year'],
                y=prophet_data['Lower_CI'],
                fill=None,
                mode='lines',
                line_color='rgba(0,100,0,0.2)',
                showlegend=False
            ))

            fig.add_trace(go.Scatter(
                x=prophet_data['Year'],
                y=prophet_data['Upper_CI'],
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,100,0,0.2)',
                name='95% Confidence Interval'
            ))

            fig.update_layout(
                title='Prophet Resistance Forecast with Confidence Interval',
                xaxis_title='Year',
                yaxis_title='Resistance Rate',
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)

        # Forecast summary table
        st.subheader("📈 Forecast Summary (2035)")

        final_year = 2035
        arima_2035 = self.forecast_data['arima'][
            self.forecast_data['arima']['Year'] == final_year
        ]['Predicted_ResistanceRate'].iloc[0] if final_year in self.forecast_data['arima']['Year'].values else None

        prophet_2035 = self.forecast_data['prophet'][
            self.forecast_data['prophet']['Year'] == final_year
        ]['Predicted_ResistanceRate'].iloc[0] if final_year in self.forecast_data['prophet']['Year'].values else None

        prophet_ci_lower = self.forecast_data['prophet'][
            self.forecast_data['prophet']['Year'] == final_year
        ]['Lower_CI'].iloc[0] if final_year in self.forecast_data['prophet']['Year'].values else None

        prophet_ci_upper = self.forecast_data['prophet'][
            self.forecast_data['prophet']['Year'] == final_year
        ]['Upper_CI'].iloc[0] if final_year in self.forecast_data['prophet']['Year'].values else None

        forecast_summary = pd.DataFrame({
            'Model': ['ARIMA', 'Prophet'],
            'Predicted Resistance Rate 2035': [f"{arima_2035:.1%}" if arima_2035 else "N/A",
                                              f"{prophet_2035:.1%}" if prophet_2035 else "N/A"],
            'Confidence Interval': ['N/A',
                                   f"{prophet_ci_lower:.1%} - {prophet_ci_upper:.1%}" if prophet_ci_lower and prophet_ci_upper else "N/A"]
        })

        st.table(forecast_summary)

    def create_correlations_tab(self, data):
        """Create correlations and scatter plots tab"""
        st.header("🔍 Correlations: AMR vs Antibiotic Consumption")

        col1, col2 = st.columns(2)

        with col1:
            # Consumption vs Resistance scatter plot
            fig = px.scatter(
                data,
                x='ConsumptionRate',
                y='ResistanceRate',
                color='Country',
                trendline="ols" if STATS_MODELS_AVAILABLE else None,
                title='Antibiotic Consumption vs Resistance Rate'
            )
            fig.update_layout(
                xaxis_title='Consumption Rate (DDD per 1000 inhabitants/day)',
                yaxis_title='Resistance Rate'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Show warning if statsmodels not available
            if not STATS_MODELS_AVAILABLE:
                st.warning("⚠️ Trendline not available - install statsmodels for OLS regression lines")

            # Add correlation coefficient
            corr = data['ConsumptionRate'].corr(data['ResistanceRate'])
            st.metric("Consumption-Resistance Correlation", f"{corr:.3f}")

        with col2:
            # Year vs Resistance scatter plot (since we don't have GDP)
            fig = px.scatter(
                data,
                x='Year',
                y='ResistanceRate',
                color='Country',
                trendline="ols" if STATS_MODELS_AVAILABLE else None,
                title='Time Trends vs Resistance Rate'
            )
            fig.update_layout(
                xaxis_title='Year',
                yaxis_title='Resistance Rate'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Time trend correlation
            corr_year = data['Year'].corr(data['ResistanceRate'])
            st.metric("Time-Trend Correlation", f"{corr_year:.3f}")

        # Correlation heatmap
        st.subheader("📊 Correlation Matrix")

        # Use available numeric columns
        numeric_cols = ['ResistanceRate', 'ConsumptionRate', 'Sanitation', 'Year']
        available_cols = [col for col in numeric_cols if col in data.columns]
        corr_matrix = data[available_cols].corr().round(3)

        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Correlation Matrix of Key Variables",
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    dashboard = AMRDashboard()
    dashboard.run_dashboard()
