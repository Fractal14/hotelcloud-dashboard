import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
import plotly.graph_objects as go
from PIL import Image
import plotly.colors as pc
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

st.set_page_config(layout="wide")

# Streamlit app
col1, col2 = st.columns([1, 3])

def get_hotel_id(hotel_name):
    hotel_mapping = {
        'Mercure Hyde Park': 6,
        'Mercure Paddington': 3,
        'Hotel Indigo Paddington': 4,
        'Stonehouse Hotel': 529,
        'Stonehouse Court Hotel': 4559
    }
    return hotel_mapping.get(hotel_name)


with col1:
    logo = Image.open('cloudbeds_logo.png')
    st.image(logo, width=300)
    hotel_name = st.selectbox("Select Hotel ID", options=['Mercure Hyde Park', 'Mercure Paddington', 'Hotel Indigo Paddington', 'Stonehouse Hotel', 'Stonehouse Court Hotel'], index=0)
    hotel_id = get_hotel_id(hotel_name)

PICKUP_COLOR = 'rgb(31, 119, 180)'  # Blue
REVENUE_COLOR = 'rgb(44, 160, 44)'  # Green
RATE_COLOR = 'rgb(214, 39, 40)'     # Red

def load_data(hotel_id):
    if hotel_id == 3: 
        pickup_data = pd.read_csv('3_pickup.csv')
        full_refundable_rates_data = pd.read_csv('full_refundables_rate_3.csv')
        bookings_forecast_data = pd.read_csv('forecasted_rev_3.csv')
        compset_predictions_data = None
    if hotel_id == 4:
        pickup_data = pd.read_csv('4_pickup.csv')
        full_refundable_rates_data = pd.read_csv('full_refundables_rate_data_4.csv')
        bookings_forecast_data = pd.read_csv('forecast_revenue_4.csv')
        compset_predictions_data = None
    if hotel_id == 6:
        pickup_data = pd.read_csv('6_pickup.csv')
        full_refundable_rates_data = pd.read_csv('full_refundables_rate_data.csv')
        bookings_forecast_data = pd.read_csv('bookings_forecast.csv')
        compset_predictions_data = pd.read_parquet('6_rolling_predictions_t_competitor_median_rate.parquet')
    if hotel_id == 529:
        pickup_data = pd.read_csv('529_pickup.csv')
        full_refundable_rates_data = pd.read_csv('full_refundables_rate_529.csv')
        bookings_forecast_data = pd.read_csv('forecast_revenue_529.csv')
        compset_predictions_data = None
    if hotel_id == 4559:
        pickup_data = pd.read_csv('4559_pickup.csv')
        full_refundable_rates_data = pd.read_csv('full_refundable_rates_4559.csv')
        bookings_forecast_data = pd.read_csv('forecast_data_4559.csv')
        compset_predictions_data = None

    return pickup_data, full_refundable_rates_data, bookings_forecast_data, compset_predictions_data

pickup_data, full_refundable_rates_data, bookings_forecast_data, compset_predictions_data = load_data(hotel_id)

# Helper functions
def convert_to_previous_year(start_date, end_date, years_back=1):
    def get_previous_year_same_weekday(date, years):
        prev_year_date = date - timedelta(days=364 * years)
        while prev_year_date.weekday() != date.weekday():
            prev_year_date -= timedelta(days=1)
        return prev_year_date

    prev_start = get_previous_year_same_weekday(start_date, years_back)
    prev_end = get_previous_year_same_weekday(end_date, years_back)
    return prev_start, prev_end

@st.cache_data
def create_normalized_heatmap(data, start_date, end_date, value_column='refundable_rate', columns_label = 'report_date'):
    data['report_date'] = pd.to_datetime(data['report_date'])
    data['stay_date'] = pd.to_datetime(data['stay_date'])
    
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    all_dates = pd.date_range(start=start_date, end=end_date)
    
    pivot_data = data.pivot_table(values=value_column, index='stay_date', columns=columns_label, aggfunc='sum')
    pivot_data = pivot_data.reindex(index=all_dates, columns=all_dates, fill_value=np.nan)
    
    original_index = pivot_data.index
    original_columns = pivot_data.columns
    
    def normalize_year(date):
        if date.year == start_date.year:
            return date.replace(year=2000) if date.month != 2 or date.day != 29 else date.replace(year=2000, day=28)
        else:
            return date.replace(year=2001) if date.month != 2 or date.day != 29 else date.replace(year=2001, day=28)
    
    pivot_data.index = pivot_data.index.map(normalize_year)
    pivot_data.columns = pivot_data.columns.map(normalize_year)
    
    if np.isinf(pivot_data).any().any() or (np.abs(pivot_data) > 1e10).any().any():
        pivot_data = pivot_data.replace([np.inf, -np.inf], np.nan)
    
    return pivot_data, original_index, original_columns

def plot_heatmap_plotly(data_current, data_prev, title, value_column, start_date, end_date, colorbar_min=None, colorbar_max=None, selected_stay_date=None):
    data_current, orig_index_current, orig_columns_current = data_current
    if data_prev is not None:
        data_prev, orig_index_prev, orig_columns_prev = data_prev

    if colorbar_min is None:
        if data_prev is not None:
            colorbar_min = min(data_current.values.min(), data_prev.values.min())
        else:
            colorbar_min = data_current.values.min()
    if colorbar_max is None:
        if data_prev is not None:
            colorbar_max = max(data_current.values.max(), data_prev.values.max())
        else:
            colorbar_min = data_current.values.max()

    colors = pc.sequential.Rainbow
    colorscale = pc.make_colorscale(['rgb(255,255,255)'] + colors)

    def create_heatmap(data, orig_index, orig_columns):
        mask = ~np.isnan(data.values)
        customdata = np.array([
            [[col.strftime('%Y-%m-%d'), idx.strftime('%Y-%m-%d')] if mask[i, j] else [None, None]
             for j, col in enumerate(orig_columns)]
            for i, idx in enumerate(orig_index)
        ])
        
        return go.Heatmap(
            z=data.values,
            x=data.columns,
            y=data.index,
            colorscale=colorscale,
            zmin=colorbar_min,
            zmax=colorbar_max,
            colorbar=dict(
                title=value_column,
                orientation='h',
                y=-0.15,
                yanchor='top',
                thickness=20,
                len=0.9
            ),
            hovertemplate=(
                "Report Date: %{customdata[0]}<br>" +
                "Stay Date: %{customdata[1]}<br>" +
                f"{value_column}: %{{z:.2f}}<extra></extra>"
            ),
            customdata=customdata,
            hoverinfo='text'
        )

    heatmap_current = create_heatmap(data_current, orig_index_current, orig_columns_current)
    if data_prev is not None:
        heatmap_prev = create_heatmap(data_prev, orig_index_prev, orig_columns_prev)
        heatmap_prev.visible = False

    heatmap_current.visible = True

    if data_prev is not None:
        fig = go.Figure(data=[heatmap_current, heatmap_prev])
    else:
        fig = go.Figure(data=[heatmap_current])

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                active=0,
                x=0.57,
                y=1.2,
                buttons=list([
                    dict(label=f"{start_date.year}-{end_date.year}",
                         method="update",
                         args=[{"visible": [True, False]},
                               {"title": f'{title}<br>{start_date.year}-{end_date.year}'}]),
                    dict(label=f"{start_date.year-1}-{end_date.year-1}",
                         method="update",
                         args=[{"visible": [False, True]},
                               {"title": f'{title}<br>{start_date.year-1}-{end_date.year-1}'}]),
                ]),
            )
        ]
    )

    fig.update_xaxes(tickformat="%b %d", tickformatstops=[dict(dtickrange=[None, "M1"], value="%b %d"), dict(dtickrange=["M1", None], value="%b")])
    fig.update_yaxes(tickformat="%b %d", tickformatstops=[dict(dtickrange=[None, "M1"], value="%b %d"), dict(dtickrange=["M1", None], value="%b")], autorange="reversed")

    if selected_stay_date:
        selected_stay_date = pd.to_datetime(selected_stay_date)
        normalized_selected_date = selected_stay_date.replace(year=2000 if selected_stay_date.year == start_date.year else 2001)
        fig.add_shape(
            type="line",
            x0=data_current.columns.min(),
            x1=data_current.columns.max(),
            y0=normalized_selected_date,
            y1=normalized_selected_date,
            line=dict(color="red", width=2, dash="dash"),
        )

    fig.update_layout(
        title=f'{title}<br>{start_date.year}-{end_date.year}',
        xaxis_title='Report Date',
        yaxis_title='Stay Date',
        height=700,
        margin=dict(b=150, t=150)
    )

    return fig

@st.cache_data
def create_multi_year_line_plot(pickup_data, bookings_forecast_data, full_refundable_rates_data, stay_date, stay_date_2, stay_date_3, selected_tab):
    def create_year_plot(data, date, color):
        fig = go.Figure()
        filtered_data = data[data['stay_date'] == date.strftime('%Y-%m-%d')]
        filtered_data = filtered_data.sort_values('report_date')

        if selected_tab == "Occupancy":
            y_values = filtered_data['total_rooms']
            y_axis_title = 'Total Rooms'
        elif selected_tab == "Revenue":
            filtered_data['cumulative_revenue'] = filtered_data['revenue'].cumsum()
            y_values = filtered_data['cumulative_revenue']
            y_axis_title = 'Cumulative Revenue'
        else:  # Rates
            y_values = filtered_data['refundable_rate']
            y_axis_title = 'Refundable Rate'
        
        fig.add_trace(go.Scatter(x=filtered_data['report_date'], y=y_values,
                                 mode='lines+markers', name=f'{date.year}', line=dict(color=color)))
        
        fig.update_layout(
            title=f'{selected_tab} for {date.year}',
            xaxis_title='Report Date',
            yaxis_title=y_axis_title,
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig

    data = pickup_data if selected_tab == "Occupancy" else \
           bookings_forecast_data if selected_tab == "Revenue" else \
           full_refundable_rates_data

    # Current Year
    fig_current = create_year_plot(data, stay_date, PICKUP_COLOR)
    
    # LY1
    ly1_date = stay_date_2
    fig_ly1 = create_year_plot(data, ly1_date, PICKUP_COLOR)
    
    # LY2
    ly2_date = stay_date_3
    fig_ly2 = create_year_plot(data, ly2_date, RATE_COLOR)

    return fig_current, fig_ly1, fig_ly2

def calculate_step_size(min_value, max_value):
    range_value = max_value - min_value
    if range_value == 0:
        return 0.1  # Default small step if there's no range
    magnitude = 10 ** math.floor(math.log10(range_value))
    scaled_range = range_value / magnitude
    if scaled_range < 1:
        step_size = magnitude / 10
    elif scaled_range < 5:
        step_size = magnitude / 2
    else:
        step_size = magnitude
    return float(step_size)


# Calculate step sizes for each dataset
pickup_step = calculate_step_size(pickup_data['total_rooms'].min(), pickup_data['total_rooms'].max())
revenue_step = calculate_step_size(bookings_forecast_data['revenue'].min(), bookings_forecast_data['revenue'].max())
rate_step = calculate_step_size(full_refundable_rates_data['refundable_rate'].min(), full_refundable_rates_data['refundable_rate'].max())

# Add the logo to the first (narrower) column
with col1:

    stay_date_1 = st.date_input("Select stay date", value=pd.Timestamp(date(2023, 1, 1)), min_value=date(2023, 1, 1), max_value=date(2024, 12, 31))
    stay_date_2, _ = convert_to_previous_year(stay_date_1, stay_date_1)
    stay_date_3, _ = convert_to_previous_year(stay_date_1, stay_date_1, years_back=2)

    # Display info boxes for all three dates
    st.info(f"Selected Stay Date: {stay_date_1.strftime('%A, %B %d, %Y')}")
    st.info(f"LY1 (Previous Year): {stay_date_2.strftime('%A, %B %d, %Y')}")
    st.info(f"LY2 (2 Years Previous): {stay_date_3.strftime('%A, %B %d, %Y')}")

    # Add checkbox to enable custom colorbar range
    custom_range = st.checkbox("Use custom colorbar range")
    
    # Add sliders for colorbar range, only shown if custom range is enabled
    if custom_range:
        st.write("Occupancy:")
        pickup_min, pickup_max = pickup_data['total_rooms'].min(), pickup_data['total_rooms'].max()
        pickup_cmin = st.slider("Pickup Min", min_value=float(pickup_min), max_value=float(pickup_max), value=float(pickup_min), step=pickup_step)
        pickup_cmax = st.slider("Pickup Max", min_value=float(pickup_min), max_value=float(pickup_max), value=float(pickup_max), step=pickup_step)

        st.write("Revenue:")
        revenue_min, revenue_max = bookings_forecast_data['revenue'].min(), bookings_forecast_data['revenue'].max()
        revenue_cmin = st.slider("Revenue Min", min_value=float(revenue_min), max_value=float(revenue_max), value=float(revenue_min), step=revenue_step)
        revenue_cmax = st.slider("Revenue Max", min_value=float(revenue_min), max_value=float(revenue_max), value=float(revenue_max), step=revenue_step)

        st.write("Rates:")
        rate_min, rate_max = full_refundable_rates_data['refundable_rate'].min(), full_refundable_rates_data['refundable_rate'].max()
        rate_cmin = st.slider("Rate Min", min_value=float(rate_min), max_value=float(rate_max), value=float(rate_min), step=rate_step)
        rate_cmax = st.slider("Rate Max", min_value=float(rate_min), max_value=float(rate_max), value=float(rate_max), step=rate_step)
    else:
        pickup_cmin, pickup_cmax = None, None
        revenue_cmin, revenue_cmax = None, None
        rate_cmin, rate_cmax = None, None

    # Add more vertical space
    st.markdown("<br><br><br>", unsafe_allow_html=True)

# Precompute data
@st.cache_data
def get_previous_year_dates():
    prev_start_date, prev_end_date = convert_to_previous_year(date(2023, 1, 1), date(2024, 12, 31))
    prev2_start_date, prev2_end_date = convert_to_previous_year(date(2023, 1, 1), date(2024, 12, 31), years_back=2)
    return prev_start_date, prev_end_date, prev2_start_date, prev2_end_date

@st.cache_data
def get_normalized_heatmaps(pickup_data, full_refundable_rates_data, bookings_forecast_data):
    prev_start_date, prev_end_date, _, _ = get_previous_year_dates()
    
    pickup_norm = create_normalized_heatmap(pickup_data, date(2023, 1, 1), date(2024, 12, 31), 'total_rooms')
    pickup_norm_prev = create_normalized_heatmap(pickup_data, prev_start_date, prev_end_date, 'total_rooms')
    
    full_refundable_norm = create_normalized_heatmap(full_refundable_rates_data, date(2023, 1, 1), date(2024, 12, 31), 'refundable_rate')
    full_refundable_norm_prev = create_normalized_heatmap(full_refundable_rates_data, prev_start_date, prev_end_date, 'refundable_rate')
    
    forecast_norm = create_normalized_heatmap(bookings_forecast_data, date(2023, 1, 1), date(2024, 12, 31), 'revenue')
    forecast_norm_prev = create_normalized_heatmap(bookings_forecast_data, prev_start_date, prev_end_date, 'revenue')
    
    return pickup_norm, pickup_norm_prev, full_refundable_norm, full_refundable_norm_prev, forecast_norm, forecast_norm_prev

# Use the cached functions
prev_start_date, prev_end_date, prev2_start_date, prev2_end_date = get_previous_year_dates()

pickup_norm, pickup_norm_prev, full_refundable_norm, full_refundable_norm_prev, forecast_norm, forecast_norm_prev = get_normalized_heatmaps(pickup_data, full_refundable_rates_data, bookings_forecast_data)

def plot_simple_heatmap(df, x_column, y_column, z_column, title):
    # Pivot the dataframe
    heatmap_data = df.pivot(index=y_column, columns=x_column, values=z_column)
    
    # Sort the index (y-axis) in ascending order
    heatmap_data = heatmap_data.sort_index(ascending=True)
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='Rainbow',
        colorbar=dict(
            orientation='h',  # Make colorbar horizontal
            titleside='top',  # Place the title on top of the colorbar
            x=0.5,            # Center the colorbar horizontally
            y=-0.2,           # Position the colorbar further below the heatmap
            len=0.9           # Adjust the length of the colorbar
        )
    ))

    # Update the layout
    fig.update_layout(
        xaxis_title='Stay Date',
        yaxis_title='Report Date',
        height=1000,  # Increased height to accommodate more space for the colorbar
        width=900,
        yaxis=dict(autorange='reversed'),  # This ensures dates are in ascending order from top to bottom
        margin=dict(b=200)  # Increase bottom margin significantly to make room for the colorbar
    )
    
    return fig

with col2:
    # Tabs for different datasets
    tab1, tab2, tab3, tab4 = st.tabs(['Occupancy', 'Rates', 'Revenue', 'Compset Median Rates'])

    # Pickup Tab
    with tab1:
        # Pickup heatmap
        fig1 = plot_heatmap_plotly(
            pickup_norm, pickup_norm_prev, 
            'Occupancy', 'total_rooms', 
            date(2023, 1, 1), date(2024, 12, 31), 
            colorbar_min=pickup_cmin, colorbar_max=pickup_cmax, 
            selected_stay_date=stay_date_1
        )
        st.plotly_chart(fig1)

        # Add line plot for Pickup
        fig_current, fig_ly1, fig_ly2 = create_multi_year_line_plot(
            pickup_data, bookings_forecast_data, full_refundable_rates_data, 
            stay_date_1, stay_date_2, stay_date_3, "Occupancy"
        )
        # Tabs for current and last year plots
        tab1a, tab1b = st.tabs(["Current Year", "Last Year 1"])

        with tab1a:
            st.plotly_chart(fig_current, use_container_width=True)
        with tab1b:
            st.plotly_chart(fig_ly1, use_container_width=True)

    # Full Refundable Rates Tab
    with tab2:
        # Fix: Ensure proper data reference for the Full Refundable Rates heatmap
        fig2 = plot_heatmap_plotly(
            full_refundable_norm, full_refundable_norm_prev, 
            'Rate', 'refundable_rate', 
            date(2023, 1, 1), date(2024, 12, 31), 
            colorbar_min=rate_cmin, colorbar_max=rate_cmax, 
            selected_stay_date=stay_date_1
        )
        st.plotly_chart(fig2)

        # Add line plot for Full Refundable Rates
        fig_current, fig_ly1, fig_ly2 = create_multi_year_line_plot(
            pickup_data, bookings_forecast_data, full_refundable_rates_data, 
            stay_date_1, stay_date_2, stay_date_3, "Rates"
        )
        # Tabs for current and last year plots
        tab2a, tab2b = st.tabs(["Current Year", "Last Year 1"])

        with tab2a:
            st.plotly_chart(fig_current, use_container_width=True)
        with tab2b:
            st.plotly_chart(fig_ly1, use_container_width=True)

    # Forecast Tab
    with tab3:
        # Forecast heatmap
        fig3 = plot_heatmap_plotly(
            forecast_norm, forecast_norm_prev,
            'Revenue', 'revenue', 
            date(2023, 1, 1), date(2024, 12, 31), 
            colorbar_min=revenue_cmin, colorbar_max=revenue_cmax, 
            selected_stay_date=stay_date_1
        )
        st.plotly_chart(fig3)

        # Add line plot for Forecast Revenue Data
        fig_current, fig_ly1, fig_ly2 = create_multi_year_line_plot(
            pickup_data, bookings_forecast_data, full_refundable_rates_data, 
            stay_date_1, stay_date_2, stay_date_3, "Revenue"
        )
        # Tabs for current and last year plots
        tab3a, tab3b = st.tabs(["Current Year", "Last Year 1"])

        with tab3a:
            st.plotly_chart(fig_current, use_container_width=True)
        with tab3b:
            st.plotly_chart(fig_ly1, use_container_width=True)

    @st.cache_data
    def load_and_process_compset_data(compset_predictions_data):
        # This function will load and process the data once, then cache the result
        return compset_predictions_data

    @st.cache_data
    def filter_compset_data(compset_predictions_data, prediction_date, ly1_prediction_date):
        # This function will filter the data based on the selected dates
        compset_predictions_current = compset_predictions_data[compset_predictions_data['report_date'] == str(prediction_date)]
        compset_predictions_prev = compset_predictions_data[compset_predictions_data['report_date'] == str(ly1_prediction_date)]
        return compset_predictions_current, compset_predictions_prev

    @st.cache_data
    def create_heatmaps(compset_predictions_current, compset_predictions_prev, value_column):
        prev_start_date, prev_end_date = convert_to_previous_year(date(2024, 1, 1), date(2025, 1, 1))
        compset_norm = create_normalized_heatmap(compset_predictions_current, date(2024, 1, 1), date(2025, 1, 1), value_column, columns_label='future_report_date')
        compset_norm_prev = create_normalized_heatmap(compset_predictions_prev, prev_start_date, prev_end_date, value_column, columns_label='future_report_date')
        return compset_norm, compset_norm_prev

    # Load and cache the full dataset
    compset_predictions_data = load_and_process_compset_data(compset_predictions_data)

    with tab4:
        prediction_date = st.date_input("Select Prediction Date", value=pd.Timestamp(date(2024, 1, 1)), min_value=date(2024, 1, 1), max_value=date(2025, 1, 1))
        ly1_prediction_date, _ = convert_to_previous_year(prediction_date, prediction_date, years_back=1)
        st.info(f"LY1 (Previous Year): {ly1_prediction_date.strftime('%A, %B %d, %Y')}")
        
        data_type = st.radio("Select data type:", ("Actual", "Predicted"))
        
        prediction_date = pd.to_datetime(prediction_date)
        
        # Use the cached function to filter data
        compset_predictions_current, compset_predictions_prev = filter_compset_data(compset_predictions_data, prediction_date, ly1_prediction_date)

        if compset_predictions_current.empty or compset_predictions_prev.empty:
            st.warning("No data available for the selected date(s).")
        else:
            value_column = 'actual' if data_type == "Actual" else 'pred'
            
            # Use the cached function to create heatmaps
            compset_norm, compset_norm_prev = create_heatmaps(compset_predictions_current, compset_predictions_prev, value_column)

            overall_min = min(compset_predictions_current[value_column].min(), compset_predictions_prev[value_column].min())
            overall_max = max(compset_predictions_current[value_column].max(), compset_predictions_prev[value_column].max())
            
            fig4 = plot_heatmap_plotly(
                compset_norm, compset_norm_prev,
                f'CompSet Median {data_type}', value_column,
                date(2024, 1, 1), date(2025, 1, 1),
                selected_stay_date=None, 
                colorbar_min=overall_min, colorbar_max=overall_max
            )
            st.plotly_chart(fig4, use_container_width=True)



            



