import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import plotly.graph_objects as go
from PIL import Image
import plotly.colors as pc
import math
import calendar
st.set_page_config(layout="wide")

# Add this near the top of your script, after the imports
PICKUP_COLOR = 'rgb(31, 119, 180)'  # Blue
REVENUE_COLOR = 'rgb(44, 160, 44)'  # Green
RATE_COLOR = 'rgb(214, 39, 40)'     # Red

# Read in Data
@st.cache_data
def load_data():
    pickup_data = pd.read_csv('6_pickup.csv')
    full_refundable_rates_data = pd.read_csv('full_refundables_rate_data.csv')
    bookings_forecast_data = pd.read_csv('bookings_forecast.csv')
    return pickup_data, full_refundable_rates_data, bookings_forecast_data

pickup_data, full_refundable_rates_data, bookings_forecast_data = load_data()

def convert_to_previous_year(start_date, end_date, years_back=1):
    # Convert input strings to datetime objects if they're not already
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')

    # Function to get the same weekday from previous year(s)
    def get_previous_year_same_weekday(date, years):
        prev_year_date = date - timedelta(days=364 * years)
        while prev_year_date.weekday() != date.weekday():
            prev_year_date -= timedelta(days=1)
        return prev_year_date

    # Convert to previous year(s) maintaining the same weekday
    prev_start = get_previous_year_same_weekday(start_date, years_back)
    prev_end = get_previous_year_same_weekday(end_date, years_back)

    return prev_start, prev_end
@st.cache_data
def create_normalized_heatmap(data, start_date, end_date, value_column='refundable_rate'):
    # Convert 'report_date' and 'stay_date' to datetime
    data['report_date'] = pd.to_datetime(data['report_date'])
    data['stay_date'] = pd.to_datetime(data['stay_date'])
    
    # Ensure start_date and end_date are datetime objects
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Create a date range for both axes
    all_dates = pd.date_range(start=start_date, end=end_date)
    
    # Create a pivot table with the full date range
    pivot_data = data.pivot_table(values=value_column, index='stay_date', columns='report_date', aggfunc='sum')
    pivot_data = pivot_data.reindex(index=all_dates, columns=all_dates, fill_value=np.nan)
    
    # Store original dates
    original_index = pivot_data.index
    original_columns = pivot_data.columns
    
    # Normalize dates to year 2000 and 2001
    def normalize_year(date):
        if date.year == start_date.year:
            try:
                return date.replace(year=2000)
            except ValueError:  # Handle February 29
                return date.replace(year=2000, day=28)
        else:
            try:
                return date.replace(year=2001)
            except ValueError:  # Handle February 29
                return date.replace(year=2001, day=28)
    
    pivot_data.index = pivot_data.index.map(normalize_year)
    pivot_data.columns = pivot_data.columns.map(normalize_year)
    
    # Check for any infinities or very large values
    if np.isinf(pivot_data).any().any() or (np.abs(pivot_data) > 1e10).any().any():
        print("Warning: Infinite or very large values detected in the data.")
        pivot_data = pivot_data.replace([np.inf, -np.inf], np.nan)
    
    return pivot_data, original_index, original_columns

@st.cache_data
def create_line_plot(pickup_data, bookings_forecast_data, full_refundable_rates_data, stay_date, selected_tab):
    # Filter data for the selected stay date
    pickup_filtered = pickup_data[pickup_data['stay_date'] == stay_date.strftime('%Y-%m-%d')]
    bookings_filtered = bookings_forecast_data[bookings_forecast_data['stay_date'] == stay_date.strftime('%Y-%m-%d')]
    rates_filtered = full_refundable_rates_data[full_refundable_rates_data['stay_date'] == stay_date.strftime('%Y-%m-%d')]

    # Create the line plot
    fig = go.Figure()

    if selected_tab == "Pickup Data":
        fig.add_trace(go.Scatter(x=pickup_filtered['report_date'], y=pickup_filtered['total_rooms'],
                                 mode='lines+markers', name='Pickup (Total Rooms)', line=dict(color=PICKUP_COLOR)))
        y_axis_title = 'Total Rooms'
    elif selected_tab == "Forecasted Revenue Data":
        # Sort the data by report_date to ensure the line is drawn correctly
        bookings_filtered = bookings_filtered.sort_values('report_date')
        # Calculate cumulative sum of revenue
        bookings_filtered['cumulative_revenue'] = bookings_filtered['revenue'].cumsum()
        fig.add_trace(go.Scatter(x=bookings_filtered['report_date'], y=bookings_filtered['cumulative_revenue'],
                                 mode='lines+markers', name='Cumulative Revenue Forecast', line=dict(color=REVENUE_COLOR)))
        y_axis_title = 'Cumulative Revenue'
    else:  # Full Refundable Rates Data
        fig.add_trace(go.Scatter(x=rates_filtered['report_date'], y=rates_filtered['refundable_rate'],
                                 mode='lines+markers', name='Refundable Rate', line=dict(color=RATE_COLOR)))
        y_axis_title = 'Refundable Rate'

    # Update layout
    fig.update_layout(
        title=f'{selected_tab} for Stay Date: {stay_date.strftime("%Y-%m-%d")}',
        xaxis_title='Report Date',
        yaxis_title=y_axis_title,
        height=400,
        showlegend=False,
    )

    return fig

def plot_heatmap_plotly(data_current, data_prev, data_prev2, title, value_column, start_date, end_date, colorbar_min=None, colorbar_max=None, selected_stay_date=None):
    # Unpack the data and original dates
    data_current, orig_index_current, orig_columns_current = data_current
    data_prev, orig_index_prev, orig_columns_prev = data_prev
    data_prev2, orig_index_prev2, orig_columns_prev2 = data_prev2

    # Set default values if not provided
    if colorbar_min is None:
        colorbar_min = min(data_current.values.min(), data_prev.values.min(), data_prev2.values.min())
    if colorbar_max is None:
        colorbar_max = max(data_current.values.max(), data_prev.values.max(), data_prev2.values.max())

    # Create a custom colorscale
    colors = pc.sequential.Rainbow
    colorscale = pc.make_colorscale(['rgb(255,255,255)'] + colors)

    # Function to create heatmap with custom hover template
    def create_heatmap(data, orig_index, orig_columns):
        mask = ~np.isnan(data.values)
        customdata = np.array([
        [
            [col.strftime('%Y-%m-%d'), idx.strftime('%Y-%m-%d')] if mask[i, j] else [None, None]
            for j, col in enumerate(orig_columns)
        ]
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

    # Create heatmaps for each year range
    heatmap_current = create_heatmap(data_current, orig_index_current, orig_columns_current)
    heatmap_prev = create_heatmap(data_prev, orig_index_prev, orig_columns_prev)
    heatmap_prev2 = create_heatmap(data_prev2, orig_index_prev2, orig_columns_prev2)

    heatmap_current.visible = True
    heatmap_prev.visible = False
    heatmap_prev2.visible = False

    fig = go.Figure(data=[heatmap_current, heatmap_prev, heatmap_prev2])

    # Add buttons for year selection
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
                         args=[{"visible": [True, False, False]},
                               {"title": f'{title}<br>{start_date.year}-{end_date.year}'}]),
                    dict(label=f"{start_date.year-1}-{end_date.year-1}",
                         method="update",
                         args=[{"visible": [False, True, False]},
                               {"title": f'{title}<br>{start_date.year-1}-{end_date.year-1}'}]),
                    dict(label=f"{start_date.year-2}-{end_date.year-2}",
                         method="update",
                         args=[{"visible": [False, False, True]},
                               {"title": f'{title}<br>{start_date.year-2}-{end_date.year-2}'}]),
                ]),
            )
        ]
    )

    # Update x and y axis settings
    fig.update_xaxes(
        tickformat="%b %d",
        tickformatstops=[
            dict(dtickrange=[None, "M1"], value="%b %d"),
            dict(dtickrange=["M1", None], value="%b")
        ]
    )
    fig.update_yaxes(
        tickformat="%b %d",
        tickformatstops=[
            dict(dtickrange=[None, "M1"], value="%b %d"),
            dict(dtickrange=["M1", None], value="%b")
        ],
        autorange="reversed"
    )

    # Add horizontal line for selected stay date
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

# Streamlit app
col1, col2 = st.columns([1, 3])

# Add the logo to the first (narrower) column
with col1:
    logo = Image.open('hotelcloud_logo.png')
    st.image(logo, width=300)

    # Move date inputs to the first column
    start_date = st.date_input("Start Date", value=date(2023, 1, 1), format="DD/MM/YYYY")
    end_date = st.date_input("End Date", value=date(2024, 7, 10), format="DD/MM/YYYY")

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Add this after the existing date inputs in the first column
    stay_date = st.date_input("Select Stay Date for Line Plot", value=start_date, format="DD/MM/YYYY")
    
    # Add checkbox to enable custom colorbar range
    custom_range = st.checkbox("Use custom colorbar range")
    
    # Add sliders for colorbar range, only shown if custom range is enabled
    if custom_range:
        st.write("Pickup Data Range:")
        pickup_min, pickup_max = pickup_data['total_rooms'].min(), pickup_data['total_rooms'].max()
        pickup_cmin = st.slider("Pickup Min", min_value=float(pickup_min), max_value=float(pickup_max), value=float(pickup_min), step=pickup_step)
        pickup_cmax = st.slider("Pickup Max", min_value=float(pickup_min), max_value=float(pickup_max), value=float(pickup_max), step=pickup_step)

        st.write("Forecasted Revenue Data Range:")
        revenue_min, revenue_max = bookings_forecast_data['revenue'].min(), bookings_forecast_data['revenue'].max()
        revenue_cmin = st.slider("Revenue Min", min_value=float(revenue_min), max_value=float(revenue_max), value=float(revenue_min), step=revenue_step)
        revenue_cmax = st.slider("Revenue Max", min_value=float(revenue_min), max_value=float(revenue_max), value=float(revenue_max), step=revenue_step)

        st.write("Full Refundable Rates Data Range:")
        rate_min, rate_max = full_refundable_rates_data['refundable_rate'].min(), full_refundable_rates_data['refundable_rate'].max()
        rate_cmin = st.slider("Rate Min", min_value=float(rate_min), max_value=float(rate_max), value=float(rate_min), step=rate_step)
        rate_cmax = st.slider("Rate Max", min_value=float(rate_min), max_value=float(rate_max), value=float(rate_max), step=rate_step)
    else:
        pickup_cmin, pickup_cmax = None, None
        revenue_cmin, revenue_cmax = None, None
        rate_cmin, rate_cmax = None, None

    # Add more vertical space
    st.markdown("<br><br><br>", unsafe_allow_html=True)

# Calculate previous year dates
prev_start_date, prev_end_date = convert_to_previous_year(start_date, end_date)
prev2_start_date, prev2_end_date = convert_to_previous_year(start_date, end_date, years_back=2)

# Precompute all normalized data and graphs
@st.cache_data
def precompute_graphs(pickup_data, full_refundable_rates_data, bookings_forecast_data, start_date, end_date, prev_start_date, prev_end_date, prev2_start_date, prev2_end_date, custom_range, pickup_cmin, pickup_cmax, revenue_cmin, revenue_cmax, rate_cmin, rate_cmax, stay_date):
    pickup_norm = create_normalized_heatmap(pickup_data, start_date, end_date, 'total_rooms')
    pickup_norm_prev = create_normalized_heatmap(pickup_data, prev_start_date, prev_end_date, 'total_rooms')
    pickup_norm_prev2 = create_normalized_heatmap(pickup_data, prev2_start_date, prev2_end_date, 'total_rooms')
    full_refundable_norm = create_normalized_heatmap(full_refundable_rates_data, start_date, end_date, 'refundable_rate')
    full_refundable_norm_prev = create_normalized_heatmap(full_refundable_rates_data, prev_start_date, prev_end_date, 'refundable_rate')
    full_refundable_norm_prev2 = create_normalized_heatmap(full_refundable_rates_data, prev2_start_date, prev2_end_date, 'refundable_rate')
    bookings_norm = create_normalized_heatmap(bookings_forecast_data, start_date, end_date, 'revenue')
    bookings_norm_prev = create_normalized_heatmap(bookings_forecast_data, prev_start_date, prev_end_date, 'revenue')
    bookings_norm_prev2 = create_normalized_heatmap(bookings_forecast_data, prev2_start_date, prev2_end_date, 'revenue')

    pickup_fig = plot_heatmap_plotly(pickup_norm, pickup_norm_prev, pickup_norm_prev2, 'Heatmap of Total Rooms', 'Total Rooms', start_date, end_date, pickup_cmin, pickup_cmax, stay_date)
    bookings_fig = plot_heatmap_plotly(bookings_norm, bookings_norm_prev, bookings_norm_prev2, 'Heatmap of Forecasted Revenue', 'Revenue', start_date, end_date, revenue_cmin, revenue_cmax, stay_date)
    full_refundable_fig = plot_heatmap_plotly(full_refundable_norm, full_refundable_norm_prev, full_refundable_norm_prev2, 'Heatmap of Refundable Rates', 'Refundable Rate', start_date, end_date, rate_cmin, rate_cmax, stay_date)

    return (pickup_fig, bookings_fig, full_refundable_fig)

# Unpack the returned values from precompute_graphs
(pickup_fig, bookings_fig, full_refundable_fig) = precompute_graphs(
    pickup_data, full_refundable_rates_data, bookings_forecast_data, start_date, end_date, prev_start_date, prev_end_date, prev2_start_date, prev2_end_date,
    custom_range, pickup_cmin, pickup_cmax, revenue_cmin, revenue_cmax, rate_cmin, rate_cmax, stay_date
)

# Use the second (wider) column for the tabs and plots
with col2:
    # Create tabs for each heatmap
    tab1, tab2, tab3 = st.tabs(["Pickup Data", "Forecasted Revenue Data", "Full Refundable Rates Data"])

    # Plot heatmaps in tabs
    with tab1:
        st.plotly_chart(pickup_fig, use_container_width=True)
        # Add line plot below the heatmap
        st.plotly_chart(create_line_plot(pickup_data, bookings_forecast_data, full_refundable_rates_data, stay_date, "Pickup Data"), use_container_width=True)

    with tab2:
        st.plotly_chart(bookings_fig, use_container_width=True)
        # Add line plot below the heatmap
        st.plotly_chart(create_line_plot(pickup_data, bookings_forecast_data, full_refundable_rates_data, stay_date, "Forecasted Revenue Data"), use_container_width=True)

    with tab3:
        st.plotly_chart(full_refundable_fig, use_container_width=True)
        # Add line plot below the heatmap
        st.plotly_chart(create_line_plot(pickup_data, bookings_forecast_data, full_refundable_rates_data, stay_date, "Full Refundable Rates Data"), use_container_width=True)