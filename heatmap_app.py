import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import plotly.graph_objects as go
from PIL import Image
import plotly.colors as pc
import math

st.set_page_config(layout="wide")
hotel_id = st.selectbox("Select Hotel ID", options=[3, 4, 6, 529, 4559], index=0)

# Add this near the top of your script, after the imports
PICKUP_COLOR = 'rgb(31, 119, 180)'  # Blue
REVENUE_COLOR = 'rgb(44, 160, 44)'  # Green
RATE_COLOR = 'rgb(214, 39, 40)'     # Red

# Read in Data
@st.cache_data
def load_data(hotel_id):
    if hotel_id == 3: 
        pickup_data = pd.read_csv('3_pickup.csv')
        full_refundable_rates_data = pd.read_csv('full_refundables_rate_3.csv')
        bookings_forecast_data = pd.read_csv('forecasted_rev_3.csv')
    if hotel_id == 4:
        pickup_data = pd.read_csv('4_pickup.csv')
        full_refundable_rates_data = pd.read_csv('full_refundables_rate_data_4.csv')
        bookings_forecast_data = pd.read_csv('forecast_revenue_4.csv')
    if hotel_id == 6:
        pickup_data = pd.read_csv('6_pickup.csv')
        full_refundable_rates_data = pd.read_csv('full_refundables_rate_data.csv')
        bookings_forecast_data = pd.read_csv('bookings_forecast.csv')
    if hotel_id == 529:
        pickup_data = pd.read_csv('529_pickup.csv')
        full_refundable_rates_data = pd.read_csv('full_refundables_rate_529.csv')
        bookings_forecast_data = pd.read_csv('forecast_revenue_529.csv')
    if hotel_id == 4559:
        pickup_data = pd.read_csv('4559_pickup.csv')
        full_refundable_rates_data = pd.read_csv('full_refundable_rates_4559.csv')
        bookings_forecast_data = pd.read_csv('forecast_data_4559.csv')


    return pickup_data, full_refundable_rates_data, bookings_forecast_data

pickup_data, full_refundable_rates_data, bookings_forecast_data = load_data(hotel_id)

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
    
    # Check for any infinities or very large values
    if np.isinf(pivot_data).any().any() or (np.abs(pivot_data) > 1e10).any().any():
        print("Warning: Infinite or very large values detected in the data.")
        pivot_data = pivot_data.replace([np.inf, -np.inf], np.nan)
    
    return pivot_data

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

def create_dual_line_plot(pickup_data, bookings_forecast_data, full_refundable_rates_data, stay_date_1, stay_date_2, selected_tab):
    fig1 = create_line_plot(pickup_data, bookings_forecast_data, full_refundable_rates_data, stay_date_1, selected_tab)
    
    fig2 = None
    if stay_date_2:
        fig2 = create_line_plot(pickup_data, bookings_forecast_data, full_refundable_rates_data, stay_date_2, selected_tab)
    
    return fig1, fig2

def plot_heatmap_plotly(data, title, value_column, start_date, end_date, colorbar_min=None, colorbar_max=None, selected_stay_dates=None):
    # Set default values if not provided
    if colorbar_min is None:
        colorbar_min = data.values.min()
    if colorbar_max is None:
        colorbar_max = data.values.max()

    # Create a custom colorscale
    colors = pc.sequential.Rainbow
    colorscale = pc.make_colorscale(['rgb(255,255,255)'] + colors)

    # Create an empty Figure object
    fig = go.Figure()

    # Add the heatmap trace to the figure
    fig.add_trace(go.Heatmap(
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
        )
    ))

    # Add horizontal lines for selected stay dates
    if selected_stay_dates:
        if not isinstance(selected_stay_dates, list):
            selected_stay_dates = [selected_stay_dates]
        
        for i, stay_date in enumerate(selected_stay_dates):
            if stay_date:
                color = "red" if i == 0 else "blue"
                fig.add_shape(
                    type="line",
                    x0=data.columns.min(),
                    x1=data.columns.max(),
                    y0=stay_date,
                    y1=stay_date,
                    line=dict(color=color, width=2, dash="dash"),
                )

    # Update layout
    fig.update_layout(
        title=f'{title}<br>for Dates from {start_date.strftime("%d/%m/%Y")} to {end_date.strftime("%d/%m/%Y")}',
        xaxis_title='Report Date',
        yaxis_title='Stay Date',
        height=700,
        margin=dict(b=150)
    )

    fig.update_yaxes(autorange="reversed")
    
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
    stay_date_1 = st.date_input("Select Stay Date 1 for Line Plot", value=start_date, format="DD/MM/YYYY")
    stay_date_2 = st.date_input("Select Stay Date 2 for Line Plot", value=None, format="DD/MM/YYYY")


    
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
def precompute_graphs(pickup_data, full_refundable_rates_data, bookings_forecast_data, start_date, end_date, prev_start_date, prev_end_date, prev2_start_date, prev2_end_date, custom_range, pickup_cmin, pickup_cmax, revenue_cmin, revenue_cmax, rate_cmin, rate_cmax, stay_date_1, stay_date_2):
    pickup_norm = create_normalized_heatmap(pickup_data, start_date, end_date, 'total_rooms')
    pickup_norm_prev = create_normalized_heatmap(pickup_data, prev_start_date, prev_end_date, 'total_rooms')
    pickup_norm_prev2 = create_normalized_heatmap(pickup_data, prev2_start_date, prev2_end_date, 'total_rooms')
    full_refundable_norm = create_normalized_heatmap(full_refundable_rates_data, start_date, end_date, 'refundable_rate')
    full_refundable_norm_prev = create_normalized_heatmap(full_refundable_rates_data, prev_start_date, prev_end_date, 'refundable_rate')
    full_refundable_norm_prev2 = create_normalized_heatmap(full_refundable_rates_data, prev2_start_date, prev2_end_date, 'refundable_rate')
    bookings_norm = create_normalized_heatmap(bookings_forecast_data, start_date, end_date, 'revenue')
    bookings_norm_prev = create_normalized_heatmap(bookings_forecast_data, prev_start_date, prev_end_date, 'revenue')
    bookings_norm_prev2 = create_normalized_heatmap(bookings_forecast_data, prev2_start_date, prev2_end_date, 'revenue')

    # Pass both stay dates as a list for current year plots
    pickup_fig = plot_heatmap_plotly(pickup_norm, 'Heatmap of Total Rooms', 'Total Rooms', start_date, end_date, pickup_cmin, pickup_cmax, [stay_date_1, stay_date_2])
    pickup_fig_prev = plot_heatmap_plotly(pickup_norm_prev, 'Heatmap of Total Rooms (Previous Year)', 'Total Rooms', prev_start_date, prev_end_date, pickup_cmin, pickup_cmax)
    pickup_fig_prev2 = plot_heatmap_plotly(pickup_norm_prev2, 'Heatmap of Total Rooms (2 Years Previous)', 'Total Rooms', prev2_start_date, prev2_end_date, pickup_cmin, pickup_cmax)
    
    bookings_fig = plot_heatmap_plotly(bookings_norm, 'Heatmap of Forecasted Revenue', 'Revenue', start_date, end_date, revenue_cmin, revenue_cmax, [stay_date_1, stay_date_2])
    bookings_fig_prev = plot_heatmap_plotly(bookings_norm_prev, 'Heatmap of Forecasted Revenue (Previous Year)', 'Revenue', prev_start_date, prev_end_date, revenue_cmin, revenue_cmax)
    bookings_fig_prev2 = plot_heatmap_plotly(bookings_norm_prev2, 'Heatmap of Forecasted Revenue (2 Years Previous)', 'Revenue', prev2_start_date, prev2_end_date, revenue_cmin, revenue_cmax)
    
    full_refundable_fig = plot_heatmap_plotly(full_refundable_norm, 'Heatmap of Refundable Rates', 'Refundable Rate', start_date, end_date, rate_cmin, rate_cmax, [stay_date_1, stay_date_2])
    full_refundable_fig_prev = plot_heatmap_plotly(full_refundable_norm_prev, 'Heatmap of Refundable Rates (Previous Year)', 'Refundable Rate', prev_start_date, prev_end_date, rate_cmin, rate_cmax)
    full_refundable_fig_prev2 = plot_heatmap_plotly(full_refundable_norm_prev2, 'Heatmap of Refundable Rates (2 Years Previous)', 'Refundable Rate', prev2_start_date, prev2_end_date, rate_cmin, rate_cmax)

    return (pickup_fig, pickup_fig_prev, pickup_fig_prev2, bookings_fig, bookings_fig_prev, bookings_fig_prev2, full_refundable_fig, full_refundable_fig_prev, full_refundable_fig_prev2)

# Unpack the returned values from precompute_graphs
(pickup_fig, pickup_fig_prev, pickup_fig_prev2, bookings_fig, bookings_fig_prev, bookings_fig_prev2, full_refundable_fig, full_refundable_fig_prev, full_refundable_fig_prev2) = precompute_graphs(
    pickup_data, full_refundable_rates_data, bookings_forecast_data, start_date, end_date, prev_start_date, prev_end_date, prev2_start_date, prev2_end_date,
    custom_range, pickup_cmin, pickup_cmax, revenue_cmin, revenue_cmax, rate_cmin, rate_cmax, stay_date_1, stay_date_2
)

# Use the second (wider) column for the tabs and plots
with col2:
    # Create tabs for each heatmap
    tab1, tab2, tab3 = st.tabs(["Pickup Data", "Forecasted Revenue Data", "Full Refundable Rates Data"])

    # Plot heatmaps in tabs
    with tab1:
        year_selection = st.radio("Select Year", ["Current Year", "LY1 (Previous Year)", "LY2 (2 Years Previous)"], key="pickup_year_selection", horizontal=True)
        
        if year_selection == "Current Year":
            st.plotly_chart(pickup_fig, use_container_width=True)
        elif year_selection == "LY1 (Previous Year)":
            st.plotly_chart(pickup_fig_prev, use_container_width=True)
        else:
            st.plotly_chart(pickup_fig_prev2, use_container_width=True)

        # Add line plot below the heatmap
        fig1, fig2 = create_dual_line_plot(pickup_data, bookings_forecast_data, full_refundable_rates_data, stay_date_1, stay_date_2, "Pickup Data")
        if fig1:
            st.plotly_chart(fig1, use_container_width=True)
        if fig2:
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        year_selection = st.radio("Select Year", ["Current Year", "LY1 (Previous Year)", "LY2 (2 Years Previous)"], key="bookings_year_selection", horizontal=True)
        
        if year_selection == "Current Year":
            st.plotly_chart(bookings_fig, use_container_width=True)
        elif year_selection == "LY1 (Previous Year)":
            st.plotly_chart(bookings_fig_prev, use_container_width=True)
        else:
            st.plotly_chart(bookings_fig_prev2, use_container_width=True)

        # Add line plot below the heatmap
        fig1, fig2 = create_dual_line_plot(pickup_data, bookings_forecast_data, full_refundable_rates_data, stay_date_1, stay_date_2, "Forecasted Revenue Data")
        if fig1:
            st.plotly_chart(fig1, use_container_width=True)
        if fig2:
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        year_selection = st.radio("Select Year", ["Current Year", "LY1 (Previous Year)", "LY2 (2 Years Previous)"], key="refundable_year_selection", horizontal=True)
        
        if year_selection == "Current Year":
            st.plotly_chart(full_refundable_fig, use_container_width=True)
        elif year_selection == "LY1 (Previous Year)":
            st.plotly_chart(full_refundable_fig_prev, use_container_width=True)
        else:
            st.plotly_chart(full_refundable_fig_prev2, use_container_width=True)

        # Add line plot below the heatmap
        fig1, fig2 = create_dual_line_plot(pickup_data, bookings_forecast_data, full_refundable_rates_data, stay_date_1, stay_date_2, "Full Refundable Rates Data")
        if fig1:
            st.plotly_chart(fig1, use_container_width=True)
        if fig2:
            st.plotly_chart(fig2, use_container_width=True)

    st.caption("Use the mouse to zoom and pan on the heatmaps and line plots. Use the range selector and slider below the line plots to adjust the date range.")

