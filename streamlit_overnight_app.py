#!/usr/bin/env python3
"""
Streamlit App for Crude Oil Overnight Session High/Low Probability Analysis
Interactive visualization for overnight range (19:30-03:00) with observation until 9:30

Analyzes whether highs and lows formed during the overnight session hold through
the next morning.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import time, datetime, timedelta
import numpy as np
import os

# Page configuration
st.set_page_config(
    page_title="Overnight CL Probability Analysis",
    page_icon="🌙",
    layout="wide"
)

# Title and description
st.title("🌙 Crude Oil Overnight Session High/Low Analysis")
st.markdown("""
Analyze the probability of overnight session highs and lows (19:30-03:00) holding through the morning (until 9:30).
Filter by specific time ranges to see when your overnight highs and lows were established.
""")

# Default file path
default_file_path = os.path.join(os.path.dirname(__file__), 'overnight_data_detailed_breakdown.csv')

# Initialize df as None
df = None

# Load data function
@st.cache_data
def load_data(file_or_path):
    if isinstance(file_or_path, str):
        df = pd.read_csv(file_or_path)
    else:
        df = pd.read_csv(file_or_path)
    df['date'] = pd.to_datetime(df['date'])
    df['high_time'] = pd.to_datetime(df['high_time'], format='%H:%M:%S', errors='coerce').dt.time
    df['low_time'] = pd.to_datetime(df['low_time'], format='%H:%M:%S', errors='coerce').dt.time
    df['observation_time'] = pd.to_datetime(df['observation_time'], format='%H:%M:%S', errors='coerce').dt.time
    
    # Handle break times (may be None/NaN)
    if 'high_break_time' in df.columns:
        df['high_break_time'] = pd.to_datetime(df['high_break_time'], format='%H:%M:%S', errors='coerce').dt.time
    if 'low_break_time' in df.columns:
        df['low_break_time'] = pd.to_datetime(df['low_break_time'], format='%H:%M:%S', errors='coerce').dt.time
    
    return df

# Check if default file exists and load it
if os.path.exists(default_file_path):
    df = load_data(default_file_path)
    default_file_loaded = True
else:
    default_file_loaded = False

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Time helper functions for overnight session
def time_to_minutes_overnight(t):
    """Convert time to minutes, handling overnight session (19:30-09:30)"""
    minutes = t.hour * 60 + t.minute
    # Times before noon are considered next day in overnight context
    if t.hour < 12:
        minutes += 24 * 60
    return minutes

def minutes_to_time_overnight(minutes):
    """Convert minutes back to time, handling overflow past midnight"""
    minutes = minutes % (24 * 60)
    return time(minutes // 60, minutes % 60)

def generate_5min_intervals_overnight():
    """Generate list of times in 5-minute intervals for overnight session (19:30 to 09:30)"""
    intervals = []
    
    # Evening portion: 19:30 to 23:55
    start_minutes = 19 * 60 + 30
    end_evening = 23 * 60 + 55
    for minutes in range(start_minutes, end_evening + 1, 5):
        hours = minutes // 60
        mins = minutes % 60
        intervals.append(time(hours, mins))
    
    # Morning portion: 00:00 to 09:30
    start_morning = 0
    end_morning = 9 * 60 + 30
    for minutes in range(start_morning, end_morning + 1, 5):
        hours = minutes // 60
        mins = minutes % 60
        intervals.append(time(hours, mins))
    
    return intervals

time_intervals = generate_5min_intervals_overnight()
time_interval_strings = [t.strftime('%H:%M') for t in time_intervals]

if df is not None:
    
    # Quick Time Window Filter
    st.sidebar.subheader("⚡ Quick Time Window")
    
    window_minutes = st.sidebar.number_input(
        "Window Size (±minutes)",
        min_value=5,
        max_value=120,
        value=30,
        step=5,
        help="Set the time window before and after the center time (in minutes)"
    )
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        use_quick_filter_high = st.sidebar.checkbox(
            "Enable High",
            value=False,
            help=f"Set High range to ±{window_minutes} min around center time"
        )
    with col2:
        use_quick_filter_low = st.sidebar.checkbox(
            "Enable Low",
            value=False,
            help=f"Set Low range to ±{window_minutes} min around center time"
        )
    
    # High center time
    if use_quick_filter_high:
        high_center_time_str = st.sidebar.selectbox(
            "High Center",
            options=time_interval_strings,
            index=time_interval_strings.index('21:00') if '21:00' in time_interval_strings else 18,
            help=f"High range will be ±{window_minutes} min around this time"
        )
        
        high_center_time = datetime.strptime(high_center_time_str, '%H:%M').time()
        high_center_minutes = time_to_minutes_overnight(high_center_time)
        high_start_minutes = max(19 * 60 + 30, high_center_minutes - window_minutes)
        high_end_minutes = min(9 * 60 + 30 + 24 * 60, high_center_minutes + window_minutes)
        high_start_minutes = (high_start_minutes // 5) * 5
        high_end_minutes = (high_end_minutes // 5) * 5
        high_start = minutes_to_time_overnight(high_start_minutes)
        high_end = minutes_to_time_overnight(high_end_minutes)
        
        st.sidebar.success(f"🟢 High: {high_start.strftime('%H:%M')} - {high_end.strftime('%H:%M')}")
    
    # Low center time
    if use_quick_filter_low:
        low_center_time_str = st.sidebar.selectbox(
            "Low Center",
            options=time_interval_strings,
            index=time_interval_strings.index('01:00') if '01:00' in time_interval_strings else 66,
            help=f"Low range will be ±{window_minutes} min around this time"
        )
        
        low_center_time = datetime.strptime(low_center_time_str, '%H:%M').time()
        low_center_minutes = time_to_minutes_overnight(low_center_time)
        low_start_minutes = max(19 * 60 + 30, low_center_minutes - window_minutes)
        low_end_minutes = min(9 * 60 + 30 + 24 * 60, low_center_minutes + window_minutes)
        low_start_minutes = (low_start_minutes // 5) * 5
        low_end_minutes = (low_end_minutes // 5) * 5
        low_start = minutes_to_time_overnight(low_start_minutes)
        low_end = minutes_to_time_overnight(low_end_minutes)
        
        st.sidebar.success(f"🔴 Low: {low_start.strftime('%H:%M')} - {low_end.strftime('%H:%M')}")
    
    if use_quick_filter_high or use_quick_filter_low:
        st.sidebar.markdown("---")
    else:
        st.sidebar.markdown("---")
    
    # High time range filter
    st.sidebar.subheader("⬆️ High Formation Time Range")
    
    if not use_quick_filter_high:
        high_start_str = st.sidebar.selectbox(
            "High Start Time",
            options=time_interval_strings,
            index=0,
            help="Only consider highs that occurred after this time"
        )
        high_end_str = st.sidebar.selectbox(
            "High End Time",
            options=time_interval_strings,
            index=len(time_interval_strings) - 1,
            help="Only consider highs that occurred before this time (inclusive)"
        )
        
        high_start = datetime.strptime(high_start_str, '%H:%M').time()
        high_end = datetime.strptime(high_end_str, '%H:%M').time()
    else:
        st.sidebar.text(f"Start: {high_start.strftime('%H:%M')}")
        st.sidebar.text(f"End: {high_end.strftime('%H:%M')}")
    
    # Low time range filter
    st.sidebar.subheader("⬇️ Low Formation Time Range")
    
    if not use_quick_filter_low:
        low_start_str = st.sidebar.selectbox(
            "Low Start Time",
            options=time_interval_strings,
            index=0,
            help="Only consider lows that occurred after this time"
        )
        low_end_str = st.sidebar.selectbox(
            "Low End Time",
            options=time_interval_strings,
            index=len(time_interval_strings) - 1,
            help="Only consider lows that occurred before this time (inclusive)"
        )
        
        low_start = datetime.strptime(low_start_str, '%H:%M').time()
        low_end = datetime.strptime(low_end_str, '%H:%M').time()
    else:
        st.sidebar.text(f"Start: {low_start.strftime('%H:%M')}")
        st.sidebar.text(f"End: {low_end.strftime('%H:%M')}")
    
    st.sidebar.markdown("---")
    
    # Observation time filter
    st.sidebar.subheader("🕐 Observation Time")
    # Sort observation times using overnight logic (19:30, 20:00, ..., 23:30, 00:00, ..., 09:30)
    available_obs_times_raw = df['observation_time'].dropna().unique()
    available_obs_times = sorted(available_obs_times_raw, key=lambda t: time_to_minutes_overnight(t))
    available_obs_time_strings = [t.strftime('%H:%M') for t in available_obs_times]
    
    default_obs_idx = len(available_obs_time_strings) - 1  # Default to 9:30 (last time)
    if '09:30' in available_obs_time_strings:
        default_obs_idx = available_obs_time_strings.index('09:30')
    
    selected_obs_time_str = st.sidebar.selectbox(
        "Select Observation Time",
        options=available_obs_time_strings,
        index=default_obs_idx,
        help="Time to check if highs and lows still hold"
    )
    
    selected_obs_time = datetime.strptime(selected_obs_time_str, '%H:%M').time()
    
    # Date range filter
    st.sidebar.markdown("---")
    st.sidebar.subheader("📅 Date Range")
    
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    
    date_range = st.sidebar.date_input(
        "Filter by Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        help="Select date range to analyze"
    )
    
    # Apply filters
    filtered_df = df[df['observation_time'] == selected_obs_time].copy()
    
    # Date range filter
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['date'].dt.date >= start_date) &
            (filtered_df['date'].dt.date <= end_date)
        ]
    
    # Time range filters (handling overnight logic)
    high_start_minutes = time_to_minutes_overnight(high_start)
    high_end_minutes = time_to_minutes_overnight(high_end)
    low_start_minutes = time_to_minutes_overnight(low_start)
    low_end_minutes = time_to_minutes_overnight(low_end)
    
    def is_time_in_range_overnight(t, start_minutes, end_minutes):
        """Check if time is in range, handling overnight session"""
        t_minutes = time_to_minutes_overnight(t)
        return start_minutes <= t_minutes <= end_minutes
    
    high_mask = filtered_df['high_time'].apply(
        lambda x: is_time_in_range_overnight(x, high_start_minutes, high_end_minutes) if pd.notna(x) else False
    )
    low_mask = filtered_df['low_time'].apply(
        lambda x: is_time_in_range_overnight(x, low_start_minutes, low_end_minutes) if pd.notna(x) else False
    )
    
    filtered_df = filtered_df[high_mask & low_mask]
    
    # Display metrics
    st.header("📊 Key Metrics")
    
    if len(filtered_df) > 0:
        total_days = len(filtered_df)
        high_holds = (~filtered_df['high_broken']).sum()
        low_holds = (~filtered_df['low_broken']).sum()
        both_hold = ((~filtered_df['high_broken']) & (~filtered_df['low_broken'])).sum()
        
        high_hold_pct = (high_holds / total_days) * 100
        low_hold_pct = (low_holds / total_days) * 100
        both_hold_pct = (both_hold / total_days) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Days", f"{total_days:,}")
        with col2:
            st.metric("High Holds", f"{high_hold_pct:.1f}%", f"{high_holds}/{total_days}")
        with col3:
            st.metric("Low Holds", f"{low_hold_pct:.1f}%", f"{low_holds}/{total_days}")
        with col4:
            st.metric("Both Hold", f"{both_hold_pct:.1f}%", f"{both_hold}/{total_days}")
        
        st.markdown("---")
        
        # Probability Over Time Chart
        st.header("📈 Hold Probability Over Time")
        
        # Sort observation times using overnight logic
        obs_times_raw = [t for t in df['observation_time'].unique() if pd.notna(t)]
        obs_times_list = sorted(obs_times_raw, key=lambda t: time_to_minutes_overnight(t))
        
        # Filter to only show observation period (03:00 onwards, not range formation period)
        # Range formation is 19:30-03:00, observation starts at 03:00
        obs_times_filtered = [t for t in obs_times_list if time_to_minutes_overnight(t) >= time_to_minutes_overnight(time(3, 0))]
        
        prob_data = []
        
        for obs_time in obs_times_filtered:
            temp_df = df[df['observation_time'] == obs_time].copy()
            
            # Apply same filters
            if len(date_range) == 2:
                temp_df = temp_df[
                    (temp_df['date'].dt.date >= start_date) &
                    (temp_df['date'].dt.date <= end_date)
                ]
            
            high_mask = temp_df['high_time'].apply(
                lambda x: is_time_in_range_overnight(x, high_start_minutes, high_end_minutes) if pd.notna(x) else False
            )
            low_mask = temp_df['low_time'].apply(
                lambda x: is_time_in_range_overnight(x, low_start_minutes, low_end_minutes) if pd.notna(x) else False
            )
            temp_df = temp_df[high_mask & low_mask]
            
            if len(temp_df) > 0:
                high_hold_prob = (~temp_df['high_broken']).sum() / len(temp_df) * 100
                low_hold_prob = (~temp_df['low_broken']).sum() / len(temp_df) * 100
                both_hold_prob = ((~temp_df['high_broken']) & (~temp_df['low_broken'])).sum() / len(temp_df) * 100
                
                prob_data.append({
                    'time': obs_time,
                    'time_str': obs_time.strftime('%H:%M'),
                    'high_hold': high_hold_prob,
                    'low_hold': low_hold_prob,
                    'both_hold': both_hold_prob,
                    'sample_size': len(temp_df)
                })
        
        if prob_data:
            prob_df = pd.DataFrame(prob_data)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=prob_df['time_str'],
                y=prob_df['high_hold'],
                mode='lines+markers',
                name='High Holds',
                line=dict(color='#2ecc71', width=3),
                marker=dict(size=8),
                hovertemplate='<b>%{x}</b><br>High Hold: %{y:.2f}%<br>Sample: %{customdata}<extra></extra>',
                customdata=prob_df['sample_size']
            ))
            
            fig.add_trace(go.Scatter(
                x=prob_df['time_str'],
                y=prob_df['low_hold'],
                mode='lines+markers',
                name='Low Holds',
                line=dict(color='#e74c3c', width=3),
                marker=dict(size=8),
                hovertemplate='<b>%{x}</b><br>Low Hold: %{y:.2f}%<br>Sample: %{customdata}<extra></extra>',
                customdata=prob_df['sample_size']
            ))
            
            fig.add_trace(go.Scatter(
                x=prob_df['time_str'],
                y=prob_df['both_hold'],
                mode='lines+markers',
                name='Both Hold',
                line=dict(color='#3498db', width=3, dash='dash'),
                marker=dict(size=8),
                hovertemplate='<b>%{x}</b><br>Both Hold: %{y:.2f}%<br>Sample: %{customdata}<extra></extra>',
                customdata=prob_df['sample_size']
            ))
            
            fig.update_layout(
                title=f"Hold Probability Throughout Overnight Session",
                xaxis_title="Time",
                yaxis_title="Hold Probability (%)",
                hovermode='x unified',
                height=500,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                yaxis=dict(range=[0, 105])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption(f"📊 Chart shows observation period only (03:00-09:30). Range formation period (19:30-03:00) not shown - range is still being established.")
            st.caption(f"📊 Filtered: High ({high_start.strftime('%H:%M')} - {high_end.strftime('%H:%M')}), " + 
                      f"Low ({low_start.strftime('%H:%M')} - {low_end.strftime('%H:%M')})")
        
        st.markdown("---")
        
        # Which Level Breaks First Over Time
        st.header("2️⃣ Which Level Breaks First Over Time")
        
        # Helper function for categorizing first break
        def categorize_first_break(row):
            """Determine which level broke first"""
            high_broke = row['high_broken']
            low_broke = row['low_broken']
            high_time = row.get('high_break_time')
            low_time = row.get('low_break_time')
            
            if not high_broke and not low_broke:
                return 'Both Hold'
            elif high_broke and not low_broke:
                return 'High Breaks First'
            elif low_broke and not high_broke:
                return 'Low Breaks First'
            else:  # Both broke
                if pd.notna(high_time) and pd.notna(low_time):
                    # Convert to minutes for comparison
                    high_minutes = time_to_minutes_overnight(high_time)
                    low_minutes = time_to_minutes_overnight(low_time)
                    
                    if high_minutes < low_minutes:
                        return 'High Breaks First'
                    elif low_minutes < high_minutes:
                        return 'Low Breaks First'
                    else:
                        return 'Both Break Same Time'
                return 'Both Break (Unknown Order)'
        
        # Calculate which breaks first for each observation time (03:00 onwards only)
        breaks_first_data = []
        
        for obs_time in obs_times_filtered:  # Using filtered list (03:00+)
            temp_df = df[df['observation_time'] == obs_time].copy()
            
            # Apply same filters
            if len(date_range) == 2:
                temp_df = temp_df[
                    (temp_df['date'].dt.date >= start_date) &
                    (temp_df['date'].dt.date <= end_date)
                ]
            
            high_mask = temp_df['high_time'].apply(
                lambda x: is_time_in_range_overnight(x, high_start_minutes, high_end_minutes) if pd.notna(x) else False
            )
            low_mask = temp_df['low_time'].apply(
                lambda x: is_time_in_range_overnight(x, low_start_minutes, low_end_minutes) if pd.notna(x) else False
            )
            temp_df = temp_df[high_mask & low_mask]
            
            if len(temp_df) > 0:
                temp_df['first_break'] = temp_df.apply(categorize_first_break, axis=1)
                
                # Count occurrences
                total = len(temp_df)
                high_first_pct = (temp_df['first_break'] == 'High Breaks First').sum() / total * 100
                low_first_pct = (temp_df['first_break'] == 'Low Breaks First').sum() / total * 100
                both_hold_pct = (temp_df['first_break'] == 'Both Hold').sum() / total * 100
                both_same_pct = (temp_df['first_break'] == 'Both Break Same Time').sum() / total * 100
                
                breaks_first_data.append({
                    'observation_time': obs_time,
                    'time_str': obs_time.strftime('%H:%M'),
                    'high_first_pct': high_first_pct,
                    'low_first_pct': low_first_pct,
                    'both_hold_pct': both_hold_pct,
                    'both_same_pct': both_same_pct,
                    'total': total
                })
        
        if len(breaks_first_data) > 0:
            breaks_df = pd.DataFrame(breaks_first_data)
            
            # Create stacked area chart showing which breaks first
            fig_breaks = go.Figure()
            
            fig_breaks.add_trace(go.Scatter(
                x=breaks_df['time_str'],
                y=breaks_df['both_hold_pct'],
                mode='lines',
                name='Both Hold',
                line=dict(color='#3498db', width=0),
                stackgroup='one',
                fillcolor='rgba(52, 152, 219, 0.3)',
                hovertemplate='<b>%{x}</b><br>Both Hold: %{y:.1f}%<extra></extra>'
            ))
            
            fig_breaks.add_trace(go.Scatter(
                x=breaks_df['time_str'],
                y=breaks_df['high_first_pct'],
                mode='lines',
                name='High Breaks First',
                line=dict(color='#2ecc71', width=2),
                stackgroup='one',
                fillcolor='rgba(46, 204, 113, 0.6)',
                hovertemplate='<b>%{x}</b><br>High First: %{y:.1f}%<extra></extra>'
            ))
            
            fig_breaks.add_trace(go.Scatter(
                x=breaks_df['time_str'],
                y=breaks_df['low_first_pct'],
                mode='lines',
                name='Low Breaks First',
                line=dict(color='#e74c3c', width=2),
                stackgroup='one',
                fillcolor='rgba(231, 76, 60, 0.6)',
                hovertemplate='<b>%{x}</b><br>Low First: %{y:.1f}%<extra></extra>'
            ))
            
            fig_breaks.update_layout(
                title="Which Level Breaks First Over Time (Stacked %)",
                xaxis_title="Observation Time",
                yaxis_title="Percentage (%)",
                yaxis_range=[0, 100],
                hovermode='x unified',
                height=500,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig_breaks, use_container_width=True)
            
            # Summary table at selected observation time
            selected_idx = breaks_df[breaks_df['observation_time'] == selected_obs_time].index
            if len(selected_idx) > 0:
                selected_row = breaks_df.loc[selected_idx[0]]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "High Breaks First",
                        f"{selected_row['high_first_pct']:.1f}%",
                        help=f"At {selected_obs_time_str}, high broke first (or only high broke)"
                    )
                with col2:
                    st.metric(
                        "Low Breaks First",
                        f"{selected_row['low_first_pct']:.1f}%",
                        help=f"At {selected_obs_time_str}, low broke first (or only low broke)"
                    )
                with col3:
                    st.metric(
                        "Both Hold",
                        f"{selected_row['both_hold_pct']:.1f}%",
                        help=f"At {selected_obs_time_str}, neither level broke"
                    )
            
            st.info("💡 **Insight:** This stacked chart shows the breakdown at each observation time. " +
                   "Blue = both holding, Green = high broke (first or only), Red = low broke (first or only). " +
                   "Watch how the composition changes as time progresses through the morning.")
        else:
            st.warning("⚠️ No data available for break sequence analysis with current filters.")
        
        st.markdown("---")
        
        # Break Analysis
        st.header("⚡ Break Analysis")
        
        high_breaks = filtered_df[filtered_df['high_broken'] == True]
        low_breaks = filtered_df[filtered_df['low_broken'] == True]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🟢 High Break Times")
            if len(high_breaks) > 0:
                high_break_times = high_breaks['high_break_time'].dropna()
                if len(high_break_times) > 0:
                    # Sort break times in overnight order
                    break_time_counts = high_break_times.value_counts()
                    sorted_times = sorted(break_time_counts.index, key=lambda t: time_to_minutes_overnight(t))
                    break_time_counts = break_time_counts.reindex(sorted_times)
                    
                    fig_high_breaks = px.bar(
                        x=[t.strftime('%H:%M') for t in break_time_counts.index],
                        y=break_time_counts.values,
                        labels={'x': 'Break Time', 'y': 'Count'},
                        title=f"High Break Times Distribution ({len(high_breaks)} breaks)"
                    )
                    fig_high_breaks.update_traces(marker_color='#e74c3c')
                    st.plotly_chart(fig_high_breaks, use_container_width=True)
                else:
                    st.info("No break time data available")
            else:
                st.success(f"High never broken! (100% hold rate)")
        
        with col2:
            st.subheader("🔴 Low Break Times")
            if len(low_breaks) > 0:
                low_break_times = low_breaks['low_break_time'].dropna()
                if len(low_break_times) > 0:
                    # Sort break times in overnight order
                    break_time_counts = low_break_times.value_counts()
                    sorted_times = sorted(break_time_counts.index, key=lambda t: time_to_minutes_overnight(t))
                    break_time_counts = break_time_counts.reindex(sorted_times)
                    
                    fig_low_breaks = px.bar(
                        x=[t.strftime('%H:%M') for t in break_time_counts.index],
                        y=break_time_counts.values,
                        labels={'x': 'Break Time', 'y': 'Count'},
                        title=f"Low Break Times Distribution ({len(low_breaks)} breaks)"
                    )
                    fig_low_breaks.update_traces(marker_color='#2ecc71')
                    st.plotly_chart(fig_low_breaks, use_container_width=True)
                else:
                    st.info("No break time data available")
            else:
                st.success(f"Low never broken! (100% hold rate)")
        
        st.markdown("---")
        
        # Data table
        st.header("📋 Filtered Data Table")
        
        display_cols = ['date', 'high_value', 'high_time', 'high_broken']
        if 'high_break_time' in filtered_df.columns:
            display_cols.append('high_break_time')
        display_cols.extend(['low_value', 'low_time', 'low_broken'])
        if 'low_break_time' in filtered_df.columns:
            display_cols.append('low_break_time')
        
        display_df = filtered_df[display_cols].copy()
        display_df['date'] = display_df['date'].dt.date
        display_df = display_df.sort_values('date', ascending=False)
        
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400
        )
        
        # Download
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name=f"overnight_filtered_{selected_obs_time_str.replace(':', '')}.csv",
            mime="text/csv"
        )
    
    else:
        st.warning("⚠️ No data matches your current filters. Try adjusting the time ranges or date filters.")

else:
    # No data loaded
    st.info("👈 Please check the Data Input section at the bottom of the sidebar.")
    
    st.sidebar.header("📁 Data Input")
    st.sidebar.warning("⚠️ No data loaded")
    st.sidebar.markdown("**Please upload a CSV file or place `overnight_data_detailed_breakdown.csv` in the app directory:**")
    uploaded_file = st.sidebar.file_uploader(
        "Upload overnight breakdown CSV",
        type=['csv'],
        help="Upload the overnight_data_detailed_breakdown.csv file"
    )
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.sidebar.success(f"✅ Loaded {len(df)} records")
        st.sidebar.info(f"📅 Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        st.rerun()
    
    st.markdown("""
    ### 🌙 Overnight Session Analysis
    
    This app analyzes crude oil futures overnight session performance:
    - **Range Period**: 19:30 - 03:00 (overnight session)
    - **Observation**: Track until 9:30 (next morning)
    
    ### How to use:
    
    1. **Generate Data**: Use `overnight_data_miner.py` to process your 5-minute CL candle data
    2. **Load Data**: Upload the CSV or place it in the app directory
    3. **Filter**: Set time ranges for high/low formation
    4. **Analyze**: View probabilities, break patterns, and statistics
    
    ### Example Question:
    *"If the overnight high forms around 21:00 (±30min) and low around 01:00 (±30min), 
    what's the probability they both hold until 9:30?"*
    
    This app will show you! 📊
    """)
