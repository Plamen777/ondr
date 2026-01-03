#!/usr/bin/env python3
"""
Streamlit App for Progressive Range Analysis with Break Tracking
Enhanced UI/UX with comprehensive break statistics and visualizations
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import time, datetime
import os

# Page config
st.set_page_config(
    page_title="Progressive Range Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Progressive Range Analysis with Break Tracking")
st.markdown("Analyze expanding ranges (16:00 → observation time) and track when/which extremity breaks first")

# Load data
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df['range_obs_time'] = pd.to_datetime(df['range_obs_time'], format='%H:%M:%S').dt.time
    df['check_obs_time'] = pd.to_datetime(df['check_obs_time'], format='%H:%M:%S').dt.time
    df['high_time'] = pd.to_datetime(df['high_time'], format='%H:%M:%S').dt.time
    df['low_time'] = pd.to_datetime(df['low_time'], format='%H:%M:%S').dt.time
    
    # Handle break times (may be None/NaN)
    if 'high_break_time' in df.columns:
        df['high_break_time'] = pd.to_datetime(df['high_break_time'], format='%H:%M:%S', errors='coerce').dt.time
    if 'low_break_time' in df.columns:
        df['low_break_time'] = pd.to_datetime(df['low_break_time'], format='%H:%M:%S', errors='coerce').dt.time
    
    return df

# Try to load
default_path = 'progressive_range_with_breaks.csv'
if os.path.exists(default_path):
    df = load_data(default_path)
else:
    uploaded = st.file_uploader("Upload progressive_range_with_breaks.csv", type=['csv'])
    if uploaded:
        df = load_data(uploaded)
    else:
        st.warning("Please upload progressive_range_with_breaks.csv")
        st.stop()

# Helper function
def time_to_minutes(t):
    """Convert time to minutes (handle overnight)"""
    minutes = t.hour * 60 + t.minute
    if t.hour < 12:
        minutes += 24 * 60
    return minutes

# ===========================
# SIDEBAR FILTERS
# ===========================
st.sidebar.header("🔍 Filters")

# Range Observation Time
st.sidebar.subheader("🕐 Range Set Time")
st.sidebar.markdown("**When was the range established?**")

available_range_times = sorted(df['range_obs_time'].unique(), key=time_to_minutes)

# Initialize session state
if 'selected_range_time' not in st.session_state:
    st.session_state.selected_range_time = available_range_times[3] if len(available_range_times) > 3 else available_range_times[0]

selected_range_time = st.sidebar.selectbox(
    "Range Observation Time",
    options=available_range_times,
    format_func=lambda t: t.strftime('%H:%M'),
    index=available_range_times.index(st.session_state.selected_range_time) if st.session_state.selected_range_time in available_range_times else 0,
    help="The time when the range (high/low) was established"
)
st.session_state.selected_range_time = selected_range_time

st.sidebar.success(f"**Range:** 16:00 → {selected_range_time.strftime('%H:%M')}")

# Filter by range observation time
df_range = df[df['range_obs_time'] == selected_range_time].copy()

st.sidebar.markdown("---")

# Quick Time Window Filter
st.sidebar.subheader("⚡ Quick Time Window")

# Window size
if 'window_minutes' not in st.session_state:
    st.session_state.window_minutes = 30

window_minutes = st.sidebar.number_input(
    "Window Size (±minutes)",
    min_value=5,
    max_value=300,
    value=st.session_state.window_minutes,
    step=5,
    help="Set the time window before and after the center time (in minutes)"
)
st.session_state.window_minutes = window_minutes

# Separate toggles for High and Low
col1, col2 = st.sidebar.columns(2)
with col1:
    use_quick_high = st.checkbox(
        "Enable High",
        value=st.session_state.get('use_quick_high', False),
        help=f"Set High range to ±{window_minutes} min around center time"
    )
    st.session_state.use_quick_high = use_quick_high

with col2:
    use_quick_low = st.checkbox(
        "Enable Low",
        value=st.session_state.get('use_quick_low', False),
        help=f"Set Low range to ±{window_minutes} min around center time"
    )
    st.session_state.use_quick_low = use_quick_low

# Helper function to calculate range from center time and offset
def calculate_time_range(center_time_str, offset_minutes):
    """Calculate start and end times from center time +/- offset"""
    center = datetime.strptime(center_time_str, '%H:%M')
    center_minutes = center.hour * 60 + center.minute
    
    start_minutes = center_minutes - offset_minutes
    end_minutes = center_minutes + offset_minutes
    
    # Handle overnight wrapping
    start_minutes = start_minutes % (24 * 60)
    end_minutes = end_minutes % (24 * 60)
    
    start_time = time(start_minutes // 60, start_minutes % 60)
    end_time = time(end_minutes // 60, end_minutes % 60)
    
    return start_time, end_time

# Show separator if any quick filter is enabled
if use_quick_high or use_quick_low:
    st.sidebar.markdown("---")

# HIGH FILTER
st.sidebar.subheader("🟢 High Formation Time")

available_high_times = sorted(df_range['high_time'].unique(), key=time_to_minutes)
time_strings_high = [t.strftime('%H:%M') for t in available_high_times]

if use_quick_high:
    # Quick mode
    if 'high_center_time' not in st.session_state:
        st.session_state.high_center_time = '20:00'
    
    high_center_input = st.sidebar.text_input(
        "High Center Time",
        value=st.session_state.high_center_time,
        help="e.g., 20:00"
    )
    st.session_state.high_center_time = high_center_input
    
    try:
        high_start, high_end = calculate_time_range(high_center_input, window_minutes)
        st.sidebar.success(f"🟢 High: {high_start.strftime('%H:%M')} - {high_end.strftime('%H:%M')}")
    except:
        st.sidebar.error("❌ Invalid time format. Use HH:MM")
        high_start = available_high_times[0]
        high_end = available_high_times[-1]
else:
    # Manual mode
    if 'high_start_idx' not in st.session_state:
        st.session_state.high_start_idx = 0
    if 'high_end_idx' not in st.session_state:
        st.session_state.high_end_idx = len(time_strings_high) - 1
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        high_start_idx = st.selectbox(
            "High Start",
            range(len(time_strings_high)),
            format_func=lambda i: time_strings_high[i],
            index=st.session_state.high_start_idx,
            key='high_start_select'
        )
        st.session_state.high_start_idx = high_start_idx
    with col2:
        high_end_idx = st.selectbox(
            "High End",
            range(len(time_strings_high)),
            format_func=lambda i: time_strings_high[i],
            index=st.session_state.high_end_idx,
            key='high_end_select'
        )
        st.session_state.high_end_idx = high_end_idx
    
    high_start = available_high_times[high_start_idx]
    high_end = available_high_times[high_end_idx]
    st.sidebar.info(f"🟢 High: {high_start.strftime('%H:%M')} - {high_end.strftime('%H:%M')}")

# Apply high filter
high_start_min = time_to_minutes(high_start)
high_end_min = time_to_minutes(high_end)
df_range = df_range[
    df_range['high_time'].apply(lambda t: high_start_min <= time_to_minutes(t) <= high_end_min)
]

# LOW FILTER
st.sidebar.subheader("🔴 Low Formation Time")

available_low_times = sorted(df_range['low_time'].unique(), key=time_to_minutes)
time_strings_low = [t.strftime('%H:%M') for t in available_low_times]

if use_quick_low:
    # Quick mode
    if 'low_center_time' not in st.session_state:
        st.session_state.low_center_time = '01:00'
    
    low_center_input = st.sidebar.text_input(
        "Low Center Time",
        value=st.session_state.low_center_time,
        help="e.g., 01:00"
    )
    st.session_state.low_center_time = low_center_input
    
    try:
        low_start, low_end = calculate_time_range(low_center_input, window_minutes)
        st.sidebar.success(f"🔴 Low: {low_start.strftime('%H:%M')} - {low_end.strftime('%H:%M')}")
    except:
        st.sidebar.error("❌ Invalid time format. Use HH:MM")
        low_start = available_low_times[0]
        low_end = available_low_times[-1]
else:
    # Manual mode
    if 'low_start_idx' not in st.session_state:
        st.session_state.low_start_idx = 0
    if 'low_end_idx' not in st.session_state:
        st.session_state.low_end_idx = len(time_strings_low) - 1
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        low_start_idx = st.selectbox(
            "Low Start",
            range(len(time_strings_low)),
            format_func=lambda i: time_strings_low[i],
            index=st.session_state.low_start_idx,
            key='low_start_select'
        )
        st.session_state.low_start_idx = low_start_idx
    with col2:
        low_end_idx = st.selectbox(
            "Low End",
            range(len(time_strings_low)),
            format_func=lambda i: time_strings_low[i],
            index=st.session_state.low_end_idx,
            key='low_end_select'
        )
        st.session_state.low_end_idx = low_end_idx
    
    low_start = available_low_times[low_start_idx]
    low_end = available_low_times[low_end_idx]
    st.sidebar.info(f"🔴 Low: {low_start.strftime('%H:%M')} - {low_end.strftime('%H:%M')}")

# Apply low filter
low_start_min = time_to_minutes(low_start)
low_end_min = time_to_minutes(low_end)
df_range = df_range[
    df_range['low_time'].apply(lambda t: low_start_min <= time_to_minutes(t) <= low_end_min)
]

# Show filtered count prominently
st.sidebar.markdown("---")
unique_sessions = df_range.groupby(['date', 'range_obs_time', 'high_value', 'low_value']).size().reset_index(name='count')
total_sessions = len(unique_sessions)
total_days = df_range['date'].nunique()

st.sidebar.metric("📊 Filtered Sessions", f"{total_sessions:,}")
st.sidebar.metric("📅 Trading Days", f"{total_days:,}")

# Check if we have data
if len(df_range) == 0:
    st.error("❌ No data matches your filters!")
    st.stop()

# ===========================
# MAIN CONTENT
# ===========================

# Get check observation times
check_times = sorted(df_range['check_obs_time'].unique(), key=time_to_minutes)

# Get final check time data
final_check_time = check_times[-1]
df_final = df_range[df_range['check_obs_time'] == final_check_time]
sessions_final = df_final.groupby(['date', 'range_obs_time', 'high_value', 'low_value']).first().reset_index()

# Calculate key stats for final time
high_holds = (~sessions_final['high_broken']).sum()
high_breaks = sessions_final['high_broken'].sum()
low_holds = (~sessions_final['low_broken']).sum()
low_breaks = sessions_final['low_broken'].sum()

high_hold_pct = (high_holds / len(sessions_final) * 100) if len(sessions_final) > 0 else 0
high_break_pct = (high_breaks / len(sessions_final) * 100) if len(sessions_final) > 0 else 0
low_hold_pct = (low_holds / len(sessions_final) * 100) if len(sessions_final) > 0 else 0
low_break_pct = (low_breaks / len(sessions_final) * 100) if len(sessions_final) > 0 else 0

# 📈 Key Statistics
st.header("📈 Key Statistics")
st.markdown(f"**At observation time: {final_check_time.strftime('%H:%M')} | Sessions: {total_sessions:,}**")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "High Hold %",
        f"{high_hold_pct:.1f}%",
        delta=f"{high_holds} days",
        delta_color="normal"
    )

with col2:
    st.metric(
        "High Break %",
        f"{high_break_pct:.1f}%",
        delta=f"{high_breaks} days",
        delta_color="inverse"
    )

with col3:
    st.metric(
        "Low Hold %",
        f"{low_hold_pct:.1f}%",
        delta=f"{low_holds} days",
        delta_color="normal"
    )

with col4:
    st.metric(
        "Low Break %",
        f"{low_break_pct:.1f}%",
        delta=f"{low_breaks} days",
        delta_color="inverse"
    )

st.markdown("---")

# ⚡ Break Sequence Analysis
st.header("⚡ Break Sequence Analysis")
st.markdown("**Which level breaks first when both break?**")

# Helper function to convert time to comparable format
def time_to_minutes_safe(t):
    if pd.isna(t) or t is None:
        return None
    return time_to_minutes(t)

# Calculate break sequences
sessions_final['high_break_minutes'] = sessions_final['high_break_time'].apply(time_to_minutes_safe)
sessions_final['low_break_minutes'] = sessions_final['low_break_time'].apply(time_to_minutes_safe)

# Categorize each day
def categorize_break(row):
    high_broke = row['high_broken']
    low_broke = row['low_broken']
    high_time = row['high_break_minutes']
    low_time = row['low_break_minutes']
    
    if not high_broke and not low_broke:
        return 'Both Hold'
    elif high_broke and not low_broke:
        return 'Only High Breaks'
    elif low_broke and not high_broke:
        return 'Only Low Breaks'
    else:  # Both broke
        if high_time is not None and low_time is not None:
            if high_time < low_time:
                return 'High Breaks First'
            elif low_time < high_time:
                return 'Low Breaks First'
            else:
                return 'Both Break Same Time'
        return 'Both Break (Unknown Order)'

sessions_final['break_sequence'] = sessions_final.apply(categorize_break, axis=1)

# Categorize first break (for "Which Breaks First" analysis)
def categorize_first_break(row):
    high_broke = row['high_broken']
    low_broke = row['low_broken']
    high_time = row.get('high_break_minutes')
    low_time = row.get('low_break_minutes')
    
    if not high_broke and not low_broke:
        return 'Both Hold'
    elif high_broke and not low_broke:
        return 'High Breaks First'
    elif low_broke and not high_broke:
        return 'Low Breaks First'
    else:  # Both broke - compare times
        if high_time is not None and low_time is not None:
            if high_time < low_time:
                return 'High Breaks First'
            elif low_time < high_time:
                return 'Low Breaks First'
            else:
                return 'Both Break Same Time'
        return 'Both Break (Unknown Order)'


# Calculate statistics
sequence_counts = sessions_final['break_sequence'].value_counts()
sequence_pcts = (sequence_counts / len(sessions_final) * 100).round(2)

# Display metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    count = sequence_counts.get('Only High Breaks', 0)
    pct = sequence_pcts.get('Only High Breaks', 0.0)
    st.metric(
        "🟢 Only High Breaks",
        f"{pct:.1f}%",
        delta=f"{count} days"
    )

with col2:
    count = sequence_counts.get('Only Low Breaks', 0)
    pct = sequence_pcts.get('Only Low Breaks', 0.0)
    st.metric(
        "🔴 Only Low Breaks",
        f"{pct:.1f}%",
        delta=f"{count} days"
    )

with col3:
    # Calculate "Both Break %" - sum of all scenarios where both broke
    both_break_categories = ['High Breaks First', 'Low Breaks First', 'Both Break Same Time', 'Both Break (Unknown Order)']
    both_break_count = sum(sequence_counts.get(cat, 0) for cat in both_break_categories)
    both_break_pct = (both_break_count / len(sessions_final) * 100) if len(sessions_final) > 0 else 0.0
    st.metric(
        "⚫ Both Break",
        f"{both_break_pct:.1f}%",
        delta=f"{both_break_count} days"
    )

with col4:
    count = sequence_counts.get('Both Hold', 0)
    pct = sequence_pcts.get('Both Hold', 0.0)
    st.metric(
        "🟦 Both Hold",
        f"{pct:.1f}%",
        delta=f"{count} days"
    )

st.markdown("---")

# 🔄 Which Level Breaks First? Bar Chart
st.header("🔄 Which Level Breaks First?")
st.markdown(f"**Overall probability of which extremity gets taken out first | Sessions: {total_sessions:,}**")

sessions_final['first_break'] = sessions_final.apply(categorize_first_break, axis=1)

# Calculate statistics
first_break_counts = sessions_final['first_break'].value_counts()
first_break_pcts = (first_break_counts / len(sessions_final) * 100).round(2)

# Filter to show only the breaking scenarios
break_data = sessions_final[sessions_final['first_break'] != 'Both Hold']

if len(break_data) > 0:
    break_counts = break_data['first_break'].value_counts()
    break_pcts = (break_counts / len(break_data) * 100).round(2)
    
    fig_first_break = go.Figure()
    
    # Define colors
    colors_map = {
        'High Breaks First': '#2ecc71',  # Green for High
        'Low Breaks First': '#e74c3c',   # Red for Low
        'Both Break Same Time': '#95a5a6'
    }
    
    bar_colors = [colors_map.get(cat, '#95a5a6') for cat in break_counts.index]
    
    fig_first_break.add_trace(go.Bar(
        x=break_counts.index,
        y=break_pcts.values,
        text=[f"{pct:.1f}%<br>({count} days)" for pct, count in zip(break_pcts.values, break_counts.values)],
        textposition='outside',
        marker=dict(
            color=bar_colors,
            line=dict(color='rgba(0,0,0,0.3)', width=2)
        ),
        hovertemplate='<b>%{x}</b><br>Percentage: %{y:.1f}%<br><extra></extra>'
    ))
    
    fig_first_break.update_layout(
        title=f"Which Level Breaks First? ({len(break_data)} days with at least one break)",
        yaxis_title="Percentage of Breaking Days",
        xaxis_title="First Break",
        height=450,
        yaxis_range=[0, max(break_pcts.values) * 1.2] if len(break_pcts) > 0 else [0, 100],
        showlegend=False
    )
    
    st.plotly_chart(fig_first_break, use_container_width=True)
    
    # Summary
    col1, col2 = st.columns(2)
    with col1:
        high_first_count = first_break_counts.get('High Breaks First', 0)
        high_first_pct = first_break_pcts.get('High Breaks First', 0.0)
        st.info(f"**🟢 High Breaks First:** {high_first_pct:.1f}% ({high_first_count}/{len(sessions_final)} days)")
    with col2:
        low_first_count = first_break_counts.get('Low Breaks First', 0)
        low_first_pct = first_break_pcts.get('Low Breaks First', 0.0)
        st.info(f"**🔴 Low Breaks First:** {low_first_pct:.1f}% ({low_first_count}/{len(sessions_final)} days)")

st.markdown("---")

# 📊 Cumulative Break % Over Time (BAR CHART)
st.header("📊 Cumulative Break % Over Time")
st.markdown(f"**Total % of ranges that broke (either direction) | Sessions: {total_sessions:,}**")

# Calculate total break % at each check time
total_break_data = []
for check_time in check_times:
    df_check = df_range[df_range['check_obs_time'] == check_time]
    sessions_check = df_check.groupby(['date', 'range_obs_time', 'high_value', 'low_value']).first().reset_index()
    
    total = len(sessions_check)
    if total > 0:
        at_least_one_broke = (sessions_check['high_broken'] | sessions_check['low_broken']).sum()
        total_break_pct = (at_least_one_broke / total) * 100
        
        total_break_data.append({
            'time_str': check_time.strftime('%H:%M'),
            'total_break_pct': total_break_pct
        })

if total_break_data:
    total_break_df = pd.DataFrame(total_break_data)
    
    fig_total_break = go.Figure()
    
    fig_total_break.add_trace(go.Bar(
        x=total_break_df['time_str'],
        y=total_break_df['total_break_pct'],
        marker_color='#e67e22',
        text=[f"{v:.1f}%" for v in total_break_df['total_break_pct']],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Total Break: %{y:.1f}%<extra></extra>'
    ))
    
    fig_total_break.update_layout(
        title=f"Total % of Ranges That Broke Over Time",
        xaxis_title="Check Time",
        yaxis_title="Cumulative Break %",
        yaxis_range=[0, 100],
        height=450,
        showlegend=False
    )
    
    st.plotly_chart(fig_total_break, use_container_width=True)
    
    st.caption(f"📈 Shows cumulative % of ranges where at least one level broke. Final: {total_break_df.iloc[-1]['total_break_pct']:.1f}%")

st.markdown("---")

# Break Probability Over Time (LINE CHART - existing)
st.header("1️⃣ Hold Probability Over Time")
st.markdown(f"**Probability of levels holding throughout the observation period | Sessions: {total_sessions:,}**")

# Collect hold probability data
hold_prob_data = []
for check_time in check_times:
    df_check = df_range[df_range['check_obs_time'] == check_time]
    sessions_at_check = df_check.groupby(['date', 'range_obs_time', 'high_value', 'low_value']).first().reset_index()
    
    total = len(sessions_at_check)
    if total > 0:
        high_holds = (~sessions_at_check['high_broken']).sum()
        low_holds = (~sessions_at_check['low_broken']).sum()
        
        high_hold_pct = (high_holds / total) * 100
        low_hold_pct = (low_holds / total) * 100
        
        hold_prob_data.append({
            'check_time': check_time,
            'time_str': check_time.strftime('%H:%M'),
            'high_hold_pct': high_hold_pct,
            'low_hold_pct': low_hold_pct,
            'total_sessions': total
        })

if hold_prob_data:
    hold_df = pd.DataFrame(hold_prob_data)
    
    # Create LINE chart
    fig_holds = go.Figure()
    
    fig_holds.add_trace(go.Scatter(
        x=hold_df['time_str'],
        y=hold_df['high_hold_pct'],
        mode='lines+markers',
        name='High Hold %',
        line=dict(color='#2ecc71', width=3),  # Green for high
        marker=dict(size=8),
        hovertemplate='<b>%{x}</b><br>High Hold: %{y:.1f}%<extra></extra>'
    ))
    
    fig_holds.add_trace(go.Scatter(
        x=hold_df['time_str'],
        y=hold_df['low_hold_pct'],
        mode='lines+markers',
        name='Low Hold %',
        line=dict(color='#e74c3c', width=3),  # Red for low
        marker=dict(size=8),
        hovertemplate='<b>%{x}</b><br>Low Hold: %{y:.1f}%<extra></extra>'
    ))
    
    fig_holds.update_layout(
        title=f"Probability of Levels Holding Throughout Observation Period",
        xaxis_title="Check Time",
        yaxis_title="Hold Percentage (%)",
        yaxis_range=[0, 100],
        hovermode='x unified',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig_holds, use_container_width=True)
    
    st.info("💡 **Insight:** Green = High hold %, Red = Low hold %. Later times show higher hold % since less time remains for breaks.")

st.markdown("---")

# 2️⃣ Which Level Breaks First Over Time
st.header("2️⃣ Which Level Breaks First Over Time")
st.markdown(f"**Track which extremity breaks first as time progresses | Sessions: {total_sessions:,}**")

# Calculate which breaks first at each check time
breaks_first_over_time = []

for check_time in check_times:
    df_check = df_range[df_range['check_obs_time'] == check_time]
    sessions_check = df_check.groupby(['date', 'range_obs_time', 'high_value', 'low_value']).first().reset_index()
    
    if len(sessions_check) > 0:
        # Add break minutes
        sessions_check['high_break_minutes'] = sessions_check['high_break_time'].apply(time_to_minutes_safe)
        sessions_check['low_break_minutes'] = sessions_check['low_break_time'].apply(time_to_minutes_safe)
        
        # Categorize first break
        sessions_check['first_break'] = sessions_check.apply(categorize_first_break, axis=1)
        
        # Calculate percentages
        total = len(sessions_check)
        high_first_pct = (sessions_check['first_break'] == 'High Breaks First').sum() / total * 100
        low_first_pct = (sessions_check['first_break'] == 'Low Breaks First').sum() / total * 100
        both_hold_pct = (sessions_check['first_break'] == 'Both Hold').sum() / total * 100
        
        breaks_first_over_time.append({
            'check_time': check_time,
            'time_str': check_time.strftime('%H:%M'),
            'high_first_pct': high_first_pct,
            'low_first_pct': low_first_pct,
            'both_hold_pct': both_hold_pct,
            'total': total
        })

if breaks_first_over_time:
    breaks_time_df = pd.DataFrame(breaks_first_over_time)
    
    # Create line chart showing which breaks first over time
    fig_breaks_time = go.Figure()
    
    fig_breaks_time.add_trace(go.Scatter(
        x=breaks_time_df['time_str'],
        y=breaks_time_df['high_first_pct'],
        mode='lines+markers',
        name='High Breaks First',
        line=dict(color='#2ecc71', width=3),  # Green for high
        marker=dict(size=8),
        hovertemplate='<b>%{x}</b><br>High First: %{y:.1f}%<extra></extra>'
    ))
    
    fig_breaks_time.add_trace(go.Scatter(
        x=breaks_time_df['time_str'],
        y=breaks_time_df['low_first_pct'],
        mode='lines+markers',
        name='Low Breaks First',
        line=dict(color='#e74c3c', width=3),  # Red for low
        marker=dict(size=8),
        hovertemplate='<b>%{x}</b><br>Low First: %{y:.1f}%<extra></extra>'
    ))
    
    fig_breaks_time.update_layout(
        title=f"Which Level Breaks First Over Time (as % of all sessions)",
        xaxis_title="Check Time",
        yaxis_title="Percentage (%)",
        yaxis_range=[0, 100],
        hovermode='x unified',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig_breaks_time, use_container_width=True)
    
    st.info("💡 **Insight:** Green = High breaks first, Red = Low breaks first (as % of all days). Shows directional bias evolution over time.")

st.markdown("---")

# Formation Time Distributions
st.header("📍 Formation Time Distributions")
st.markdown(f"**When do highs and lows form? | Sessions: {total_sessions:,}**")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🟢 High Formation Times")
    high_formation = sessions_final['high_time'].value_counts()
    high_formation_sorted = high_formation.reindex(sorted(high_formation.index, key=time_to_minutes))
    
    fig_high_form = go.Figure()
    fig_high_form.add_trace(go.Bar(
        x=[t.strftime('%H:%M') for t in high_formation_sorted.index],
        y=high_formation_sorted.values,
        marker_color='#2ecc71'
    ))
    fig_high_form.update_layout(
        title="When Highs Form",
        xaxis_title="Formation Time",
        yaxis_title="Count",
        height=350
    )
    st.plotly_chart(fig_high_form, use_container_width=True)

with col2:
    st.subheader("🔴 Low Formation Times")
    low_formation = sessions_final['low_time'].value_counts()
    low_formation_sorted = low_formation.reindex(sorted(low_formation.index, key=time_to_minutes))
    
    fig_low_form = go.Figure()
    fig_low_form.add_trace(go.Bar(
        x=[t.strftime('%H:%M') for t in low_formation_sorted.index],
        y=low_formation_sorted.values,
        marker_color='#e74c3c'
    ))
    fig_low_form.update_layout(
        title="When Lows Form",
        xaxis_title="Formation Time",
        yaxis_title="Count",
        height=350
    )
    st.plotly_chart(fig_low_form, use_container_width=True)

st.markdown("---")

# Data Table
st.header("📋 Sample Data")
st.markdown(f"**First 20 sessions | Total: {total_sessions:,}**")

# Select only columns that exist
available_cols = ['date', 'high_value', 'high_time', 'low_value', 'low_time']
if 'high_broken' in sessions_final.columns:
    available_cols.append('high_broken')
if 'high_break_time' in sessions_final.columns:
    available_cols.append('high_break_time')
if 'low_broken' in sessions_final.columns:
    available_cols.append('low_broken')
if 'low_break_time' in sessions_final.columns:
    available_cols.append('low_break_time')
if 'first_break' in sessions_final.columns:
    available_cols.append('first_break')

sample_df = sessions_final[available_cols].head(20).copy()
sample_df['date'] = sample_df['date'].dt.date
st.dataframe(sample_df, use_container_width=True)

# Download
csv = sessions_final.to_csv(index=False)
st.download_button(
    label="📥 Download Full Filtered Data",
    data=csv,
    file_name=f"filtered_range_{selected_range_time.strftime('%H%M')}.csv",
    mime="text/csv"
)
