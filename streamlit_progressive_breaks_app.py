#!/usr/bin/env python3
"""
Streamlit App for Progressive Range Analysis with Break Tracking
Shows cumulative break probability and which extremity breaks first
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import time, datetime
import os

# Page config
st.set_page_config(
    page_title="Progressive Range with Breaks",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Progressive Range Analysis with Break Tracking")
st.markdown("Analyze expanding ranges and track when/which extremity breaks first")

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

# Sidebar - Range Observation Time (when range was "set")
st.sidebar.header("🕐 Range Set Time")
st.sidebar.markdown("**When was the range established?**")

available_range_times = sorted(df['range_obs_time'].unique(), key=time_to_minutes)
selected_range_time = st.sidebar.selectbox(
    "Range Observation Time",
    options=available_range_times,
    format_func=lambda t: t.strftime('%H:%M'),
    index=3 if len(available_range_times) > 3 else 0,  # Default to 04:00
    help="The time when the range (high/low) was established"
)

st.sidebar.info(f"**Range set at:** {selected_range_time.strftime('%H:%M')}")

# Filter by range observation time
df_range = df[df['range_obs_time'] == selected_range_time].copy()

# Sidebar - Formation Time Filters
st.sidebar.header("🔍 Formation Time Filters")
st.sidebar.markdown("Filter by **when** high/low formed within the range")

# High formation filter
high_filter_enabled = st.sidebar.checkbox("Filter High Formation Time", value=False)
if high_filter_enabled:
    available_high_times = sorted(df_range['high_time'].unique(), key=time_to_minutes)
    time_strings = [t.strftime('%H:%M') for t in available_high_times]
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        high_start_idx = st.selectbox(
            "High Start",
            range(len(time_strings)),
            format_func=lambda i: time_strings[i],
            key='high_start'
        )
    with col2:
        high_end_idx = st.selectbox(
            "High End",
            range(len(time_strings)),
            format_func=lambda i: time_strings[i],
            index=len(time_strings)-1,
            key='high_end'
        )
    
    high_start = available_high_times[high_start_idx]
    high_end = available_high_times[high_end_idx]
    
    high_start_min = time_to_minutes(high_start)
    high_end_min = time_to_minutes(high_end)
    df_range = df_range[
        df_range['high_time'].apply(lambda t: high_start_min <= time_to_minutes(t) <= high_end_min)
    ]

# Low formation filter
low_filter_enabled = st.sidebar.checkbox("Filter Low Formation Time", value=False)
if low_filter_enabled:
    available_low_times = sorted(df_range['low_time'].unique(), key=time_to_minutes)
    time_strings = [t.strftime('%H:%M') for t in available_low_times]
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        low_start_idx = st.selectbox(
            "Low Start",
            range(len(time_strings)),
            format_func=lambda i: time_strings[i],
            key='low_start'
        )
    with col2:
        low_end_idx = st.selectbox(
            "Low End",
            range(len(time_strings)),
            format_func=lambda i: time_strings[i],
            index=len(time_strings)-1,
            key='low_end'
        )
    
    low_start = available_low_times[low_start_idx]
    low_end = available_low_times[low_end_idx]
    
    low_start_min = time_to_minutes(low_start)
    low_end_min = time_to_minutes(low_end)
    df_range = df_range[
        df_range['low_time'].apply(lambda t: low_start_min <= time_to_minutes(t) <= low_end_min)
    ]

# Check if we have data
if len(df_range) == 0:
    st.error("No data matches your filters!")
    st.stop()

# Get unique sessions (date + range_obs_time + high/low values uniquely identify a session)
unique_sessions = df_range.groupby(['date', 'range_obs_time', 'high_value', 'low_value']).size().reset_index(name='count')
total_sessions = len(unique_sessions)

# Key Metrics
st.header("📊 Key Metrics")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Sessions", f"{total_sessions:,}")
with col2:
    avg_high = unique_sessions.merge(
        df_range[['date', 'range_obs_time', 'high_value']].drop_duplicates(),
        on=['date', 'range_obs_time', 'high_value']
    )['high_value'].mean()
    st.metric("Avg High", f"{avg_high:.2f}")
with col3:
    avg_low = unique_sessions.merge(
        df_range[['date', 'range_obs_time', 'low_value']].drop_duplicates(),
        on=['date', 'range_obs_time', 'low_value']
    )['low_value'].mean()
    st.metric("Avg Low", f"{avg_low:.2f}")

st.markdown("---")

# Break Probability Over Time
st.header("📈 Break Probability Over Time")
st.markdown(f"**Cumulative break % from {selected_range_time.strftime('%H:%M')} onwards**")

# Get check observation times (future times after range was set)
check_times = sorted(df_range['check_obs_time'].unique(), key=time_to_minutes)

break_prob_data = []
for check_time in check_times:
    df_check = df_range[df_range['check_obs_time'] == check_time]
    
    # Get unique sessions at this check time
    sessions_at_check = df_check.groupby(['date', 'range_obs_time', 'high_value', 'low_value']).first().reset_index()
    
    total = len(sessions_at_check)
    if total > 0:
        high_broken_count = sessions_at_check['high_broken'].sum()
        low_broken_count = sessions_at_check['low_broken'].sum()
        
        high_broken_pct = (high_broken_count / total) * 100
        low_broken_pct = (low_broken_count / total) * 100
        
        break_prob_data.append({
            'check_time': check_time,
            'time_str': check_time.strftime('%H:%M'),
            'high_broken_pct': high_broken_pct,
            'low_broken_pct': low_broken_pct,
            'total_sessions': total
        })

if break_prob_data:
    break_df = pd.DataFrame(break_prob_data)
    
    # Create LINE chart (not filled area)
    fig_breaks = go.Figure()
    
    fig_breaks.add_trace(go.Scatter(
        x=break_df['time_str'],
        y=break_df['high_broken_pct'],
        mode='lines+markers',
        name='High Broken %',
        line=dict(color='#e74c3c', width=3),
        marker=dict(size=8),
        hovertemplate='<b>%{x}</b><br>High Broken: %{y:.1f}%<extra></extra>'
    ))
    
    fig_breaks.add_trace(go.Scatter(
        x=break_df['time_str'],
        y=break_df['low_broken_pct'],
        mode='lines+markers',
        name='Low Broken %',
        line=dict(color='#2ecc71', width=3),
        marker=dict(size=8),
        hovertemplate='<b>%{x}</b><br>Low Broken: %{y:.1f}%<extra></extra>'
    ))
    
    fig_breaks.update_layout(
        title=f"Cumulative Break Probability from {selected_range_time.strftime('%H:%M')}",
        xaxis_title="Check Time",
        yaxis_title="Cumulative Break %",
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
    
    st.plotly_chart(fig_breaks, use_container_width=True)
    
    st.caption(f"📊 Shows cumulative % of ranges that broke by each check time. " +
              f"Range was set at {selected_range_time.strftime('%H:%M')} (16:00 → {selected_range_time.strftime('%H:%M')})")

st.markdown("---")

# Helper function for categorizing breaks
def categorize_first_break(row):
    """Determine which broke first"""
    high_broke = row['high_broken']
    low_broke = row['low_broken']
    high_time = row.get('high_break_time')
    low_time = row.get('low_break_time')
    
    if not high_broke and not low_broke:
        return 'Both Hold'
    elif high_broke and not low_broke:
        return 'High Breaks (Only)'
    elif low_broke and not high_broke:
        return 'Low Breaks (Only)'
    else:  # Both broke
        if pd.notna(high_time) and pd.notna(low_time):
            high_min = time_to_minutes(high_time)
            low_min = time_to_minutes(low_time)
            
            if high_min < low_min:
                return 'High Breaks First'
            elif low_min < high_min:
                return 'Low Breaks First'
            else:
                return 'Both Break Same Time'
        return 'Both Break (Unknown Order)'

# Which Breaks First Analysis - OVER TIME
st.header("2️⃣ Which Extremity Breaks First Over Time")
st.markdown(f"**Track which extremity breaks as time progresses from {selected_range_time.strftime('%H:%M')}**")

# Calculate breakdown at each check time
breakdown_over_time = []

for check_time in check_times:
    df_check = df_range[df_range['check_obs_time'] == check_time]
    sessions_check = df_check.groupby(['date', 'range_obs_time', 'high_value', 'low_value']).first().reset_index()
    
    if len(sessions_check) > 0:
        # Categorize each session
        sessions_check['first_break'] = sessions_check.apply(categorize_first_break, axis=1)
        
        # Count
        total = len(sessions_check)
        high_first = (sessions_check['first_break'] == 'High Breaks First').sum() + (sessions_check['first_break'] == 'High Breaks (Only)').sum()
        low_first = (sessions_check['first_break'] == 'Low Breaks First').sum() + (sessions_check['first_break'] == 'Low Breaks (Only)').sum()
        both_hold = (sessions_check['first_break'] == 'Both Hold').sum()
        
        high_first_pct = (high_first / total) * 100
        low_first_pct = (low_first / total) * 100
        both_hold_pct = (both_hold / total) * 100
        
        breakdown_over_time.append({
            'check_time': check_time,
            'time_str': check_time.strftime('%H:%M'),
            'high_first_pct': high_first_pct,
            'low_first_pct': low_first_pct,
            'both_hold_pct': both_hold_pct,
            'total': total
        })

if breakdown_over_time:
    breakdown_df = pd.DataFrame(breakdown_over_time)
    
    # Create LINE chart (not stacked area)
    fig_breakdown = go.Figure()
    
    # Both Hold line (starts at 100%, decreases)
    fig_breakdown.add_trace(go.Scatter(
        x=breakdown_df['time_str'],
        y=breakdown_df['both_hold_pct'],
        mode='lines+markers',
        name='Both Hold %',
        line=dict(color='#3498db', width=3),
        marker=dict(size=8),
        hovertemplate='<b>%{x}</b><br>Both Hold: %{y:.1f}%<extra></extra>'
    ))
    
    # High Breaks First line (starts at 0%, increases)
    fig_breakdown.add_trace(go.Scatter(
        x=breakdown_df['time_str'],
        y=breakdown_df['high_first_pct'],
        mode='lines+markers',
        name='High Breaks First/Only %',
        line=dict(color='#e74c3c', width=3),
        marker=dict(size=8),
        hovertemplate='<b>%{x}</b><br>High First: %{y:.1f}%<extra></extra>'
    ))
    
    # Low Breaks First line (starts at 0%, increases)
    fig_breakdown.add_trace(go.Scatter(
        x=breakdown_df['time_str'],
        y=breakdown_df['low_first_pct'],
        mode='lines+markers',
        name='Low Breaks First/Only %',
        line=dict(color='#2ecc71', width=3),
        marker=dict(size=8),
        hovertemplate='<b>%{x}</b><br>Low First: %{y:.1f}%<extra></extra>'
    ))
    
    fig_breakdown.update_layout(
        title=f"Which Extremity Breaks First - Evolution from {selected_range_time.strftime('%H:%M')}",
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
    
    st.plotly_chart(fig_breakdown, use_container_width=True)
    
    # Show metrics at final time
    final_row = breakdown_df.iloc[-1]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(f"High Breaks (by {final_row['time_str']})", f"{final_row['high_first_pct']:.1f}%")
    with col2:
        st.metric(f"Low Breaks (by {final_row['time_str']})", f"{final_row['low_first_pct']:.1f}%")
    with col3:
        st.metric(f"Both Hold (at {final_row['time_str']})", f"{final_row['both_hold_pct']:.1f}%")
    
    st.info(f"💡 **Reading the chart:** Blue line starts at 100% (all ranges holding). " +
           f"As time progresses, red line (high breaks) and green line (low breaks) increase. " +
           f"Shows which extremity tends to break first.")
else:
    st.warning("No data for breakdown analysis")

st.markdown("---")

# Get final check time data for formation distributions and table
final_check_time = check_times[-1]
df_final = df_range[df_range['check_obs_time'] == final_check_time]
sessions_final = df_final.groupby(['date', 'range_obs_time', 'high_value', 'low_value']).first().reset_index()

# Formation Time Distributions
st.header("📍 Formation Time Distributions")

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
sample_df = sessions_final[['date', 'high_value', 'high_time', 'low_value', 'low_time', 
                            'high_broken', 'high_break_time', 'low_broken', 'low_break_time', 
                            'first_break']].head(20).copy()
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
