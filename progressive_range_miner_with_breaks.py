#!/usr/bin/env python3
"""
Progressive Range Data Miner with Break Tracking
Analyzes expanding ranges and tracks when they break after being set
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
import argparse
from pathlib import Path


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Progressive range data miner with break tracking'
    )
    parser.add_argument('input_file', type=str, help='Path to 5-minute candle CSV')
    parser.add_argument('-o', '--output', type=str, default='progressive_range_with_breaks.csv',
                       help='Output CSV filename')
    parser.add_argument('--session-start', type=str, default='16:00',
                       help='Session start time (default: 16:00)')
    parser.add_argument('--first-obs', type=str, default='03:00',
                       help='First observation time (default: 03:00)')
    parser.add_argument('--last-obs', type=str, default='09:30',
                       help='Last observation time (default: 09:30)')
    parser.add_argument('--min-candles', type=int, default=50,
                       help='Minimum candles required')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--no-gap-check', action='store_true', 
                       help='Disable data gap checking (use for overnight sessions with normal gaps)')
    parser.add_argument('--max-gap-minutes', type=int, default=90,
                       help='Maximum allowed gap in minutes (default: 90, ignored if --no-gap-check)')
    
    return parser.parse_args()


def load_candle_data(filepath):
    """Load 5-minute candle data"""
    df = pd.read_csv(filepath)
    
    # Find datetime column
    datetime_col = None
    for col in df.columns:
        if any(x in col.lower() for x in ['date', 'time', 'datetime']):
            try:
                df['datetime'] = pd.to_datetime(df[col])
                datetime_col = col
                break
            except:
                continue
    
    if datetime_col is None:
        if 'date' in df.columns and 'time' in df.columns:
            df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str))
        else:
            raise ValueError("Could not identify datetime column")
    
    df['date'] = df['datetime'].dt.date
    df['time'] = df['datetime'].dt.time
    df['day_of_week'] = df['datetime'].dt.dayofweek
    
    # Find high/low columns
    col_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'high' in col_lower and 'high' not in col_mapping:
            col_mapping['high'] = col
        elif 'low' in col_lower and 'low' not in col_mapping:
            col_mapping['low'] = col
    
    if 'high' not in col_mapping or 'low' not in col_mapping:
        raise ValueError("Could not identify 'high' and 'low' columns")
    
    df = df.rename(columns={col_mapping['high']: 'high', col_mapping['low']: 'low'})
    
    return df[['datetime', 'date', 'time', 'high', 'low', 'day_of_week']].copy()


def is_valid_session(session_date, session_df, min_candles, no_gap_check=False, max_gap_minutes=90):
    """Validate session"""
    if session_date.weekday() == 4:  # Friday
        return False, "Weekend (Friday)"
    elif session_date.weekday() == 5:  # Saturday
        return False, "Weekend (Saturday)"
    
    if len(session_df) < min_candles:
        return False, f"Insufficient data ({len(session_df)} candles)"
    
    # Optional gap checking
    if not no_gap_check and len(session_df) > 1:
        time_diffs = session_df['datetime'].diff()
        max_gap = time_diffs.max()
        if max_gap > timedelta(minutes=max_gap_minutes):
            return False, f"Data gap ({max_gap})"
    
    return True, "Valid"


def get_observation_times(first_obs_str, last_obs_str):
    """Generate 30-minute observation intervals"""
    first_obs = datetime.strptime(first_obs_str, '%H:%M').time()
    last_obs = datetime.strptime(last_obs_str, '%H:%M').time()
    
    def time_to_minutes(t):
        minutes = t.hour * 60 + t.minute
        if t.hour < 12:
            minutes += 24 * 60
        return minutes
    
    start_minutes = time_to_minutes(first_obs)
    end_minutes = time_to_minutes(last_obs)
    
    observation_times = []
    current_minutes = start_minutes
    
    while current_minutes <= end_minutes:
        minutes_in_day = current_minutes % (24 * 60)
        observation_times.append(time(minutes_in_day // 60, minutes_in_day % 60))
        current_minutes += 30
    
    return observation_times


def get_session_data(df, session_date, session_start_str, session_end_time):
    """Get data for session from session_start to session_end_time"""
    session_start = datetime.strptime(session_start_str, '%H:%M').time()
    start_datetime = datetime.combine(session_date, session_start)
    
    if session_end_time.hour < 12:
        end_datetime = datetime.combine(session_date + timedelta(days=1), session_end_time)
    else:
        end_datetime = datetime.combine(session_date, session_end_time)
    
    mask = (df['datetime'] >= start_datetime) & (df['datetime'] <= end_datetime)
    return df[mask].copy()


def check_breaks_after(df, session_date, session_start_str, obs_time, high_value, low_value, 
                       observation_times):
    """
    Check if/when high or low breaks AFTER the observation time.
    Returns break info for all future observation times.
    """
    # Get current observation index
    try:
        current_idx = observation_times.index(obs_time)
    except ValueError:
        return []
    
    # Get future observation times
    future_obs_times = observation_times[current_idx + 1:]
    
    break_records = []
    high_broken = False
    low_broken = False
    high_break_time = None
    low_break_time = None
    
    for future_obs in future_obs_times:
        # Get data from AFTER current obs time to future obs time
        current_obs_datetime = datetime.combine(session_date, obs_time)
        if obs_time.hour < 12:
            current_obs_datetime = datetime.combine(session_date + timedelta(days=1), obs_time)
        
        future_obs_datetime = datetime.combine(session_date, future_obs)
        if future_obs.hour < 12:
            future_obs_datetime = datetime.combine(session_date + timedelta(days=1), future_obs)
        
        # Get candles AFTER obs_time up to future_obs
        future_data = df[
            (df['datetime'] > current_obs_datetime) & 
            (df['datetime'] <= future_obs_datetime)
        ]
        
        if not future_data.empty:
            # Check if high was broken (any high > high_value)
            if not high_broken:
                broke_high = future_data[future_data['high'] > high_value]
                if not broke_high.empty:
                    high_broken = True
                    high_break_time = broke_high.iloc[0]['time']
            
            # Check if low was broken (any low < low_value)
            if not low_broken:
                broke_low = future_data[future_data['low'] < low_value]
                if not broke_low.empty:
                    low_broken = True
                    low_break_time = broke_low.iloc[0]['time']
        
        break_records.append({
            'future_obs_time': future_obs,
            'high_broken': high_broken,
            'high_break_time': high_break_time,
            'low_broken': low_broken,
            'low_break_time': low_break_time
        })
    
    return break_records


def analyze_progressive_session_with_breaks(df, session_date, session_start_str, 
                                            observation_times, last_obs_time):
    """Analyze session with break tracking"""
    records = []
    
    # Get full session data (for break checking)
    full_session_df = get_session_data(df, session_date, session_start_str, last_obs_time)
    
    for obs_time in observation_times:
        # Get range data up to this obs time
        range_df = get_session_data(df, session_date, session_start_str, obs_time)
        
        if range_df.empty:
            continue
        
        # Find high/low for this range
        high_value = range_df['high'].max()
        low_value = range_df['low'].min()
        
        high_row = range_df[range_df['high'] == high_value].iloc[0]
        low_row = range_df[range_df['low'] == low_value].iloc[0]
        
        high_time = high_row['time']
        low_time = low_row['time']
        
        # Check breaks after this observation time
        break_info = check_breaks_after(
            full_session_df, session_date, session_start_str, 
            obs_time, high_value, low_value, observation_times
        )
        
        # Create base record
        base_record = {
            'date': session_date,
            'range_obs_time': obs_time,  # When range was "set"
            'high_value': high_value,
            'high_time': high_time,
            'low_value': low_value,
            'low_time': low_time,
            'candles_in_range': len(range_df)
        }
        
        # Add break tracking records for each future observation
        if break_info:
            for break_rec in break_info:
                record = base_record.copy()
                record['check_obs_time'] = break_rec['future_obs_time']  # When we check for breaks
                record['high_broken'] = break_rec['high_broken']
                record['high_break_time'] = break_rec['high_break_time']
                record['low_broken'] = break_rec['low_broken']
                record['low_break_time'] = break_rec['low_break_time']
                records.append(record)
        else:
            # Last observation time - no future checks
            record = base_record.copy()
            record['check_obs_time'] = obs_time
            record['high_broken'] = False
            record['high_break_time'] = None
            record['low_broken'] = False
            record['low_break_time'] = None
            records.append(record)
    
    return records


def process_all_sessions(df, session_start_str, observation_times, min_candles, verbose, 
                        no_gap_check=False, max_gap_minutes=90):
    """Process all sessions"""
    unique_dates = sorted(df['date'].unique())
    last_obs_time = observation_times[-1]
    
    all_records = []
    valid_sessions = 0
    skipped_sessions = 0
    skip_reasons = {}
    
    print(f"Processing {len(unique_dates)} potential sessions...")
    
    for i, session_date in enumerate(unique_dates, 1):
        if i % 50 == 0:
            print(f"  Processed {i}/{len(unique_dates)} ({valid_sessions} valid)")
        
        try:
            session_df = get_session_data(df, session_date, session_start_str, last_obs_time)
            is_valid, reason = is_valid_session(session_date, session_df, min_candles, 
                                               no_gap_check, max_gap_minutes)
            
            if not is_valid:
                skipped_sessions += 1
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                if verbose:
                    print(f"  SKIP {session_date}: {reason}")
                continue
            
            session_records = analyze_progressive_session_with_breaks(
                df, session_date, session_start_str, observation_times, last_obs_time
            )
            
            if session_records:
                all_records.extend(session_records)
                valid_sessions += 1
                if verbose:
                    print(f"  ✓ {session_date}: {len(session_records)} records")
        
        except Exception as e:
            skipped_sessions += 1
            skip_reasons[f"Error: {str(e)}"] = skip_reasons.get(f"Error: {str(e)}", 0) + 1
            if verbose:
                print(f"  ERROR {session_date}: {e}")
            continue
    
    print(f"\nCompleted:")
    print(f"  Valid sessions: {valid_sessions}")
    print(f"  Skipped: {skipped_sessions}")
    
    if skip_reasons:
        print(f"\nSkip reasons:")
        for reason, count in sorted(skip_reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"  {reason}: {count}")
    
    print(f"\nGenerated {len(all_records):,} records")
    
    return pd.DataFrame(all_records)


def main():
    """Main execution"""
    args = parse_arguments()
    
    print("=" * 70)
    print("PROGRESSIVE RANGE MINER WITH BREAK TRACKING")
    print("=" * 70)
    print(f"Input: {args.input_file}")
    print(f"Output: {args.output}")
    print(f"Session: {args.session_start} → {args.last_obs}")
    print(f"Observations: {args.first_obs} to {args.last_obs} (30-min)")
    print("=" * 70)
    
    # Load
    print("\n[1/3] Loading data...")
    try:
        df = load_candle_data(args.input_file)
        print(f"  ✓ Loaded {len(df):,} candles")
        print(f"  ✓ Date range: {df['date'].min()} to {df['date'].max()}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return 1
    
    # Get observation times
    observation_times = get_observation_times(args.first_obs, args.last_obs)
    print(f"\n  Observation times: {len(observation_times)} intervals")
    print(f"  {', '.join([t.strftime('%H:%M') for t in observation_times[:3]])} ... {observation_times[-1].strftime('%H:%M')}")
    
    # Process
    print("\n[2/3] Processing with break tracking...")
    if args.no_gap_check:
        print("  ℹ️  Gap checking DISABLED")
    else:
        print(f"  ℹ️  Gap checking enabled (max gap: {args.max_gap_minutes} min)")
    
    try:
        result_df = process_all_sessions(
            df, args.session_start, observation_times, args.min_candles, args.verbose,
            args.no_gap_check, args.max_gap_minutes
        )
        
        if len(result_df) == 0:
            print("  ✗ No valid sessions!")
            return 1
        
        print(f"  ✓ Generated {len(result_df):,} records")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Save
    print("\n[3/3] Saving...")
    try:
        result_df.to_csv(args.output, index=False)
        print(f"  ✓ Saved to: {args.output}")
        
        print("\n" + "=" * 70)
        print("SAMPLE OUTPUT:")
        print("=" * 70)
        print(result_df.head(10).to_string())
        
        print("\n" + "=" * 70)
        print("SUMMARY:")
        print("=" * 70)
        print(f"Total records: {len(result_df):,}")
        print(f"Unique dates: {result_df['date'].nunique():,}")
        print(f"Date range: {result_df['date'].min()} to {result_df['date'].max()}")
        print("=" * 70)
        print("✓ Complete!")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
