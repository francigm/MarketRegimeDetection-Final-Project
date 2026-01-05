"""
Module for loading and preparing financial market data.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Tuple, Optional, Dict


def load_market_data(
    ticker: str = "^GSPC",
    start_date: str = "2005-01-01",
    end_date: Optional[str] = None,
    interval: str = "1d",
    save_raw: bool = True,
    raw_data_dir: str = "data/raw"
) -> pd.DataFrame:
    """
    Load market data from Yahoo Finance.
    Downloads historical data for ticker and date range, or loads from saved file if exists.
    Returns DataFrame with OHLCV columns (Open, High, Low, Close, Volume).
    """
    import os
    
    # Check if raw data already exists
    if save_raw:
        os.makedirs(raw_data_dir, exist_ok=True)
        end_str = end_date if end_date else 'today'
        filename = f"{raw_data_dir}/{ticker.replace('^', '')}_{start_date}_{end_str}.csv"
        
        if os.path.exists(filename):
            print(f"Loading raw data from {filename}...")
            data = pd.read_csv(filename, index_col=0, parse_dates=True)
            print(f"Loaded {len(data)} observations from saved file")
            return data
    
    # Download from Yahoo Finance
    try:
        print(f"Downloading data from Yahoo Finance for {ticker}...")
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval=interval,
            progress=False
        )
        
        if data.empty:
            raise ValueError(f"No data found for {ticker}")
        
        # Remove multi-level columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        # Save raw data
        if save_raw:
            end_str = end_date if end_date else 'today'
            filename = f"{raw_data_dir}/{ticker.replace('^', '')}_{start_date}_{end_str}.csv"
            data.to_csv(filename)
            print(f"Raw data saved to {filename}")
        
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def clean_market_data(
    data: pd.DataFrame,
    fill_missing: bool = True,
    validate_ohlc: bool = True,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Clean and validate market data.
    Performs OHLC validation, removes duplicates, handles missing values, validates returns.
    Returns cleaned DataFrame and dictionary with cleaning statistics.
    """
    original_shape = data.shape
    cleaning_stats = {
        'original_rows': original_shape[0],
        'original_cols': original_shape[1],
        'missing_values_before': {},
        'missing_values_after': {},
        'invalid_ohlc_removed': 0,
        'duplicate_rows_removed': 0,
        'unrealistic_returns_removed': 0,
        'zero_volume_days': 0,
        'temporal_gaps_detected': 0,
        'descriptive_stats_before': {},
        'descriptive_stats_after': {},
        'final_rows': 0,
        'rows_removed': 0
    }
    
    data_cleaned = data.copy()
    
    # 0. Calculate descriptive statistics BEFORE cleaning
    if 'Close' in data_cleaned.columns:
        returns_before = np.log(data_cleaned['Close'] / data_cleaned['Close'].shift(1)).dropna()
        cleaning_stats['descriptive_stats_before'] = {
            'mean_return': float(returns_before.mean()),
            'std_return': float(returns_before.std()),
            'min_return': float(returns_before.min()),
            'max_return': float(returns_before.max()),
            'median_return': float(returns_before.median())
        }
    
    # 0.1. Check for duplicate dates/rows
    if data_cleaned.index.duplicated().any():
        duplicates = data_cleaned.index.duplicated()
        cleaning_stats['duplicate_rows_removed'] = duplicates.sum()
        data_cleaned = data_cleaned[~duplicates]
        if verbose:
            print(f"Removed {duplicates.sum()} duplicate rows (same date)")
    
    # 0.2. Check temporal continuity (gaps in dates)
    if isinstance(data_cleaned.index, pd.DatetimeIndex):
        # Expected daily frequency (business days only, excluding weekends)
        # freq='B' excludes weekends, so missing_dates are holidays/vacations only
        date_range = pd.date_range(start=data_cleaned.index.min(), 
                                   end=data_cleaned.index.max(), 
                                   freq='B')  # Business days
        missing_dates = date_range.difference(data_cleaned.index)
        cleaning_stats['temporal_gaps_detected'] = len(missing_dates)
        if verbose and len(missing_dates) > 0:
            print(f"Detected {len(missing_dates)} missing trading days (holidays/vacations - weekends already excluded)")
    
    # 0.3. Check for zero volume (suspicious for index data)
    if 'Volume' in data_cleaned.columns:
        zero_volume = (data_cleaned['Volume'] == 0).sum()
        cleaning_stats['zero_volume_days'] = int(zero_volume)
        if verbose and zero_volume > 0:
            print(f"Detected {zero_volume} days with zero volume (flagged)")
    
    # 1. Validate OHLC relationships
    if validate_ohlc:
        required_cols = ['Open', 'High', 'Low', 'Close']
        if all(col in data_cleaned.columns for col in required_cols):
            # High should be >= Open, Close, Low
            invalid_high = (
                (data_cleaned['High'] < data_cleaned['Open']) |
                (data_cleaned['High'] < data_cleaned['Close']) |
                (data_cleaned['High'] < data_cleaned['Low'])
            )
            
            # Low should be <= Open, Close, High
            invalid_low = (
                (data_cleaned['Low'] > data_cleaned['Open']) |
                (data_cleaned['Low'] > data_cleaned['Close']) |
                (data_cleaned['Low'] > data_cleaned['High'])
            )
            
            # Prices should be positive
            negative_prices = (
                (data_cleaned[required_cols] <= 0).any(axis=1)
            )
            
            invalid_rows = invalid_high | invalid_low | negative_prices
            cleaning_stats['invalid_ohlc_removed'] = invalid_rows.sum()
            
            if invalid_rows.any():
                data_cleaned = data_cleaned[~invalid_rows]
                if verbose:
                    print(f"Removed {invalid_rows.sum()} rows with invalid OHLC relationships")
    
    # 2. Check for missing values
    missing_before = data_cleaned.isnull().sum()
    cleaning_stats['missing_values_before'] = missing_before.to_dict()
    
    if fill_missing and missing_before.any():
        # Forward fill for price data, then backward fill
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if col in data_cleaned.columns:
                data_cleaned[col] = data_cleaned[col].ffill().bfill()
        
        # For volume, fill missing values with 0
        if 'Volume' in data_cleaned.columns:
            data_cleaned['Volume'] = data_cleaned['Volume'].fillna(0)
        
        # Fill any remaining missing values with forward fill
        data_cleaned = data_cleaned.ffill().bfill()
        
        missing_after = data_cleaned.isnull().sum()
        cleaning_stats['missing_values_after'] = missing_after.to_dict()
        
        if verbose and missing_before.any():
            print(f"Filled missing values: {missing_before.sum()} -> {missing_after.sum()}")
    
    # 3. Validate returns are realistic (for index, daily returns > 20% are suspicious)
    # This catches obvious data errors (e.g., price doubling overnight)
    if 'Close' in data_cleaned.columns:
        returns = np.log(data_cleaned['Close'] / data_cleaned['Close'].shift(1))
        unrealistic_returns = np.abs(returns) > 0.20  # 20% daily return threshold
        cleaning_stats['unrealistic_returns_removed'] = unrealistic_returns.sum()
        
        if unrealistic_returns.any():
            if verbose:
                print(f"Detected {unrealistic_returns.sum()} days with unrealistic returns (>20%):")
                extreme_dates = data_cleaned.index[unrealistic_returns]
                extreme_vals = returns[unrealistic_returns]
                # Show first 5 examples
                n_to_show = min(5, len(extreme_dates))
                for i in range(n_to_show):
                    date = extreme_dates.iloc[i]
                    val = extreme_vals.iloc[i]
                    print(f"  {date.strftime('%Y-%m-%d')}: {val*100:.2f}%")
                if unrealistic_returns.sum() > 5:
                    print(f"  ... and {unrealistic_returns.sum() - 5} more")
            # Remove unrealistic returns (likely data errors, not regime changes)
            data_cleaned = data_cleaned[~unrealistic_returns]
    
    # Final statistics
    cleaning_stats['final_rows'] = data_cleaned.shape[0]
    cleaning_stats['rows_removed'] = original_shape[0] - data_cleaned.shape[0]
    cleaning_stats['data_loss_percentage'] = (cleaning_stats['rows_removed'] / original_shape[0]) * 100
    
    # Calculate descriptive statistics AFTER cleaning
    if 'Close' in data_cleaned.columns:
        returns_after = np.log(data_cleaned['Close'] / data_cleaned['Close'].shift(1)).dropna()
        cleaning_stats['descriptive_stats_after'] = {
            'mean_return': float(returns_after.mean()),
            'std_return': float(returns_after.std()),
            'min_return': float(returns_after.min()),
            'max_return': float(returns_after.max()),
            'median_return': float(returns_after.median())
        }
    
    if verbose:
        print(f"\nData Cleaning Summary:")
        print(f"  Original rows: {cleaning_stats['original_rows']}")
        print(f"  Final rows: {cleaning_stats['final_rows']}")
        print(f"  Rows removed: {cleaning_stats['rows_removed']} ({cleaning_stats['data_loss_percentage']:.2f}%)")
        
        # Show descriptive statistics comparison
        if cleaning_stats['descriptive_stats_before'] and cleaning_stats['descriptive_stats_after']:
            print(f"\nDescriptive Statistics (Returns):")
            print(f"  Before cleaning: mean={cleaning_stats['descriptive_stats_before']['mean_return']:.6f}, "
                  f"std={cleaning_stats['descriptive_stats_before']['std_return']:.6f}")
            print(f"  After cleaning:  mean={cleaning_stats['descriptive_stats_after']['mean_return']:.6f}, "
                  f"std={cleaning_stats['descriptive_stats_after']['std_return']:.6f}")
    
    return data_cleaned, cleaning_stats


def calculate_returns(data: pd.DataFrame) -> pd.Series:
    """
    Calculate logarithmic returns from price data.
    Returns Series of returns with same index as input data.
    """
    if "Close" not in data.columns:
        raise ValueError("'Close' column is required")
    
    returns = np.log(data["Close"] / data["Close"].shift(1))
    return returns.dropna()


def calculate_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate rolling annualized volatility.
    Computes standard deviation of returns over rolling window, then annualizes (multiplies by sqrt(252)).
    Returns Series with same index as input.
    """
    return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized


def prepare_features(
    data: pd.DataFrame,
    include_volatility: bool = True,
    volatility_window: int = 20,
    verbose: bool = False
) -> Tuple[pd.DataFrame, Dict]:
    """
    Prepare features for regime detection analysis.
    Creates returns and volatility features for regime detection.
    Returns DataFrame with engineered features and dictionary with feature statistics.
    """
    features = pd.DataFrame(index=data.index)
    feature_stats = {}
    
    # 1. Returns (logarithmic returns)
    features["returns"] = calculate_returns(data)
    feature_stats['returns'] = {
        'mean': float(features["returns"].mean()),
        'std': float(features["returns"].std()),
        'min': float(features["returns"].min()),
        'max': float(features["returns"].max())
    }
    
    # 2. Volatility (rolling annualized volatility)
    if include_volatility:
        features["volatility"] = calculate_volatility(
            features["returns"],
            window=volatility_window
        )
        feature_stats['volatility'] = {
            'mean': float(features["volatility"].mean()),
            'std': float(features["volatility"].std()),
            'min': float(features["volatility"].min()),
            'max': float(features["volatility"].max())
        }
    
    # Remove rows with NaN (from rolling calculations)
    features_cleaned = features.dropna()
    
    if verbose:
        print(f"\nFeature Engineering Summary:")
        print(f"  Total features created: {len(features.columns)}")
        print(f"  Final observations: {len(features_cleaned)}")
        print(f"  Features: {', '.join(features.columns.tolist())}")
    
    return features_cleaned, feature_stats
