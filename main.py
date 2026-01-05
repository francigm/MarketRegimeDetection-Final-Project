"""
Main script for market regime detection and risk analysis.
Compares GMM, HMM, and K-Means models for detecting market regimes (2005-2025).
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Matplotlib configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Import custom modules
from src import (
    load_market_data,
    prepare_features,
    clean_market_data,
    ModelComparison
)
from src.visualization import (
    plot_regime_detection,
    plot_regime_statistics,
    plot_model_comparison,
    plot_transition_matrix,
    plot_risk_metrics
)
from src.comparison import (
    calculate_regime_statistics_from_regimes
)


def main():
    """
    Main entry point for market regime detection and risk analysis.
    Loads data, trains GMM/HMM/K-Means models, compares them, and saves results.
    """
    print("=" * 60)
    print("MARKET REGIME DETECTION AND RISK ANALYSIS")
    print("GMM, HMM, and K-Means Comparison")
    print("=" * 60)
    
    # ========================================================================
    # 1. DATA LOADING AND PREPARATION
    # ========================================================================
    print("\n" + "=" * 60)
    print("1. DATA LOADING AND PREPARATION")
    print("=" * 60)
    
    # Parameters
    # Using ^GSPC (S&P 500 Index) instead of SPY (ETF) for more accurate market representation:
    # - ^GSPC represents the actual market index without ETF effects (premium/discount, fees)
    # - Better for regime detection as it reflects pure market dynamics
    TICKER = "^GSPC"  # S&P 500 Index
    START_DATE = "2005-01-01"  # Pre-financial crisis period  
    END_DATE = "2025-10-31"  # End date: October 31, 2025
    # This period allows testing the model's ability to detect major regime transitions

    # Load data
    print(f"\nLoading data for {TICKER}...")
    market_data_raw = load_market_data(
        ticker=TICKER,
        start_date=START_DATE,
        end_date=END_DATE
    )
    
    print(f"Raw data loaded: {len(market_data_raw)} observations")
    print(f"Period: {market_data_raw.index[0]} to {market_data_raw.index[-1]}")
    
    # Clean data
    # Note: We do NOT remove return outliers as they represent important regime changes
    # (e.g., market crashes, rallies). We only validate data quality (OHLC relationships).
    print("\nCleaning data...")
    market_data, cleaning_stats = clean_market_data(
        market_data_raw,
        fill_missing=True,
        validate_ohlc=True,
        verbose=True
    )
    
    print(f"\nCleaned data: {len(market_data)} observations")
    
    # Remove Volume column (not used in models, and for indices it's just a proxy)
    if 'Volume' in market_data.columns:
        market_data = market_data.drop(columns=['Volume'])
        print("Removed 'Volume' column (not used in analysis)")
    
    print(f"\nData preview: {len(market_data)} rows, {len(market_data.columns)} columns")
    print(f"  Columns: {', '.join(market_data.columns.tolist())}")
    
    # Prepare features
    print("\nPreparing features...")
    features, feature_stats = prepare_features(
        market_data,
        include_volatility=True,
        volatility_window=20,
        verbose=True
    )
    
    # Select features for analysis
    X = features[['returns', 'volatility']].copy()
    returns = features['returns'].copy()
    
    print(f"\nSelected features: returns, volatility")
    print(f"  Final data shape: {X.shape[0]} samples x {X.shape[1]} features")
    
    # ========================================================================
    # 2. MODEL TRAINING
    # ========================================================================
    print("\n" + "=" * 60)
    print("2. MODEL TRAINING")
    print("=" * 60)
    
    # We use 3 regimes by convention for comparability between models,
    # interpretability (bull/bear/sideways), and academic precedent
    N_REGIMES = 3
    
    # Use ModelComparison class to train and compare
    # Regimes are automatically reorganized by volatility (0=low, 1=medium, 2=high) during fit()
    print("\nInitializing model comparison...")
    comparison = ModelComparison(n_components=N_REGIMES, random_state=42)
    comparison.fit(X, returns=returns, sort_by="volatility")
    
    # Get regimes 
    gmm_regimes = comparison.gmm_regimes
    hmm_regimes = comparison.hmm_regimes
    kmeans_regimes = comparison.kmeans_regimes
    
    print(f"Models trained (regimes sorted: 0=Low vol, 1=Medium, 2=High vol)")
    
    # Get transition matrix (already reorganized in HMM model)
    transition_matrix = comparison.hmm_model.get_transition_matrix()
    
    # ========================================================================
    # 3. DETECTED REGIMES ANALYSIS
    # ========================================================================
    print("\n" + "=" * 60)
    print("3. DETECTED REGIMES ANALYSIS")
    print("=" * 60)
    
    # Calculate statistics for each model
    # Regime 0 = Low volatility, Regime 1 = Medium volatility, Regime 2 = High volatility
    gmm_stats = calculate_regime_statistics_from_regimes(gmm_regimes, returns)
    hmm_stats = calculate_regime_statistics_from_regimes(hmm_regimes, returns)
    kmeans_stats = calculate_regime_statistics_from_regimes(kmeans_regimes, returns)
    
    print("\nStatistics calculated for all models")
    print("  (See saved CSV files for detailed statistics)")
    
    # Check for unrealistic K-Means results
    if kmeans_stats['sharpe_ratio'].abs().max() > 10:
        print("\nWARNING: K-Means produced extreme Sharpe ratios (>10 or <-10).")
        print("   This indicates poor regime separation, likely due to:")
        print("   - Lack of temporal structure (regimes change too frequently)")
        print("   - Poor handling of market transitions")
        print("   - Sensitivity to outliers")
    
    if kmeans_stats['percentage'].max() > 75:
        print(f"\nWARNING: K-Means regime distribution is highly imbalanced")
        print(f"   (one regime covers {kmeans_stats['percentage'].max():.1f}% of observations).")
        print("   This suggests the model fails to capture meaningful regime diversity.")
    
    # Visualize detected regimes
    print("\nGenerating visualizations...")
    plot_regime_detection(
        market_data,
        returns,
        gmm_regimes,
        title="Regime Detection - GMM Model",
        save_path='results/gmm_regime_detection.png'
    )
    
    plot_regime_detection(
        market_data,
        returns,
        hmm_regimes,
        title="Regime Detection - HMM Model",
        save_path='results/hmm_regime_detection.png'
    )
    
    plot_regime_detection(
        market_data,
        returns,
        kmeans_regimes,
        title="Regime Detection - K-Means Model",
        save_path='results/kmeans_regime_detection.png'
    )
    
    # Detailed statistics
    plot_regime_statistics(
        gmm_stats,
        title="Statistics by Regime - GMM",
        save_path='results/gmm_statistics.png'
    )
    plot_regime_statistics(
        hmm_stats,
        title="Statistics by Regime - HMM",
        save_path='results/hmm_statistics.png'
    )
    plot_regime_statistics(
        kmeans_stats,
        title="Statistics by Regime - K-Means",
        save_path='results/kmeans_statistics.png'
    )
    
    # ========================================================================
    # 4. MODEL COMPARISON
    # ========================================================================
    print("\n" + "=" * 60)
    print("4. MODEL COMPARISON")
    print("=" * 60)
    
    # Complete model comparison
    print("\n=== COMPLETE MODEL COMPARISON ===")
    comparison_results = comparison.compare_models(X, returns)
    
    # Display results
    gmm_metrics = comparison_results['gmm']['metrics']
    hmm_metrics = comparison_results['hmm']['metrics']
    
    # Determine best model
    best_model, justification = comparison.get_best_model(X, returns)
    summary = comparison_results['summary']
    
    print(f"\nBest model: {best_model} (Score: {justification['score']:.4f})")
    print(f"            HMM: BIC={summary['hmm_bic']:.0f}, Sharpe={summary['hmm_avg_sharpe']:.2f}")
    print(f"            GMM: BIC={summary['gmm_bic']:.0f}, Sharpe={summary['gmm_avg_sharpe']:.2f}")
    print(f"            K-Means: N/A (no BIC), Sharpe={summary['kmeans_avg_sharpe']:.2f}")
    print("  (Detailed metrics in comparison_summary.csv)")
    
    # Visual comparison
    plot_model_comparison(
        gmm_stats,
        hmm_stats,
        kmeans_stats,
        save_path='results/model_comparison.png'
    )
    
    # Calculate and visualize transition matrix
    plot_transition_matrix(
        transition_matrix,
        title="HMM Transition Matrix",
        save_path='results/hmm_transition_matrix.png'
    )
    
    # ========================================================================
    # 5. RISK ANALYSIS
    # ========================================================================
    print("\n" + "=" * 60)
    print("5. RISK ANALYSIS")
    print("=" * 60)
    
    # Risk metrics (already calculated in comparison_results)
    gmm_risk = comparison_results['gmm']['risk_metrics']
    hmm_risk = comparison_results['hmm']['risk_metrics']
    kmeans_risk = comparison_results['kmeans']['risk_metrics']
    
    print("\nRisk metrics calculated for all models")
    if not hmm_risk.empty:
        var_str = ', '.join([f'R{int(row["regime"])}: {row["VaR_95"]*100:.2f}%' for _, row in hmm_risk.iterrows()])
        cvar_str = ', '.join([f'R{int(row["regime"])}: {row["CVaR_95"]*100:.2f}%' for _, row in hmm_risk.iterrows()])
        drawdown_str = ', '.join([f'R{int(row["regime"])}: {row["max_drawdown"]*100:.2f}%' for _, row in hmm_risk.iterrows()])
        print(f"\nHMM VaR 95% by regime: {var_str}")
        print(f"HMM CVaR 95% by regime: {cvar_str}")
        print(f"HMM Max Drawdown by regime: {drawdown_str}")
    else:
        print("\nNo HMM risk metrics available")
    print("  (Detailed metrics for all models saved in CSV files and visualized in PNG figures)")
    
    # Visualize risk metrics
    plot_risk_metrics(
        gmm_risk,
        title="Risk Metrics by Regime - GMM",
        save_path='results/gmm_risk_metrics.png'
    )
    plot_risk_metrics(
        hmm_risk,
        title="Risk Metrics by Regime - HMM",
        save_path='results/hmm_risk_metrics.png'
    )
    plot_risk_metrics(
        kmeans_risk,
        title="Risk Metrics by Regime - K-Means",
        save_path='results/kmeans_risk_metrics.png'
    )
    
    # ========================================================================
    # 6. SAVE RESULTS
    # ========================================================================
    print("\n" + "=" * 60)
    print("6. SAVING RESULTS")
    print("=" * 60)
    
    # Save
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DataFrame with complete results
    # Use X.index to ensure alignment with regime predictions (which are based on X)
    # This ensures all arrays have the same length
    results_summary = pd.DataFrame({
        'Returns': returns.loc[X.index].values,
        'GMM_Regime': gmm_regimes,
        'HMM_Regime': hmm_regimes,
        'KMeans_Regime': kmeans_regimes
    }, index=X.index)
    
    # Save with index=True to preserve dates as index (useful for time series analysis)
    results_summary.to_csv(f'{output_dir}/regime_predictions.csv', index=True)
    gmm_stats.to_csv(f'{output_dir}/gmm_statistics.csv', index=False)
    hmm_stats.to_csv(f'{output_dir}/hmm_statistics.csv', index=False)
    kmeans_stats.to_csv(f'{output_dir}/kmeans_statistics.csv', index=False)
    gmm_risk.to_csv(f'{output_dir}/gmm_risk_metrics.csv', index=False)
    hmm_risk.to_csv(f'{output_dir}/hmm_risk_metrics.csv', index=False)
    kmeans_risk.to_csv(f'{output_dir}/kmeans_risk_metrics.csv', index=False)
    
    # Save comparison summary
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(f'{output_dir}/comparison_summary.csv', index=False)
    
    # Save best model justification
    justification_df = pd.DataFrame([justification])
    justification_df.to_csv(f'{output_dir}/best_model_justification.csv', index=False)
    
    # Save data cleaning statistics
    cleaning_stats_df = pd.DataFrame([cleaning_stats])
    cleaning_stats_df.to_csv(f'{output_dir}/data_cleaning_stats.csv', index=False)
    
    # Save feature engineering statistics
    feature_stats_df = pd.DataFrame([feature_stats])
    feature_stats_df.to_csv(f'{output_dir}/feature_engineering_stats.csv', index=False)
    
    print(f"\nAll results saved in '{output_dir}/'")
    print(f"\nSummary: {best_model} selected (Score: {justification['score']:.4f})")
    print(f"         HMM: BIC={hmm_metrics['BIC']:.0f}, Sharpe={summary['hmm_avg_sharpe']:.2f}")
    print(f"\nOutput: {len([f for f in os.listdir(output_dir) if f.endswith('.csv')])} CSV files + {len([f for f in os.listdir(output_dir) if f.endswith('.png')])} PNG figures")
    print(f"   See '{output_dir}/' for all results")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()