"""
Module for result visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
from matplotlib.figure import Figure


def _handle_figure(fig: Figure, save_path: Optional[str] = None) -> Optional[Figure]:
    """Helper function to handle figure saving/display consistently."""
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return None
    return fig

def plot_regime_detection(
    data: pd.DataFrame,
    returns: pd.Series,
    regimes: np.ndarray,
    title: str = "Market Regime Detection",
    save_path: Optional[str] = None
) -> Optional[Figure]:
    """
    Visualize detected market regimes over time.
    Creates two-panel plot: closing prices colored by regime (top), returns colored by regime (bottom).
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Plot 1: Prices with regimes
    ax1 = axes[0]
    prices = data["Close"].loc[returns.index]
    colors = plt.cm.Set3(np.linspace(0, 1, len(np.unique(regimes))))
    
    for regime in np.unique(regimes):
        mask = regimes == regime
        ax1.plot(prices.index[mask], prices[mask], 
                'o', alpha=0.6, label=f'Regime {regime}', 
                color=colors[regime], markersize=3)
    
    ax1.plot(prices.index, prices, 'k-', alpha=0.3, linewidth=0.5)
    ax1.set_ylabel('Closing Price', fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Returns with regimes
    ax2 = axes[1]
    for regime in np.unique(regimes):
        mask = regimes == regime
        ax2.scatter(returns.index[mask], returns[mask], 
                   alpha=0.6, label=f'Regime {regime}', 
                   color=colors[regime], s=10)
    
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_ylabel('Returns', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return _handle_figure(fig, save_path)


def plot_regime_statistics(
    stats_df: pd.DataFrame,
    title: str = "Statistics by Regime",
    save_path: Optional[str] = None
) -> Optional[Figure]:
    """
    Visualize descriptive statistics for each detected regime.
    Creates 2x2 grid showing annualized returns, volatility, Sharpe ratio, and regime distribution.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Annualized returns
    ax1 = axes[0, 0]
    ax1.bar(stats_df['regime'], stats_df['annualized_return'] * 100, 
            color=plt.cm.Set3(np.linspace(0, 1, len(stats_df))))
    ax1.set_xlabel('Regime')
    ax1.set_ylabel('Annualized Return (%)')
    ax1.set_title('Returns by Regime')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Annualized volatility
    ax2 = axes[0, 1]
    ax2.bar(stats_df['regime'], stats_df['annualized_volatility'] * 100,
            color=plt.cm.Set3(np.linspace(0, 1, len(stats_df))))
    ax2.set_xlabel('Regime')
    ax2.set_ylabel('Annualized Volatility (%)')
    ax2.set_title('Volatility by Regime')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Sharpe Ratio
    ax3 = axes[1, 0]
    ax3.bar(stats_df['regime'], stats_df['sharpe_ratio'],
            color=plt.cm.Set3(np.linspace(0, 1, len(stats_df))))
    ax3.set_xlabel('Regime')
    ax3.set_ylabel('Sharpe Ratio')
    ax3.set_title('Sharpe Ratio by Regime')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Temporal distribution
    ax4 = axes[1, 1]
    ax4.pie(stats_df['percentage'], labels=[f'Regime {r}' for r in stats_df['regime']],
            autopct='%1.1f%%', startangle=90)
    ax4.set_title('Temporal Distribution of Regimes')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    return _handle_figure(fig, save_path)


def plot_model_comparison(
    gmm_stats: pd.DataFrame,
    hmm_stats: pd.DataFrame,
    kmeans_stats: Optional[pd.DataFrame] = None,
    save_path: Optional[str] = None
) -> Optional[Figure]:
    """
    Compare regime statistics across GMM, HMM, and K-Means models side-by-side.
    Creates 2x2 grid comparing returns, volatility, Sharpe ratio, and temporal distribution.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Determine width based on number of models
    n_models = 3 if kmeans_stats is not None else 2
    x = np.arange(len(gmm_stats))
    width = 0.25 if n_models == 3 else 0.35
    
    # 1. Compared returns
    ax1 = axes[0, 0]
    ax1.bar(x - width, gmm_stats['annualized_return'] * 100, 
            width, label='GMM', alpha=0.8)
    ax1.bar(x, hmm_stats['annualized_return'] * 100, 
            width, label='HMM', alpha=0.8)
    if kmeans_stats is not None:
        ax1.bar(x + width, kmeans_stats['annualized_return'] * 100, 
                width, label='K-Means', alpha=0.8)
    ax1.set_xlabel('Regime')
    ax1.set_ylabel('Annualized Return (%)')
    ax1.set_title('Returns: Model Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Regime {r}' for r in gmm_stats['regime']])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Compared volatility
    ax2 = axes[0, 1]
    ax2.bar(x - width, gmm_stats['annualized_volatility'] * 100, 
            width, label='GMM', alpha=0.8)
    ax2.bar(x, hmm_stats['annualized_volatility'] * 100, 
            width, label='HMM', alpha=0.8)
    if kmeans_stats is not None:
        ax2.bar(x + width, kmeans_stats['annualized_volatility'] * 100, 
                width, label='K-Means', alpha=0.8)
    ax2.set_xlabel('Regime')
    ax2.set_ylabel('Annualized Volatility (%)')
    ax2.set_title('Volatility: Model Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Regime {r}' for r in gmm_stats['regime']])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Compared Sharpe Ratio
    ax3 = axes[1, 0]
    ax3.bar(x - width, gmm_stats['sharpe_ratio'], 
            width, label='GMM', alpha=0.8)
    ax3.bar(x, hmm_stats['sharpe_ratio'], 
            width, label='HMM', alpha=0.8)
    if kmeans_stats is not None:
        ax3.bar(x + width, kmeans_stats['sharpe_ratio'], 
                width, label='K-Means', alpha=0.8)
    ax3.set_xlabel('Regime')
    ax3.set_ylabel('Sharpe Ratio')
    ax3.set_title('Sharpe Ratio: Model Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'Regime {r}' for r in gmm_stats['regime']])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Temporal distribution
    ax4 = axes[1, 1]
    ax4.bar(x - width, gmm_stats['percentage'], 
            width, label='GMM', alpha=0.8)
    ax4.bar(x, hmm_stats['percentage'], 
            width, label='HMM', alpha=0.8)
    if kmeans_stats is not None:
        ax4.bar(x + width, kmeans_stats['percentage'], 
                width, label='K-Means', alpha=0.8)
    ax4.set_xlabel('Regime')
    ax4.set_ylabel('Percentage of Time (%)')
    ax4.set_title('Temporal Distribution: Model Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'Regime {r}' for r in gmm_stats['regime']])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    title = 'GMM, HMM, and K-Means Models Comparison' if kmeans_stats is not None else 'GMM and HMM Models Comparison'
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return _handle_figure(fig, save_path)


def plot_transition_matrix(
    transition_matrix: np.ndarray,
    title: str = "HMM Transition Matrix",
    save_path: Optional[str] = None
) -> Optional[Figure]:
    """
    Visualize HMM transition matrix as a heatmap.
    Displays transition probabilities between regimes as color-coded matrix.
    Diagonal shows regime persistence, off-diagonal shows transition probabilities.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(transition_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    
    # Add values in cells with adaptive text color
    n_regimes = transition_matrix.shape[0]
    for i in range(n_regimes):
        for j in range(n_regimes):
            value = transition_matrix[i, j]
            # Use white text for dark backgrounds (high values), black for light backgrounds
            text_color = 'white' if value > 0.5 else 'black'
            ax.text(j, i, f'{value:.3f}',
                   ha="center", va="center", color=text_color, 
                   fontweight='bold', fontsize=12)
    
    ax.set_xticks(np.arange(n_regimes))
    ax.set_yticks(np.arange(n_regimes))
    ax.set_xticklabels([f'Regime {i}' for i in range(n_regimes)])
    ax.set_yticklabels([f'Regime {i}' for i in range(n_regimes)])
    ax.set_xlabel('Next Regime')
    ax.set_ylabel('Current Regime')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Probability')
    plt.tight_layout()
    return _handle_figure(fig, save_path)


def plot_risk_metrics(
    risk_df: pd.DataFrame,
    title: str = "Risk Metrics by Regime",
    save_path: Optional[str] = None
) -> Optional[Figure]:
    """
    Visualize risk metrics for each detected regime.
    Creates two-panel plot showing VaR 95% and CVaR 95% (Conditional VaR) by regime.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. VaR 95%
    ax1 = axes[0]
    ax1.bar(risk_df['regime'], risk_df['VaR_95'] * 100,
            color=plt.cm.Reds(np.linspace(0.3, 0.9, len(risk_df))))
    ax1.set_xlabel('Regime')
    ax1.set_ylabel('VaR 95% (%)')
    ax1.set_title('Value at Risk (95%)')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. CVaR 95%
    ax2 = axes[1]
    ax2.bar(risk_df['regime'], risk_df['CVaR_95'] * 100,
            color=plt.cm.Reds(np.linspace(0.3, 0.9, len(risk_df))))
    ax2.set_xlabel('Regime')
    ax2.set_ylabel('CVaR 95% (%)')
    ax2.set_title('Conditional VaR (95%)')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return _handle_figure(fig, save_path)
