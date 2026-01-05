"""
Main module for comparing GMM, HMM, and K-Means models for market regime detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

from .gmm_model import MarketRegimeGMM
from .hmm_model import MarketRegimeHMM
from .kmeans_model import MarketRegimeKMeans
from .risk_analysis import calculate_regime_risk_metrics


def calculate_regime_statistics_from_regimes(
    regimes: np.ndarray,
    returns: pd.Series
) -> pd.DataFrame:
    """
    Calculate descriptive statistics for each detected regime.
    Computes distribution, returns, volatility, Sharpe ratio, and extreme values per regime.
    Returns DataFrame with statistics for each regime.
    """
    stats = []
    unique_regimes = np.unique(regimes)
    
    for regime in unique_regimes:
        mask = regimes == regime
        regime_returns = returns[mask]
        
        if len(regime_returns) > 0:
            stats.append({
                "regime": int(regime),
                "count": len(regime_returns),
                "percentage": len(regime_returns) / len(returns) * 100,
                "mean_return": regime_returns.mean(),
                "std_return": regime_returns.std(),
                "annualized_return": regime_returns.mean() * 252,
                "annualized_volatility": regime_returns.std() * np.sqrt(252),
                "sharpe_ratio": (regime_returns.mean() * 252) / (regime_returns.std() * np.sqrt(252)) if regime_returns.std() > 0 else 0,
                "min_return": regime_returns.min(),
                "max_return": regime_returns.max()
            })
    
    return pd.DataFrame(stats).sort_values('regime')


class ModelComparison:
    """
    Class for comparing GMM, HMM, and K-Means models for market regime detection.
    Provides unified interface to train, compare, and evaluate three different models.
    """
    
    def __init__(self, n_components: int = 3, random_state: int = 42):
        """
        Initialize ModelComparison with specified number of regimes.
        Typically uses 3 regimes (low, medium, high volatility).
        """
        self.n_components = n_components
        self.random_state = random_state
        
        self.gmm_model = None
        self.hmm_model = None
        self.kmeans_model = None
        self.gmm_regimes = None
        self.hmm_regimes = None
        self.kmeans_regimes = None
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame, returns: Optional[pd.Series] = None, sort_by: str = "volatility") -> 'ModelComparison':
        """
        Train all three models (GMM, HMM, K-Means) on features.
        If returns provided, regimes are sorted by characteristic (0=low, 1=medium, 2=high).
        Returns self.
        """
        # Train GMM
        self.gmm_model = MarketRegimeGMM(
            n_components=self.n_components,
            covariance_type='full',
            random_state=self.random_state
        )
        self.gmm_model.fit(X, returns=returns, sort_by=sort_by)
        self.gmm_regimes = self.gmm_model.predict(X)
        
        # Train HMM
        self.hmm_model = MarketRegimeHMM(
            n_components=self.n_components,
            covariance_type='full',
            random_state=self.random_state
        )
        self.hmm_model.fit(X, returns=returns, sort_by=sort_by)
        self.hmm_regimes = self.hmm_model.predict(X)
        
        # Train K-Means
        self.kmeans_model = MarketRegimeKMeans(
            n_clusters=self.n_components,
            random_state=self.random_state
        )
        self.kmeans_model.fit(X, returns=returns, sort_by=sort_by)
        self.kmeans_regimes = self.kmeans_model.predict(X)
        
        self.is_fitted = True
        return self
    
    def compare_models(self, X: pd.DataFrame, returns: pd.Series) -> Dict:
        """
        Compare all three models and compute comprehensive metrics.
        Returns nested dictionary with performance metrics, regime statistics, risk metrics,
        and transition matrices for each model.
        """
        if not self.is_fitted:
            raise ValueError("Models must be trained before comparison")
        
        # GMM Statistics
        gmm_stats = self.gmm_model.get_regime_statistics(X, returns)
        gmm_metrics = self.gmm_model.get_bic_aic(X)
        gmm_risk = calculate_regime_risk_metrics(returns, self.gmm_regimes)
        
        # HMM Statistics
        hmm_stats = self.hmm_model.get_regime_statistics(X, returns)
        hmm_metrics = self.hmm_model.get_bic_aic(X)
        hmm_risk = calculate_regime_risk_metrics(returns, self.hmm_regimes)
        transition_matrix = self.hmm_model.get_transition_matrix()
        
        # K-Means Statistics
        kmeans_stats = self.kmeans_model.get_regime_statistics(X, returns)
        kmeans_inertia = self.kmeans_model.get_inertia(X)
        kmeans_silhouette = self.kmeans_model.get_silhouette_score(X)
        kmeans_risk = calculate_regime_risk_metrics(returns, self.kmeans_regimes)
        
        comparison = {
            'gmm': {
                'statistics': gmm_stats,
                'metrics': gmm_metrics,
                'risk_metrics': gmm_risk,
                'regimes': self.gmm_regimes
            },
            'hmm': {
                'statistics': hmm_stats,
                'metrics': hmm_metrics,
                'risk_metrics': hmm_risk,
                'regimes': self.hmm_regimes,
                'transition_matrix': transition_matrix
            },
            'kmeans': {
                'statistics': kmeans_stats,
                'metrics': {
                    'inertia': kmeans_inertia,
                    'silhouette_score': kmeans_silhouette
                },
                'risk_metrics': kmeans_risk,
                'regimes': self.kmeans_regimes
            },
            'summary': self._create_summary(
                gmm_stats, hmm_stats, kmeans_stats, gmm_metrics, hmm_metrics
            )
        }
        
        return comparison
    
    def _create_summary(
        self,
        gmm_stats: pd.DataFrame,
        hmm_stats: pd.DataFrame,
        kmeans_stats: pd.DataFrame,
        gmm_metrics: Dict,
        hmm_metrics: Dict
    ) -> Dict:
        """Create comparison summary for all three models."""
        return {
            'n_regimes': self.n_components,
            'gmm_bic': gmm_metrics['BIC'],
            'gmm_aic': gmm_metrics['AIC'],
            'hmm_bic': hmm_metrics['BIC'],
            'hmm_aic': hmm_metrics['AIC'],
            'hmm_log_likelihood': hmm_metrics['log_likelihood'],
            'gmm_regime_distribution': {
                int(row['regime']): row['percentage'] 
                for _, row in gmm_stats.iterrows()
            },
            'hmm_regime_distribution': {
                int(row['regime']): row['percentage'] 
                for _, row in hmm_stats.iterrows()
            },
            'kmeans_regime_distribution': {
                int(row['regime']): row['percentage'] 
                for _, row in kmeans_stats.iterrows()
            },
            'gmm_avg_sharpe': gmm_stats['sharpe_ratio'].mean(),
            'hmm_avg_sharpe': hmm_stats['sharpe_ratio'].mean(),
            'kmeans_avg_sharpe': kmeans_stats['sharpe_ratio'].mean()
        }
    
    def get_best_model(self, X: pd.DataFrame, returns: pd.Series) -> Tuple[str, Dict]:
        """
        Determine the best model based on composite score.
        Evaluates models using Sharpe ratio (40%), model quality metrics (30%), and volatility (30%).
        Returns model name and justification dictionary.
        """
        if not self.is_fitted:
            raise ValueError("Models must be trained")
        
        comparison = self.compare_models(X, returns)
        
        gmm_stats = comparison['gmm']['statistics']
        hmm_stats = comparison['hmm']['statistics']
        kmeans_stats = comparison['kmeans']['statistics']
        
        # Comparison criteria
        criteria = {
            'gmm': {
                'avg_sharpe': gmm_stats['sharpe_ratio'].mean(),
                'bic': comparison['gmm']['metrics']['BIC'],
                'avg_volatility': gmm_stats['annualized_volatility'].mean()
            },
            'hmm': {
                'avg_sharpe': hmm_stats['sharpe_ratio'].mean(),
                'bic': comparison['hmm']['metrics']['BIC'],
                'avg_volatility': hmm_stats['annualized_volatility'].mean()
            },
            'kmeans': {
                'avg_sharpe': kmeans_stats['sharpe_ratio'].mean(),
                'silhouette': comparison['kmeans']['metrics']['silhouette_score'],
                'inertia': comparison['kmeans']['metrics']['inertia'],
                'avg_volatility': kmeans_stats['annualized_volatility'].mean()
            }
        }
        
        # Composite score (higher score = better model)
        # For GMM and HMM: use BIC
        gmm_score = (
            criteria['gmm']['avg_sharpe'] * 0.4 +
            (1 / (1 + abs(criteria['gmm']['bic']) / 10000)) * 0.3 +
            (1 / (1 + criteria['gmm']['avg_volatility'])) * 0.3
        )
        
        hmm_score = (
            criteria['hmm']['avg_sharpe'] * 0.4 +
            (1 / (1 + abs(criteria['hmm']['bic']) / 10000)) * 0.3 +
            (1 / (1 + criteria['hmm']['avg_volatility'])) * 0.3
        )
        
        # For K-Means: use silhouette score (higher is better) and inertia (lower is better)
        kmeans_score = (
            criteria['kmeans']['avg_sharpe'] * 0.4 +
            (criteria['kmeans']['silhouette'] + 1) / 2 * 0.3 +  # Normalize to 0-1
            (1 / (1 + criteria['kmeans']['inertia'] / 1000)) * 0.3
        )
        
        # Find best model
        scores = {
            'GMM': gmm_score,
            'HMM': hmm_score,
            'K-Means': kmeans_score
        }
        best_model = max(scores, key=scores.get)
        
        justification = {
            'model': best_model,
            'score': scores[best_model],
            'criteria': criteria[best_model.lower().replace('-', '')],
            'reason': f'Best composite score based on Sharpe, model quality metrics, and volatility',
            'all_scores': scores
        }
        
        return best_model, justification


