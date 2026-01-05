"""
Module for market regime detection using K-Means clustering.

Note: K-Means is included as a baseline to demonstrate limitations:
no temporal structure, no probabilistic framework, sensitive to outliers.
Expected to show poor regime distribution and extreme risk metrics.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Optional


class MarketRegimeKMeans:
    """
    Class for detecting market regimes with K-Means clustering.
    Baseline comparison model (no temporal structure, no probabilistic framework).
    """
    
    def __init__(
        self,
        n_clusters: int = 3,
        random_state: int = 42,
        max_iter: int = 300,
        n_init: int = 10
    ):
        """Initialize K-Means model for regime detection (baseline comparison)."""
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_init = n_init
        
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            max_iter=max_iter,
            n_init=n_init
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame, returns: Optional[pd.Series] = None, sort_by: str = "volatility") -> 'MarketRegimeKMeans':
        """
        Train K-Means model on features.
        If returns provided, regimes are reorganized by characteristic (0=low, 1=medium, 2=high).
        Returns self.
        """
        # Select numeric columns
        X_numeric = X.select_dtypes(include=[np.number])
        
        # Normalize data
        X_scaled = self.scaler.fit_transform(X_numeric)
        
        # Train the model
        self.model.fit(X_scaled)
        self.is_fitted = True
        
        # Reorganize model parameters if returns provided
        if returns is not None:
            self._reorganize_by_characteristic(X, returns, sort_by)
        
        return self
    
    def _reorganize_by_characteristic(self, X: pd.DataFrame, returns: pd.Series, sort_by: str = "volatility") -> None:
        """Reorganize model parameters so regimes are sorted by characteristic (0=low, 1=medium, 2=high)."""
        # Get predictions to identify which regime has which characteristic
        X_numeric = X.select_dtypes(include=[np.number])
        X_scaled = self.scaler.transform(X_numeric)
        original_regimes = self.model.predict(X_scaled)
        unique_regimes = np.unique(original_regimes)
        
        # Calculate characteristics for each regime
        regime_chars = {}
        for regime in unique_regimes:
            mask = original_regimes == regime
            regime_returns = returns[mask]
            
            if sort_by == "volatility":
                char_value = regime_returns.std()
            elif sort_by == "return":
                char_value = regime_returns.mean()
            elif sort_by == "sharpe":
                char_value = regime_returns.mean() / regime_returns.std() if regime_returns.std() > 0 else 0
            else:
                raise ValueError(f"sort_by must be 'volatility', 'return', or 'sharpe'")
            
            regime_chars[regime] = char_value
        
        # Sort regimes by characteristic: [low, medium, high]
        sorted_regimes = sorted(unique_regimes, key=lambda r: regime_chars[r])
        
        # Ensure all regimes are included in permutation (even if not predicted in training data)
        all_regimes = set(range(self.n_clusters))
        missing_regimes = sorted(all_regimes - set(unique_regimes))
        sorted_regimes = sorted_regimes + missing_regimes
        
        permutation = sorted_regimes
        
        # Reorganize cluster centers
        new_centers = np.zeros_like(self.model.cluster_centers_)
        for new_idx, old_idx in enumerate(permutation):
            new_centers[new_idx] = self.model.cluster_centers_[old_idx]
        self.model.cluster_centers_ = new_centers
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict regimes for new data.
        Returns array of regime labels (0, 1, 2) for each observation.
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        X_numeric = X.select_dtypes(include=[np.number])
        X_scaled = self.scaler.transform(X_numeric)
        return self.model.predict(X_scaled)
    
    def get_regime_statistics(self, X: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
        """Calculate statistics for each detected regime."""
        if not self.is_fitted:
            raise ValueError("Model must be trained")
        
        regimes = self.predict(X)
        
        stats = []
        for regime in range(self.n_clusters):
            mask = regimes == regime
            regime_returns = returns[mask]
            
            if len(regime_returns) > 0:
                stats.append({
                    "regime": regime,
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
    
    def get_inertia(self, X: pd.DataFrame) -> float:
        """Calculate inertia (within-cluster sum of squares)."""
        if not self.is_fitted:
            raise ValueError("Model must be trained")
        
        return self.model.inertia_
    
    def get_silhouette_score(self, X: pd.DataFrame) -> float:
        """Calculate silhouette score (higher is better, range: -1 to 1)."""
        from sklearn.metrics import silhouette_score
        
        if not self.is_fitted:
            raise ValueError("Model must be trained")
        
        X_numeric = X.select_dtypes(include=[np.number])
        X_scaled = self.scaler.transform(X_numeric)
        regimes = self.model.predict(X_scaled)
        
        if len(np.unique(regimes)) < 2:
            return -1.0  # Cannot compute with only one cluster
        
        return silhouette_score(X_scaled, regimes)

