"""
Module for market regime detection using Gaussian Mixture Models (GMM).
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Dict


class MarketRegimeGMM:
    """
    Class for detecting market regimes with Gaussian Mixture Models (GMM).
    Uses probabilistic clustering to identify distinct market phases.
    """
    
    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = "full",
        random_state: int = 42,
        max_iter: int = 100
    ):
        """Initialize GMM model for regime detection."""
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.max_iter = max_iter
        
        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state,
            max_iter=max_iter
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame, returns: Optional[pd.Series] = None, sort_by: str = "volatility") -> 'MarketRegimeGMM':
        """
        Train GMM model on features.
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
        all_regimes = set(range(self.n_components))
        missing_regimes = sorted(all_regimes - set(unique_regimes))
        sorted_regimes = sorted_regimes + missing_regimes
        
        permutation = sorted_regimes
        
        # Reorganize all model parameters
        # Reorganize means
        new_means = np.zeros_like(self.model.means_)
        for new_idx, old_idx in enumerate(permutation):
            new_means[new_idx] = self.model.means_[old_idx]
        self.model.means_ = new_means
        
        # Reorganize covariances (full, diag, spherical all use same logic)
        if self.covariance_type in ["full", "diag", "spherical"]:
            new_covariances = np.zeros_like(self.model.covariances_)
            for new_idx, old_idx in enumerate(permutation):
                new_covariances[new_idx] = self.model.covariances_[old_idx]
            self.model.covariances_ = new_covariances
        # "tied" doesn't need reorganization (shared covariance)
        
        # Reorganize weights
        new_weights = np.zeros_like(self.model.weights_)
        for new_idx, old_idx in enumerate(permutation):
            new_weights[new_idx] = self.model.weights_[old_idx]
        self.model.weights_ = new_weights
    
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
        for regime in range(self.n_components):
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
        
        return pd.DataFrame(stats)
    
    def get_bic_aic(self, X: pd.DataFrame) -> Dict[str, float]:
        """Calculate BIC and AIC criteria."""
        if not self.is_fitted:
            raise ValueError("Model must be trained")
        
        X_numeric = X.select_dtypes(include=[np.number])
        X_scaled = self.scaler.transform(X_numeric)
        
        return {
            "BIC": self.model.bic(X_scaled),
            "AIC": self.model.aic(X_scaled)
        }
