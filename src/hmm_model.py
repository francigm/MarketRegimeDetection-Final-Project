"""
Module for market regime detection using Hidden Markov Models (HMM).
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Dict


class MarketRegimeHMM:
    """
    Class for detecting market regimes with Hidden Markov Models (HMM).
    Models temporal dependencies and transition probabilities between regimes.
    """
    
    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = "full",
        n_iter: int = 100,
        random_state: int = 42
    ):
        """Initialize HMM model for regime detection."""
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        
        self.model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame, returns: Optional[pd.Series] = None, sort_by: str = "volatility") -> 'MarketRegimeHMM':
        """
        Train HMM model on features.
        If returns provided, regimes are reorganized by characteristic (0=low, 1=medium, 2=high).
        Returns self.
        """
        # Select numeric columns
        X_numeric = X.select_dtypes(include=[np.number])
        
        # Normalize data
        X_scaled = self.scaler.fit_transform(X_numeric)
        
        # HMM requires 2D array even for single feature
        if X_scaled.ndim == 1:
            X_scaled = X_scaled.reshape(-1, 1)
        
        # Train the model
        # Note: sklearn/hmmlearn assign arbitrary labels (0,1,2) during training.
        # We cannot control this order during learning, so we reorganize after training.
        self.model.fit(X_scaled)
        self.is_fitted = True
        
        # Reorganize model parameters if returns provided
        # This ensures regimes are sorted by characteristic (0=low, 1=medium, 2=high)
        # This reorganization is necessary because the learning algorithm doesn't
        # guarantee any specific order of components/states.
        if returns is not None:
            self._reorganize_by_characteristic(X, returns, sort_by)
        
        return self
    
    def _reorganize_by_characteristic(self, X: pd.DataFrame, returns: pd.Series, sort_by: str = "volatility") -> None:
        """Reorganize model parameters so regimes are sorted by characteristic (0=low, 1=medium, 2=high)."""
        # Get predictions to identify which regime has which characteristic
        X_numeric = X.select_dtypes(include=[np.number])
        X_scaled = self.scaler.transform(X_numeric)
        if X_scaled.ndim == 1:
            X_scaled = X_scaled.reshape(-1, 1)
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
        # Missing regimes will be placed at the end
        all_regimes = set(range(self.n_components))
        missing_regimes = sorted(all_regimes - set(unique_regimes))
        sorted_regimes = sorted_regimes + missing_regimes
        
        # Create permutation: new_index = sorted_regimes.index(old_regime)
        # This tells us: new position i should have old regime sorted_regimes[i]
        permutation = sorted_regimes
        
        # Reorganize all model parameters
        # Reorganize means
        new_means = np.zeros_like(self.model.means_)
        for new_idx, old_idx in enumerate(permutation):
            new_means[new_idx] = self.model.means_[old_idx]
        self.model.means_ = new_means
        
        # Reorganize covariances (full, diag, spherical all use same logic)
        if self.covariance_type in ["full", "diag", "spherical"]:
            new_covars = np.zeros_like(self.model.covars_)
            for new_idx, old_idx in enumerate(permutation):
                new_covars[new_idx] = self.model.covars_[old_idx]
            self.model.covars_ = new_covars
        # "tied" doesn't need reorganization (shared covariance)
        
        # Reorganize transition matrix
        # permutation[i] = old regime index that should be at new position i
        # Example: if permutation = [1, 2, 0], then:
        #   - new position 0 gets old regime 1
        #   - new position 1 gets old regime 2
        #   - new position 2 gets old regime 0
        # For transition matrix: new_transmat[new_i, new_j] should contain
        # the probability of transitioning from old regime permutation[new_i]
        # to old regime permutation[new_j], which is old_transmat[permutation[new_i], permutation[new_j]]
        new_transmat = np.zeros_like(self.model.transmat_)
        for new_i, old_i in enumerate(permutation):
            for new_j, old_j in enumerate(permutation):
                new_transmat[new_i, new_j] = self.model.transmat_[old_i, old_j]
        self.model.transmat_ = new_transmat
        
        # Reorganize initial state distribution
        if hasattr(self.model, 'startprob_'):
            new_startprob = np.zeros_like(self.model.startprob_)
            for new_idx, old_idx in enumerate(permutation):
                new_startprob[new_idx] = self.model.startprob_[old_idx]
            self.model.startprob_ = new_startprob
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict regimes for new data.
        Returns array of regime labels (0, 1, 2) for each observation.
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        X_numeric = X.select_dtypes(include=[np.number])
        X_scaled = self.scaler.transform(X_numeric)
        
        if X_scaled.ndim == 1:
            X_scaled = X_scaled.reshape(-1, 1)
        
        return self.model.predict(X_scaled)
    
    def get_transition_matrix(self) -> np.ndarray:
        """
        Return transition matrix between regimes.
        Square matrix (n_regimes x n_regimes) with probabilities of transitioning from one regime to another.
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained")
        
        return self.model.transmat_
    
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
        """
        Calculate BIC and AIC criteria for the HMM model.
        Used for model selection, penalizing model complexity.
        Returns dictionary with 'BIC', 'AIC', 'log_likelihood', and 'n_params'.
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained")
        
        X_numeric = X.select_dtypes(include=[np.number])
        n_samples = len(X)
        
        X_scaled = self.scaler.transform(X_numeric)
        
        if X_scaled.ndim == 1:
            X_scaled = X_scaled.reshape(-1, 1)
        
        # Calculate log-likelihood
        log_likelihood = self.model.score(X_scaled)
        
        # Calculate number of parameters
        n_components = self.n_components
        n_features = X_scaled.shape[1]
        
        # Transition matrix parameters: n_components * (n_components - 1)
        # (each row sums to 1, so one less parameter per row)
        n_transition_params = n_components * (n_components - 1)
        
        # Initial state distribution: n_components - 1 (sums to 1)
        n_initial_params = n_components - 1
        
        # Emission parameters (means and covariances)
        n_means = n_components * n_features
        
        # Covariance parameters depend on type
        if self.covariance_type == "full":
            # Full covariance: n_components * n_features * (n_features + 1) / 2
            n_cov_params = n_components * n_features * (n_features + 1) // 2
        elif self.covariance_type == "tied":
            # Tied: shared covariance matrix
            n_cov_params = n_features * (n_features + 1) // 2
        elif self.covariance_type == "diag":
            # Diagonal: n_components * n_features
            n_cov_params = n_components * n_features
        elif self.covariance_type == "spherical":
            # Spherical: n_components (one variance per component)
            n_cov_params = n_components
        else:
            n_cov_params = 0
        
        # Total number of parameters
        n_params = n_transition_params + n_initial_params + n_means + n_cov_params
        
        # Calculate AIC and BIC
        aic = -2 * log_likelihood + 2 * n_params
        bic = -2 * log_likelihood + n_params * np.log(n_samples)
        
        return {
            "BIC": bic,
            "AIC": aic,
            "log_likelihood": log_likelihood,
            "n_params": n_params
        }
