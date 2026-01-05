"""
Module for risk analysis based on detected market regimes.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


def calculate_var_cvar(
    returns: pd.Series,
    confidence_level: float = 0.05
) -> Dict[str, float]:
    """
    Calculate Value at Risk (VaR) and Conditional VaR (CVaR).
    VaR is maximum expected loss at given confidence level.
    CVaR (Expected Shortfall) is expected loss given that loss exceeds VaR.
    """
    var = returns.quantile(confidence_level)
    cvar = returns[returns <= var].mean()
    
    return {
        "VaR": var,
        "CVaR": cvar,
        "confidence_level": confidence_level
    }


def calculate_regime_risk_metrics(
    returns: pd.Series,
    regimes: np.ndarray
) -> pd.DataFrame:
    """
    Calculate risk metrics for each detected market regime.
    Computes VaR/CVaR at 95% and 99%, and maximum drawdown.
    Returns DataFrame with risk metrics for each regime.
    """
    unique_regimes = np.unique(regimes)
    risk_metrics = []
    
    for regime in unique_regimes:
        mask = regimes == regime
        regime_returns = returns[mask]
        
        if len(regime_returns) > 0:
            # VaR and CVaR
            var_95 = calculate_var_cvar(regime_returns, 0.05)
            var_99 = calculate_var_cvar(regime_returns, 0.01)
            
            # Maximum Drawdown
            cumulative = (1 + regime_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            risk_metrics.append({
                "regime": regime,
                "count": len(regime_returns),
                "mean_return": regime_returns.mean(),
                "volatility": regime_returns.std(),
                "VaR_95": var_95["VaR"],
                "CVaR_95": var_95["CVaR"],
                "VaR_99": var_99["VaR"],
                "CVaR_99": var_99["CVaR"],
                "max_drawdown": max_drawdown
            })
    
    return pd.DataFrame(risk_metrics)
