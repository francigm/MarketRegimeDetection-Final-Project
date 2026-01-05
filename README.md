# Market Regime Detection and Risk Analysis using Gaussian Mixture Models

## Project Description

This project aims to detect different market regimes (bull, bear, sideways) using three machine learning models: Gaussian Mixture Models (GMM), Hidden Markov Models (HMM), and K-Means clustering. The project then analyzes the risks associated with each regime and compares the performance of all three models.

**Note on K-Means**: K-Means is included as a baseline comparison model to demonstrate why clustering methods without temporal structure are not well-suited for financial time series regime detection. The results show that K-Means produces unrealistic regime distributions and extreme risk metrics, highlighting the importance of probabilistic models with temporal structure (GMM, HMM) for this type of analysis.

## Research Question

**Which model (Gaussian Mixture Model, Hidden Markov Model, or K-Means clustering) provides more accurate market regime detection and better risk assessment for portfolio management?**

This research question guides our comparative analysis of three models (GMM, HMM, and K-Means), evaluating their performance in:
- Detecting distinct market regimes (bull, bear, sideways markets)
- Providing accurate risk metrics (VaR, CVaR, maximum drawdown) and regime statistics (returns, volatility, Sharpe ratio) for each regime
- Supporting portfolio management decisions through regime-specific risk assessment

**Note on Number of Regimes**: We use 3 regimes by convention for comparability between models, interpretability (bull/bear/sideways), and academic precedent. This choice aligns with common practice in regime detection literature and provides clear, actionable regime classifications.

## Data Choices

### Ticker Selection: ^GSPC (S&P 500 Index)

We use **^GSPC** (S&P 500 Index) instead of SPY (ETF) for the following reasons:
- **Pure market representation**: The index directly reflects market dynamics without ETF effects
- **No premium/discount**: ETFs can trade at premiums or discounts to their Net Asset Value (NAV)
- **No management fees**: Index data is not affected by ETF expense ratios
- **Better for regime detection**: More accurate representation of underlying market regimes

### Time Period: 2005-2025

The selected period (2005-01-01 to 2025-10-31) includes multiple distinct market regimes, allowing us to test the model's ability to detect major regime transitions:

- **Pre-crisis bull market** (2005-2006): Normal market conditions before the financial crisis
- **Crisis build-up** (2007): Early signs of subprime crisis
- **Financial crisis** (2008-2009): Major bear market and market crash (Lehman Brothers, etc.)
- **Post-crisis recovery** (2010-2012): Recovery from 2008 financial crisis
- **Extended bull market** (2013-2019): Long period of sustained market growth
- **COVID-19 crash and recovery** (2020-2021): Pandemic-induced volatility and recovery
- **Inflation and rate hikes** (2022-2023): Monetary policy tightening period
- **Recent period** (2024-2025): Current market conditions

**Rationale**: Including the 2008 financial crisis period is crucial for testing regime detection models, as it represents one of the most significant market regime changes in recent history. This extended period allows us to:
1. Test if models can detect the transition from bull to bear market (2007-2008)
2. Validate detection of crisis regimes vs. normal regimes
3. Compare model performance across different types of market stress (2008 crisis vs. COVID-19)

## Project Structure

```
ProjectData/
├── README.md                # What, why, how
├── PROPOSAL.md              # Project proposal
├── AI_USAGE.md              # AI tools usage disclosure
├── environment.yml          # Dependencies (conda)
├── main.py                  # Entry point
├── .gitignore              # Files to ignore for Git
├── src/                     # Source code
│   ├── data_loader.py      # Load and preprocess
│   ├── gmm_model.py        # GMM model definitions
│   ├── hmm_model.py        # HMM model definitions
│   ├── kmeans_model.py     # K-Means model definitions
│   ├── comparison.py       # Model comparison
│   ├── risk_analysis.py    # Risk metrics
│   └── visualization.py    # Metrics and plots
├── data/                    # Data directory
│   └── raw/                # Original data
├── results/                 # Outputs (generated)
├── notebooks/               # Exploration
│   └── main_analysis.ipynb # Main analysis notebook
```

## Installation

This project uses **Conda** for environment management, which is standard for data science projects and recommended for platforms like Nuvolos.

### Step 1: Create the Conda Environment

```bash
# Create new environment from environment.yml
conda env create -f environment.yml
```

This will create an environment named `market-regime-detection` with Python 3.11 and all required dependencies.

### Step 2: Activate the Environment

```bash
# Activate the environment
conda activate market-regime-detection
```

### Step 3: Verify Installation

```bash
# Check that Python and key packages are installed
python --version  # Should show Python 3.11
python -c "import numpy, pandas, sklearn, hmmlearn; print('All packages installed')"
```

**Note**: The `environment.yml` file includes:
- Core scientific libraries (numpy, pandas, scipy, scikit-learn)
- Visualization libraries (matplotlib, seaborn, plotly)
- Specialized packages (hmmlearn for HMM, yfinance for data, statsmodels)
- Jupyter for notebook support

All dependencies are specified with minimum versions to ensure compatibility.

## Usage

### Running the Main Script

1. **Activate the Conda environment** (if not already active):
   ```bash
   conda activate market-regime-detection
   ```

2. **Run the main script**:
   ```bash
   python main.py
   ```

The script will:
- Load and clean market data (S&P 500 Index, 2005-2025)
- Train GMM, HMM, and K-Means models with 3 regimes (by convention)
- Generate regime detection visualizations
- Calculate risk metrics (VaR, CVaR, maximum drawdown) for each regime
- Save all results and figures in the `results/` folder

**Expected Output:**
After running `python main.py`, you should see:
- Console output showing data loading, cleaning statistics, model training progress, and comparison results
- A `results/` folder containing:
  - **CSV files**: Data cleaning stats, feature engineering stats, regime predictions, model statistics, risk metrics (VaR, CVaR, max drawdown), and comparison summaries
  - **PNG figures**: Regime detection plots for each model, regime statistics, transition matrices (HMM), risk metrics, and model comparison charts

### Running the Notebook

1. **Activate the Conda environment**:
   ```bash
   conda activate market-regime-detection
   ```

2. **Launch Jupyter**:
   ```bash
   jupyter notebook
   ```

3. **Open** `notebooks/main_analysis.ipynb` and run cells sequentially.

## Author

Francisco Gomez