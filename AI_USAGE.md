# AI Tools Usage Documentation

This document explains how AI tools were used as learning aids during the development of this project.

## Tools Used

- **ChatGPT (GPT-5)**: Primary tool for code debugging, library learning, code review, and LaTeX assistance
- **Gemini (Google)**: Used for documentation writing

---

## Examples of Uses

### 1. Learning and Understanding Libraries

**ChatGPT** was used to:
- Understand the `hmmlearn` library API and how to properly implement Hidden Markov Models
- Learn the correct way to calculate BIC and AIC for HMM models (parameter counting)

**Example**: Asked ChatGPT to explain how to count parameters for HMM BIC calculation, which helped me implement the correct formula in `src/hmm_model.py`.

### 2. Debugging Help

**ChatGPT** assisted with:
- Debugging visualization code when plots weren't displaying correctly or had formatting issues
- Resolving errors with feature scaling and standardization in model training

**Example**: When the HMM model was throwing errors during fitting, ChatGPT helped identify that the issue was with data shape after scaling (needed to reshape 1D arrays to 2D), leading to the fix in `src/hmm_model.py`.

### 3. Code Review and Suggestions

**ChatGPT** provided suggestions for:
- Improving code structure and organization in `main.py`
- Refactoring repetitive code in visualization functions
- Optimizing feature engineering functions for better performance

**Example**: ChatGPT suggested using `StandardScaler` from scikit-learn instead of manual normalization, which improved code maintainability.

### 4. Documentation Writing

**Gemini** was used for:
- Writing docstrings for functions in the source code
- Improving the README.md structure and clarity

**Example**: Used Gemini to help improve the README.md structure and ensure clear explanations of the project's methodology.

### 5. Research and Justification

**ChatGPT** helped with:
- Finding academic references for the choice of 3 regimes (Ang & Bekaert 2002, Guidolin & Timmermann 2007)
- Clarifying statistical concepts like BIC vs AIC for model comparison

**Example**: Asked ChatGPT to explain why 3 regimes are commonly used in regime detection studies, which informed the justification in `main.py` and the report.

### 6. LaTeX Assistance

**ChatGPT** assisted with:
- Fixing LaTeX compilation errors
- Formatting tables and figures correctly

**Example**: ChatGPT helped troubleshoot LaTeX compilation errors by explaining common issues with table formatting, figure placement, and ensuring proper package dependencies were included.

---

## Learning Outcomes

Using AI tools helped me:
- Understand complex statistical concepts (BIC/AIC, regime detection theory)
- Learn new libraries more efficiently (`hmmlearn`, `scikit-learn`)
- Write better documentation and maintainable code
- Debug issues faster while understanding the root causes