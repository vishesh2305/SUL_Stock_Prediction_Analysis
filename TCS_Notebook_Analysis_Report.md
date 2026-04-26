# TCS Stock Prediction Notebook - Analysis & Improvement Report

> Reference guide for improving the IT_Notebook_TCS.ipynb model accuracy and methodology.

---

## 1. Notebook Structure

| Phase | Cells | Purpose |
|-------|-------|---------|
| Phase 1: Data Import & Preparation | 0-16 | Load TCS stock data (2015-present, 2759 rows), clean types, create Volume_Ratio |
| Phase 2: Feature Engineering & EDA | 17-58 | Engineer 20+ technical indicators, visualize data, create binary target |
| Phase 3: Model Training & Evaluation | 59-102 | Train 5 models, evaluate with accuracy/confusion matrix/classification report |
| Phase 4: TimeSeriesSplit CV | 103-104 | 5-fold time-series cross-validation comparison |

---

## 2. Current Model Performance (TimeSeriesSplit 5-Fold CV)

| Model | Mean Accuracy | Std | Mean F1 (Down) | Mean F1 (Up) |
|-------|--------------|-----|----------------|--------------|
| Random Forest | 51.92% | 0.031 | 0.456 | 0.567 |
| SVC | 51.86% | 0.033 | 0.254 | 0.640 |
| Logistic Regression | 51.69% | 0.010 | 0.324 | 0.619 |
| Decision Tree | 51.10% | 0.033 | 0.424 | 0.567 |
| XGBoost | 49.94% | 0.024 | 0.448 | 0.541 |

All models hover around 50% - features are the bottleneck.

---

## 3. Technical Errors Fixed

| Cell | Issue | Fix Applied |
|------|-------|-------------|
| 3 | `SyntaxWarning: invalid escape sequence '\I'` | Added raw string prefix `r"..."` |
| 13 | Volume_Ratio rolling window=2 (too noisy) | Changed to `window=20` |
| 18 | Manual loop for daily returns | Replaced with vectorized `pct_change()` |
| 29 | MACD Signal span=12 (incorrect standard) | Changed to `span=9` |
| 30 | Volatility rolling window=3 (too noisy) | Changed to `window=20` |
| 31 | RSI using SMA instead of EMA | Changed to `ewm(span=14, adjust=False)` (Wilder's RSI) |
| 94 | Deprecated `use_label_encoder=False` in XGBoost | Removed the parameter |

---

## 4. Remaining Issues to Address

### A. Feature Selection Problem
- 20+ features engineered but only 4 hand-picked: `Return_Lag1, Volume_Ratio, RSI, RSI_Lag1`
- No systematic feature selection process to justify this choice
- Dropped features include MACD, MACD Histogram, Bollinger Bands, ATR Ratio, Volatility, Day_of_Week - all untested

### B. Target Variable Issues
- Arbitrary 0.3% threshold for classifying up/down days (no data-driven justification)
- Returns between -0.3% and +0.3% are dropped entirely (significant data loss)
- Class imbalance not addressed (no SMOTE, no class weights)

### C. Model Training Issues
- No hyperparameter tuning for 4 of 5 models (only Decision Tree has depth search)
- Decision Tree max_depth=14 is likely overfitting
- Inconsistent scaling (RF uses unscaled, others use scaled)

### D. Evaluation Issues
- ROC/AUC computed for only 2 of 5 models
- No final comparison table or conclusion cell
- No profit-based evaluation (simulated returns)

---

## 5. New Features to Implement

### A. Technical Indicators Not Yet Used

| Feature | Description | Why It Helps |
|---------|-------------|-------------|
| Stochastic Oscillator (%K, %D) | Momentum indicator comparing close to high-low range over 14 days | Captures overbought/oversold differently than RSI |
| Williams %R | Similar to Stochastic but inverted scale (-100 to 0) | Quick momentum reversal signals |
| OBV (On-Balance Volume) | Cumulative volume weighted by price direction | Volume precedes price; captures accumulation/distribution |
| Commodity Channel Index (CCI) | Deviation from statistical mean | Identifies cyclical trends |
| ADX (Average Directional Index) | Trend strength (not direction) | Tells whether market is trending or ranging |
| Ichimoku Cloud components | Multi-component trend system | Support/resistance/momentum/trend in one indicator |
| Price Rate of Change (ROC) | % change over N periods | Different lookback periods capture different momentum |
| Parabolic SAR | Trailing stop-and-reverse indicator | Trend-following signals |

### B. Non-Technical / External Features

| Feature | Description | Why It Helps |
|---------|-------------|-------------|
| Nifty IT Index returns | Sector index daily returns | TCS moves with its sector; sector momentum is predictive |
| Nifty 50 returns | Broad market daily returns | Market-wide risk-on/risk-off affects all stocks |
| INR/USD exchange rate | Currency movement | IT companies earn in USD; FX directly impacts revenue |
| S&P 500 prior day return | US market overnight signal | Indian IT stocks react to US market due to client base |
| VIX India | Volatility index | Fear gauge; high VIX correlates with down days |
| Month/Quarter encoding | Seasonal features | Quarterly results, budget season create patterns |
| Days to earnings | Distance to next earnings date | Stocks behave differently near earnings |
| FII/DII buy/sell data | Institutional flow data | Smart money flow is a strong signal |

### C. Interaction / Derived Features

| Feature | Description |
|---------|-------------|
| RSI x Volume_Ratio | High RSI + high volume = strong conviction signal |
| MACD_Histogram change | Rate of change of MACD histogram (acceleration) |
| Multi-timeframe RSI | RSI on 7-day, 14-day, and 28-day periods |
| Bollinger Band width | Squeeze detection (low width = impending breakout) |
| Distance from 52-week high/low | Psychological support/resistance levels |

### How New Features Improve Accuracy
- **Diversified signal sources**: Current features are mostly price-derived and correlated. Adding volume-based (OBV), external (Nifty, USD), and sentiment (VIX) features introduces orthogonal information.
- **Regime detection**: Features like ADX and Bollinger Band width help the model know when to trust momentum vs. mean-reversion signals.
- **Market context**: External features (S&P 500, VIX, FII flows) provide the "why" behind price moves, which purely technical indicators miss.

---

## 6. Feature Selection Methods (How to Pick the Best Features)

| Method | Type | How It Works | When to Use |
|--------|------|-------------|-------------|
| Mutual Information | Filter | Measures non-linear dependency between feature and target | First pass - quick screening |
| Chi-Squared Test | Filter | Statistical test for feature-target independence | Categorical/discretized features |
| Recursive Feature Elimination (RFE) | Wrapper | Iteratively removes least important features | When you have a preferred model |
| Sequential Feature Selection | Wrapper | Greedily adds/removes features based on CV score | Small-medium feature sets |
| L1 Regularization (Lasso) | Embedded | Logistic Regression with L1 penalty zeros out weak features | Built-in feature selection |
| Tree-based importance | Embedded | Feature importance from RF/XGBoost | Already partially done |
| Boruta Algorithm | Wrapper | Shadow features + RF to find all relevant features | Rigorous feature selection |
| SHAP Values | Model-agnostic | Game-theoretic feature contribution analysis | Best for understanding AND selection |

### Recommended Approach
1. Start with Mutual Information to screen candidates
2. Use SHAP values on best model to understand feature interactions
3. Use RFE with TimeSeriesSplit CV for final selection
4. Test combinations systematically with cross-validation

---

## 7. Techniques for Better Accuracy

### A. Low Complexity (Implement First)

| Technique | Expected Impact |
|-----------|----------------|
| TimeSeriesSplit Cross-Validation | Reliable performance estimates (DONE) |
| GridSearchCV / RandomizedSearchCV | +2-5% accuracy (better hyperparameters for all models) |
| Class weight balancing (`class_weight='balanced'`) | Better recall for minority class |
| LightGBM / CatBoost | Often outperform XGBoost on tabular data |
| Probability Calibration | Better confidence estimates for threshold tuning |
| Threshold Optimization | Find optimal cutoff via precision-recall curve instead of default 0.5 |

### B. Medium Complexity

| Technique | Expected Impact |
|-----------|----------------|
| SMOTE / ADASYN | Synthetic oversampling for imbalanced classes |
| Stacking Ensemble | Combine all 5 models with a meta-learner (+1-3%) |
| Walk-Forward Optimization | Simulate real trading with expanding retrain window |
| Bayesian Hyperparameter Tuning (Optuna) | More efficient than grid search |

### C. High Complexity

| Technique | Expected Impact |
|-----------|----------------|
| LSTM / GRU Neural Networks | Capture temporal sequences (+2-5%) |
| Transformer-based models (Temporal Fusion Transformer) | State-of-art for time-series |

### D. Target Variable Redesign

| Approach | Description |
|----------|-------------|
| Regression instead of classification | Predict actual return value, threshold at inference time |
| 3-class target | Up / Flat / Down with distinct thresholds |
| Adaptive threshold | Use rolling percentiles of returns instead of fixed 0.3% |
| Multi-horizon targets | Predict 1-day, 3-day, 5-day returns simultaneously |

---

## 8. Priority Implementation Order

| # | Action | Expected Impact |
|---|--------|----------------|
| 1 | TimeSeriesSplit cross-validation | Reliable performance estimates (DONE) |
| 2 | Systematic feature selection (use all 20+ features, let algorithms pick) | +3-8% accuracy |
| 3 | Hyperparameter tuning with RandomizedSearchCV | +2-5% accuracy |
| 4 | Add external features (Nifty 50, USD/INR, VIX) | +3-7% accuracy |
| 5 | Try stacking ensemble of best models | +1-3% accuracy |
| 6 | Redesign target variable (regression or adaptive threshold) | Fundamental improvement |
| 7 | Add LSTM/GRU for sequence modeling | +2-5% accuracy |
| 8 | Add profit simulation for practical evaluation | Better model selection |

---

## 9. Code Quality Suggestions

- Remove empty cells (Cells 14-15)
- Use `sklearn.metrics` import once at top instead of re-importing in each evaluation cell
- Use `sklearn.pipeline.Pipeline` to bundle scaling + model for cleaner code
- Add a final comparison table cell with all model results
- Set random seeds consistently across all models
- Pin library versions in a requirements cell at the top
- Add markdown cells explaining why each feature/model was chosen
