# FlightRank 2025: 2nd Place Solution
## Aeroclub RecSys Cup - Flight Recommendation Ranking Challenge

**Final Ranking:** 2nd Place  
**Private Leaderboard Score:** 0.54175  
**Best Single Model Score:** 0.53807 (NB0036)

## Competition Overview

The FlightRank 2025 challenge focused on building recommendation systems for flight ranking, where participants needed to predict which flight options users would select from search results. The task involved ranking flight options within each search session based on various factors including price, convenience, user preferences, and company travel policies.

## Solution Approach

The solution employs a sophisticated multi-algorithm ensemble that combines three gradient boosting frameworks (XGBoost, LightGBM, and CatBoost) with an innovative bucket-wise ensemble optimization strategy.

### Key Components

1. **Multi-Algorithm Ensemble (13 Models)**
   - 7 XGBoost models with varying `lambdarank_num_pair_per_sample` values (8, 16, 32, 64, 96, 128)
   - 4 LightGBM models with different hyperparameter configurations  
   - 2 CatBoost models (limited to 5 folds due to computational constraints)

2. **Advanced Feature Engineering (300+ Features)**
   - Cross-validation aware behavioral aggregations without data leakage
   - Segment tier positioning and competitive context analysis
   - User convenience profiles and company travel policy inference
   - Route familiarity and complexity patterns

3. **Bucket-Wise Ensemble Optimization**
   - Groups search sessions by size (10-14, 14-18, ..., up to 3500+ options)
   - Uses Optuna to optimize ensemble weights separately for each bucket
   - Recognizes that ranking behavior differs based on option availability

## Key Technical Innovations

- **Leakage-Free Behavioral Features**: Cross-validation aware aggregation system that creates historical user/company behavior features without temporal leakage
- **Segment Intelligence**: Novel approach to modeling flight complexity through segment tier analysis and position-within-tier rankings
- **Adaptive Ensemble Weighting**: Bucket-wise optimization recognizing that user behavior varies with choice set size
- **Multi-Framework Diversity**: Systematic exploration of different ranking algorithms and hyperparameters

## Performance Results

| Model Type | Count | Cross-Validation | Training Folds |
|------------|-------|------------------|----------------|
| XGBoost | 7 | 10-fold GroupKFold | ~45 min/fold |
| LightGBM | 4 | 10-fold GroupKFold | ~45 min/fold |
| CatBoost | 2 | 5-fold GroupKFold | ~90 min/fold |


## Repository Structure

```
├── NB0034.ipynb          # XGBoost (pairs=8)
├── NB0036.ipynb          # XGBoost (pairs=8, 3500 iters) - Best Single Model
├── NB0037.ipynb          # LightGBM (pairs=8)
├── NB0039.ipynb          # LightGBM (pairs=64)
├── NB0040.ipynb          # XGBoost (pairs=64)
├── NB0041.ipynb          # LightGBM (pairs=96)
├── NB0042.ipynb          # LightGBM (pairs=32)
├── NB0043.ipynb          # XGBoost (pairs=96)
├── NB0045.ipynb          # XGBoost (pairs=16)
├── NB0046.ipynb          # XGBoost (pairs=32)
├── NB0047.ipynb          # CatBoost (pairs=32)
├── NB0049.ipynb          # CatBoost (pairs=128)
├── NB0050.ipynb          # LightGBM (pairs=16)
└── ensemble_optuna_bucketoptim_final.ipynb  # Final ensemble optimization
```

## Setup and Reproduction

### Requirements

```bash
pip install polars pandas numpy scikit-learn
pip install xgboost lightgbm catboost
pip install optuna
```

### Hardware Requirements

- **CPU:** Multi-core processor
- **RAM:** 512GB (recommended for full reproduction)
- **Storage:** ~50GB for intermediate files and models

### Reproduction Steps

1. **Data Preparation**: Ensure `train.parquet` and `test.parquet` are available
2. **Model Training**: Run notebooks NB0034 through NB0050 sequentially
   - Update data paths in each notebook before execution
   - Update save paths for model outputs
   - Each notebook will generate OOF predictions and test predictions
3. **Ensemble Optimization**: Run `ensemble_optuna_bucketoptim_final.ipynb`
   - Optimizes weights across 25 group size buckets
   - Generates final ranked predictions
4. **Final Output**: `ensemble_test_predictions_ranked_pd.csv` (Private LB: 0.54175)
5. **Local download**: To test any other saved prediction file run this -

```python
pred = pd.read_csv('ensemble_test_predictions_ranked.csv')
pred = pred[['Id','ranker_id','rank']]
pred.rename(columns={'rank':'selected'},inplace=True)
display(pred) ### You can do local download like this or can save to csv
```

### Individual Model Results

- **Best Single Model**: NB0036 generates `ensemble_test_predictions_ranked.csv` (Private LB: 0.53807)

### Training Time Estimates

- **Feature Engineering**: ~2 hours per notebook
- **Model Training**: ~45-90 minutes per fold depending on algorithm
- **Ensemble Optimization**: ~3 hours
- **Total**: ~12-15 hours for complete reproduction

## Technical Details

### Cross-Validation Strategy
- **GroupKFold**: Ensures all flights from same search session stay in same fold
- **Leakage Prevention**: Behavioral features calculated only on appropriate training folds
- **Fold-Specific Test Features**: Test set gets different feature values for each fold

### Feature Engineering Highlights
- Segment tier positioning within search sessions
- Cross-validated user preference aggregations
- Company travel policy inference
- Route complexity and familiarity scoring
- Competitive positioning and value gap analysis

### Ensemble Strategy
- 25 different group size buckets (10-14 options, 14-18 options, etc.)
- Optuna optimization with 1500 trials per bucket
- Softmax normalization within each search session
- Final ranking based on blended predictions

## Citation

If you use this code or approach, please cite:

```
FlightRank 2025 - 2nd Place Solution
Aeroclub RecSys Cup: Flight Recommendation Ranking Challenge
https://github.com/Rajneesh-Tiwari/FlightRank-2025-Aeroclub-RecSys-Cup
```

## Acknowledgments

- Aeroclub for organizing the RecSys Cup
- Kaggle community for discussions and insights
- Open source libraries: Polars, XGBoost, LightGBM, CatBoost, Optuna
