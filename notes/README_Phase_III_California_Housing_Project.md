# California Housing Prices Project -- Phase III Notes

## Objective

Build a complete preprocessing pipeline and train a Feedforward Neural
Network for a regression task using the California Housing dataset.

## Topics Covered

### 1. One-Hot Encoding

-   Why categorical variables cannot be fed directly to neural networks.
-   `OneHotEncoder` converts categories into binary vectors.
-   Fit **only** on training data.
-   Transform validation and test data with the same encoder.
-   Encoder returns a **SciPy sparse matrix** by default.
-   Sparse matrices store only non-zero values, making storage and
    computation efficient.
-   Convert to dense using:

``` python
encoded = encoded.toarray()
```

-   Convert to DataFrame:

``` python
pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out(),
    index=x_train.index
)
```

### 2. DataFrame vs Series

-   `x[['column']]` returns a DataFrame (2D).
-   `x['column']` returns a Series (1D).
-   Scikit-learn transformers expect 2D input.

### 3. Sparse Matrix Issue

-   Shape looked correct but DataFrame creation failed because sparse
    matrices are treated differently.
-   Convert to dense NumPy arrays before creating DataFrames.
-   A later feature-name warning in Google Colab was traced to stale
    kernel state. Restarting the runtime resolved it.

### 4. Standardization

Used `StandardScaler`.

Why? - Neural networks train faster when numerical features have similar
scales. - Prevents large-valued features from dominating gradients.

Pipeline: 1. `fit()` only on training numerical features. 2.
`transform()` training, validation and test sets.

Do **not** standardize one-hot encoded columns because they already
contain only 0s and 1s.

### 5. Final Preprocessing Pipeline

1.  Train/Validation/Test split.
2.  Separate numerical and categorical columns.
3.  Median imputation.
4.  One-Hot Encoding.
5.  Standardization.
6.  Concatenate processed numerical + encoded categorical features.

------------------------------------------------------------------------

# Neural Network Architecture

Input: - 8 standardized numerical features - 5 one-hot encoded
categorical features

Total: - 13 input features

Architecture:

``` text
Input (13)
    ↓
Dense(64, ReLU)
    ↓
Dense(32, ReLU)
    ↓
Dense(1)
```

Output layer: - One neuron. - Linear activation (default).

Reason: Regression predicts continuous numeric values.

------------------------------------------------------------------------

# Labels

Decision: Do **not** normalize `median_house_value`.

Reason: - Directly predicts actual house prices. - Easier
interpretation. - No inverse transformation needed.

Trade-off: - MSE values become numerically very large.

------------------------------------------------------------------------

# Compilation

Optimizer: - Adam

Loss: - Mean Squared Error (MSE)

Metric: - Mean Absolute Error (MAE)

------------------------------------------------------------------------

# EarlyStopping

Used: - monitor = `val_loss` - patience = 10 - restore_best_weights =
True

Understanding: - Stops only after validation loss stops improving. -
Default `min_delta = 0`. - `min_delta` represents the minimum
improvement required to reset patience.

------------------------------------------------------------------------

# Understanding Regression Metrics

## MAE

Average absolute prediction error.

Interpretation: "If MAE is \$47,000, predictions are off by roughly
\$47k on average."

## MSE

Average squared error.

Purpose: - Optimization. - Penalizes large mistakes much more heavily.

Large values are expected because errors are squared.

## RMSE

Square root of MSE. Returns the error in the original unit (dollars).

------------------------------------------------------------------------

# Training Results

Initial (20 epochs): - Train MAE ≈ 49.5k - Validation MAE ≈ 50.5k - Test
MAE ≈ 49.9k

Extended Training (\~100 epochs): - Train MAE ≈ 46.3k - Validation MAE ≈
48.0k - Test MAE ≈ 47.3k

Observation: - Longer training improved generalization. - No meaningful
overfitting. - Train/Validation/Test metrics remained close.

------------------------------------------------------------------------

# Loss Curve Interpretation

Healthy learning characteristics: - Rapid decrease during early
epochs. - Gradual flattening afterwards. - Training and validation
curves decrease together.

No signs of: - Divergence - Oscillation - Severe overfitting

------------------------------------------------------------------------

# Plotting

Plot both: 1. Training & Validation Loss 2. Training & Validation MAE

Avoid markers for every epoch. Prefer smooth lines or `markevery` for
cleaner visualization.

------------------------------------------------------------------------

# Lessons Learned

-   Always fit preprocessing objects only on training data.
-   Sparse matrices are memory-efficient but sometimes need conversion
    to dense arrays.
-   DataFrame indices must align with transformed outputs.
-   Neural networks for regression use a linear output neuron.
-   Regression evaluation focuses on prediction error, not accuracy.
-   Baseline models are valuable---future experiments should be compared
    against them scientifically.

------------------------------------------------------------------------

# Future Experiments

-   Increase/decrease hidden neurons.
-   Add more hidden layers.
-   Experiment with target scaling.
-   Try Huber Loss.
-   Add L2 regularization or Dropout.
-   Tune learning rate.
-   Compare optimizers.
-   Evaluate RMSE and R².
