# California Housing Project -- Phase II Notes

## Topics Covered

-   Why split into Train/Validation/Test before preprocessing.
-   Chosen split: **80% Train, 10% Validation, 10% Test**.
-   Stratification discussion:
    -   Random split is acceptable for this dataset because it is large
        (\~20k samples).
    -   Stratification is mainly to preserve representative
        distributions.
    -   `median_income` is commonly used because it strongly correlates
        with house prices.
    -   Stratifying directly on continuous targets is not supported
        without binning.
    -   Decision for this project: **use a simple random split**.

## Shuffling

Two different shuffles: 1. **Dataset shuffle** (before splitting) -
Produces representative train/validation/test sets. 2. **Training
shuffle** (before each epoch) - Randomizes mini-batches. - Helps
gradient descent by making each batch a better approximation of the
whole dataset. - In true Batch Gradient Descent, order does not
matter. - In Mini-batch Gradient Descent, order changes the optimization
path because weights change after every batch.

## Preprocessing Pipeline

Load Dataset → EDA → Split Data → Separate Features and Labels →
Separate Numeric and Categorical Features → Numeric Pipeline: - Median
Imputation - Feature Scaling → Categorical Pipeline: - Imputation if
required - Encoding → Combine Features → Model Training

## Separating Features and Labels

Correct: - X = all input features - y = `median_house_value`

Do **not** remove `total_bedrooms`. Remove the target column
(`median_house_value`) from X.

Leaving the target inside X causes **target leakage**.

## Why Separate Numeric and Categorical Columns?

Median exists only for numeric values.

Numeric: - longitude - latitude - housing_median_age - total_rooms -
total_bedrooms - population - households - median_income

Categorical: - ocean_proximity

## Median Imputation

Steps: 1. Create `SimpleImputer(strategy="median")` 2. `fit()` only on
training numeric data. 3. `transform()` training, validation and test
sets. 4. Convert returned NumPy arrays back into DataFrames if
convenient.

### Internal Working

-   `fit()` computes the median of every numeric column and stores it.
-   `transform()` replaces missing values using those stored medians.

### Golden Rule

Anything that **learns from data** uses: - `fit()` on the training set
only.

Everything else uses: - `transform()`.

Applies to: - SimpleImputer - StandardScaler - MinMaxScaler -
OneHotEncoder - PCA - Feature selection methods

## Data Leakage

Data leakage is broader than model training.

It occurs whenever information from validation/test data influences any
preprocessing or model-building step before evaluation.

Examples: - Fitting an imputer on validation data. - Fitting a scaler on
test data. - Fitting an encoder on validation data.

## Pandas: \[\] vs \[\[\]\]

Single brackets:

``` python
df["column"]
```

Returns a Series (1D).

Double brackets:

``` python
df[["column"]]
```

Returns a DataFrame (2D).

Use double brackets for feature columns because most Scikit-Learn
transformers expect 2D input.

## Label vs Features

Convention: - X → Features (2D) - y → Target (1D)

## Code Review Findings

### Correct

-   Fit imputer only on training data.
-   Transform validation/test using the same imputer.
-   Verified missing values after imputation.
-   Converted transformed NumPy arrays back into DataFrames.
-   Clear explanatory comments.

### Mistakes Found

1.  Dropped `total_bedrooms` instead of `median_house_value`.
2.  Used `LabelEncoder` for `ocean_proximity`.
3.  Re-fitted the encoder on validation and test data.

## Why LabelEncoder is Wrong Here

It imposes an artificial order:

INLAND → 0 ISLAND → 1 NEAR BAY → 2 NEAR OCEAN → 3 \<1H OCEAN → 4

These categories have no numerical ordering.

Correct approach: - OneHotEncoder.

## Upcoming Topics

-   One-Hot Encoding
-   Combining numeric and categorical features
-   Feature Scaling
-   Final feature matrix
-   Neural Network model

## Key Takeaways

-   Separate preprocessing by feature type.
-   Fit preprocessing only on training data.
-   Transform validation/test with learned statistics.
-   Prevent data leakage.
-   Understand manually before using automation like Pipeline and
    ColumnTransformer.
