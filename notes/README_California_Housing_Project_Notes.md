# California Housing Project -- Notes (EDA & Missing Values)

## Project Rules

-   Learn by reasoning, not copying code.
-   Receive hints instead of complete implementations.
-   Understand *why* each preprocessing step is performed.

------------------------------------------------------------------------

# Project Goal

Predict **`median_house_value`** using a Feed Forward Neural Network.

-   **Problem Type:** Regression
-   **Reason:** Output is a continuous numerical value rather than a
    class label.

------------------------------------------------------------------------

# Dataset

Total columns: **10**

### Target (Label)

-   `median_house_value`

### Features (9)

1.  longitude
2.  latitude
3.  housing_median_age
4.  total_rooms
5.  total_bedrooms
6.  population
7.  households
8.  median_income
9.  ocean_proximity

------------------------------------------------------------------------

# Dataset Source

### Recommended

Download the CSV once and keep it inside:

``` text
project/
│
├── data/
│   └── housing.csv
```

Load using:

``` python
pd.read_csv(...)
```

### Why not KaggleHub?

-   External dependency
-   API/version changes
-   Requires internet
-   Less reproducible

Using local CSV better resembles typical ML project organization.

------------------------------------------------------------------------

# Missing Values

Only one column contains missing values.

    total_bedrooms

Rows in dataset:

    20640

Non-null values:

    20433

Missing values:

    207

Approximately **1%** of the dataset.

------------------------------------------------------------------------

# Possible Strategies

## 1. Drop Column

Not recommended because the feature is valuable.

## 2. Drop Rows

Reasonable when only a very small percentage is missing.

## 3. Mean Imputation

Replace missing values using the average.

Pros - Simple - Preserves mean

Cons - Sensitive to outliers - Can insert unrealistic values

## 4. Median Imputation

Replace missing values using the median.

Pros - Robust to outliers - Preferred for skewed numerical data

## 5. Mode

Mainly useful for categorical variables.

## 6. KNN Imputation

Predict missing values using similar observations.

## 7. Regression Imputation

Train another model to estimate missing values.

## 8. Iterative Imputation (MICE)

Advanced statistical technique that repeatedly predicts missing values.

------------------------------------------------------------------------

# Summary Statistics

For `total_bedrooms`

    Mean   = 537.87
    Median = 435
    Min    = 1
    Max    = 6445

Observation:

-   Mean \> Median
-   Very large maximum
-   Strong indication of a right-skewed distribution

------------------------------------------------------------------------

# Why Mean Changes

Mean:

μ = (Σx)/n

Every observation contributes equally.

A few extremely large values pull the mean upward.

Example:

    50
    52
    53
    54
    55
    4000

Mean becomes unrealistically large.

------------------------------------------------------------------------

# Why Median Doesn't

Median depends only on the ordered position of observations.

Large outliers barely affect it.

Therefore it is called a **robust statistic**.

------------------------------------------------------------------------

# Histogram

## Purpose

A histogram groups continuous values into intervals called **bins**.

Each bar represents:

-   X-axis → range of feature values
-   Y-axis → frequency (number of observations)

Example:

    300–360 bedrooms

    Height = 2400

    Meaning:
    Approximately 2400 districts have between
    300 and 360 bedrooms.

------------------------------------------------------------------------

# What is a Bin?

A bin is simply a numerical interval.

Example:

    0–64
    64–128
    128–192
    ...

With 100 bins over values from 1 to 6445,

each bin is approximately 64 units wide.

------------------------------------------------------------------------

# Choosing Number of Bins

Too few bins: - Oversmooths the data - Hides important structure

Too many bins: - Produces noisy graphs - Difficult to interpret

Good exploratory choices: - 20--30 - 50 - 100 (reasonable for \~20k
observations)

There is no universally perfect number.

------------------------------------------------------------------------

# How to Read a Histogram

Ask these questions:

1.  What does the X-axis represent?
2.  What does the Y-axis represent?
3.  Where is most of the data?
4.  Is it symmetric?
5.  Is it skewed?
6.  Are there outliers?
7.  Are there multiple peaks?
8.  Does it agree with numerical statistics?

------------------------------------------------------------------------

# Understanding Skewness

Skewness is determined by the **tail**, not the tallest bar.

## Right-skewed

-   Tail extends toward larger values.
-   Mean \> Median.

## Left-skewed

-   Tail extends toward smaller values.
-   Mean \< Median.

## Symmetric

-   Mean ≈ Median.

Important:

The graph may be rotated visually, but skewness is always described
relative to the direction in which the variable increases.

------------------------------------------------------------------------

# Cause vs Effect

Cause: Large values stretch the distribution to the right.

Effect: The mean shifts toward those large values.

Therefore:

Mean \> Median is **evidence** of right skewness, not the definition of
it.

------------------------------------------------------------------------

# Key Takeaways

-   This is a regression project.
-   Use local CSV datasets for learning projects.
-   Keep all nine input features.
-   `median_house_value` is the target.
-   Only `total_bedrooms` has missing values.
-   Median imputation is appropriate because the distribution is
    right-skewed.
-   Histograms summarize distributions through bins.
-   Read histograms by studying tails, concentration, and shape---not
    just the tallest bars.
-   Always combine visualizations with numerical statistics before
    making preprocessing decisions.
