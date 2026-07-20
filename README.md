

# KhushiDL - California Housing Rate Predictor

A Deep Learning project that builds a regression model using TensorFlow and Keras to predict median housing prices in California based on 1990 census data. The project tracks multiple development phases, version-controlled models, and historical training visualizations.

---

## 📁 Directory Structure

```text
.
├── California_housing_rate_predictor.html       # Exported HTML version of the main notebook
├── California_housing_rate_predictor.ipynb      # Main Jupyter Notebook for model development
├── data
│   └── housing.csv                            # Raw California housing dataset
├── graphs
│   ├── V1
│   │   ├── loss.png                           # Training vs Validation Loss for V1 model
│   │   └── mae.png                            # Training vs Validation MAE for V1 model
│   ├── V2
│   │   ├── loss.png                           # Training vs Validation Loss for V2 model
│   │   └── mae.png                            # Training vs Validation MAE for V2 model
│   └── bedrooms_histogram.png                 # Exploratory data distribution plot
├── jpnotebooks_versions
│   ├── California_housing_rate_predictor.ipynb  # Version-controlled notebook backups
│   └── California_housing_rate_predictor_v1.html # Previous notebook runtime export
├── model_versions
│   ├── KhushiDL_California_House_Prices_v1.keras # Saved Keras model (Version 1)
│   └── KhushiDL_California_House_Prices_v2.keras # Saved Keras model (Version 2)
├── notes
│   ├── README_California_Housing_Project_Notes.md # Project notes and development logs
│   ├── README_Phase_III_California_Housing_Project.md # Phase 3 implementation details
│   └── README_Phase_II_California_Housing_Notes.md    # Phase 2 milestones and updates
├── preprocessing.py                           # Standalone script containing clean data pipelines
└── requirements.txt                           # Project environment dependencies

```

---

## 📊 Dataset Overview

The project uses the classic **California Housing dataset** (`data/housing.csv`), which contains 20,640 records from the 1990 census. The model utilizes the following features to estimate home valuations:

* **Geographic Metrics:** `longitude`, `latitude`
* **Structural Metrics:** `housing_median_age`, `total_rooms`, `total_bedrooms`
* **Demographic Metrics:** `population`, `households`, `median_income`
* **Target Label:** `median_house_value` (The target value to be predicted)
* **Categorical Metric:** `ocean_proximity`

---

## ⚙️ Data Preprocessing & Engineering

The preprocessing logic is implemented inside the notebook and modularized within `preprocessing.py`. Core pipeline steps include:

1. **Handling Missing Values:** Processing null values natively present in features like `total_bedrooms`.
2. **Categorical Encoding:** Converting the textual `ocean_proximity` feature into numerical feature configurations suitable for a Deep Learning model.
3. **Feature Splitting:** Separating independent variables ($X$) from the target price tag variable ($y$).
4. **Feature Scaling:** Standardizing features to ensure uniform gradient propagation during neural network training.

---

## 🤖 Deep Learning Model Details

The predictive engine is an artificial neural network built using the **Keras API** via **TensorFlow**.

### Training Configurations:

* **Loss Function:** Mean Squared Error (MSE) / `loss`
* **Evaluation Metric:** Mean Absolute Error (MAE)
* **Epochs:** Up to 500 epochs.
* **Callbacks:** Integrated `EarlyStopping` callback to monitor `val_loss` with a patience threshold of 20 epochs, dynamically restoring best weights to prevent overfitting.

### Final Validation Performance (V2 Reference Plot):

* **Training Loss (MSE):** `~3.44e+09` | **Validation Loss:** `~3.70e+09`
* **Training MAE:** `~$40,193` | **Validation MAE:** `~$41,186`

---

## 📈 Visualizations

The `graphs/` directory maintains full line charts detailing model progression over training iterations:

* **V1 & V2 Comparison:** Check `graphs/V1/` and `graphs/V2/` to inspect execution curves mapping training losses and errors against validation validation tests over time.
* **Exploratory Data Analysis:** Check `graphs/bedrooms_histogram.png` to analyze structural feature layout balances.

---

## 🚀 Getting Started

### 1. Prerequisites

Ensure you have Python 3.8+ installed.

### 2. Install Dependencies

Install all required libraries, including TensorFlow, Pandas, NumPy, and Matplotlib using the provided file:

```bash
pip install -r requirements.txt

```

### 3. Run Preprocessing Pipeline

To clean and prepare the dataset dynamically via python script execution:

```bash
python preprocessing.py

```

### 4. Open the Notebook

Launch Jupyter Lab or Notebook environment to interact with the training model:

```bash
jupyter notebook California_housing_rate_predictor.ipynb

```

---

Made with ❤️ by Lucky Pawar