# Solar Power Generation Prediction

An end-to-end Machine Learning project utilizing Linear Regression to predict solar power output (kW) based on ambient environmental and weather conditions.

---

## 📌 Project Overview
As solar energy becomes increasingly integral to global power grids, predicting output helps operators optimize energy storage, grid integration, and power distribution. This project conducts Exploratory Data Analysis (EDA) and builds a linear regression model to predict the amount of power generated based on solar radiance, temperature, and other weather metrics.

---

## 📁 Repository Structure
```text
├── README.md
├── solar_power_prediction.ipynb   # Main Jupyter Notebook
└── solar_power_data.csv           # Raw dataset containing weather and generation metrics
```

---

## 📊 Dataset Description
The dataset (`solar_power_data.csv`) contains continuous features monitored at a solar array installation:
*   **`generated_power_kw`**: The target variable (Amount of solar power generated).
*   **Environmental factors**: Atmospheric pressure, temperature, humidity, wind speed, solar irradiance, and panel temperature.

---

## 🚀 Model Architecture & Pipeline

1. **Exploratory Data Analysis (EDA)**:
   * Data distributions visualized using Histograms and Boxplots (outlier detection).
   * Relationship mapping using Correlation Heatmaps and Bivariate Scatter Plots.
2. **Preprocessing**:
   * Outlier detection and missing value verification.
   * Splitting dataset into training (80%) and testing (20%) partitions.
   * Feature Scaling using `StandardScaler` to ensure uniform scale for linear regression optimization.
3. **Model Selection**:
   * `LinearRegression` from `scikit-learn` trained on the scaled training dataset.
4. **Performance Evaluation**:
   * The model is evaluated using **Mean Absolute Error (MAE)**:
     * **Train Set MAE**: ~392.42 kW
     * **Test Set MAE**: ~391.79 kW

---

## 🛠️ Installation & Setup
To run the project locally, install the required packages:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn jupyter
```

Then start the notebook environment:
```bash
jupyter notebook solar_power_prediction.ipynb
```

---

## 🔮 Future Enhancements
*   **Advanced Modeling**: Experiment with ensemble algorithms (Random Forest Regressor, XGBoost, or LightGBM) to capture non-linear relationships.
*   **Feature Engineering**: Incorporate time-of-day or seasonal indicators to capture cyclic solar trends.
*   **Hyperparameter Optimization**: Perform cross-validation and grid search for tuning model features.
