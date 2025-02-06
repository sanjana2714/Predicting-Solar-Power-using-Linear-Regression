# Predicting Solar Power Output using Linear Regression

## Overview
This project aims to predict solar power generation using a linear regression model. The dataset contains various environmental and operational parameters that influence solar power output. The project includes exploratory data analysis (EDA) and machine learning model training using scikit-learn.

## Dataset
The dataset used in this project is solarpowergeneration.csv, which contains multiple features related to solar power generation.

### Steps in the Project:

### 1. *Data Preprocessing and Exploration*
- Load the dataset using Pandas
- Display dataset structure and summary statistics
- Check for missing and duplicate values
- Perform exploratory data analysis (EDA) using histograms, scatter plots, and correlation heatmaps

### 2. *Data Visualization*
- Histogram plots to understand data distribution
- Scatter plots to analyze relationships between features and the target variable
- Correlation heatmap to identify relationships between variables
- Boxplots to detect outliers

### 3. *Machine Learning Model*
- *Feature selection:* Splitting dataset into independent (X) and dependent (y) variables
- *Train-test split:* Dividing the dataset into training (80%) and testing (20%) subsets
- *Feature scaling:* Standardizing features using StandardScaler
- *Model training:* Using LinearRegression from sklearn
- *Evaluation:* Calculating Mean Absolute Error (MAE) on train and test datasets

## Installation
To run this project, you need to install the required Python libraries:
sh
pip install pandas numpy seaborn matplotlib scikit-learn


## Usage
1. Clone the repository:
   sh
   git clone https://github.com/charan300804/Predicting-Solar-Power-Output-using-Linear-Regression.git
   
2. Navigate to the project directory:
   sh
   cd Predicting-Solar-Power-Output-using-Linear-Regression
   
3. Run the Python script or Jupyter Notebook to train and evaluate the model.

## Results
- The trained linear regression model produced a *Mean Absolute Error (MAE)* of approximately:
  - *Train Set:* 392.42
  - *Test Set:* 391.79
- The results suggest the model provides a reasonable approximation but might be improved with feature engineering or advanced models.

## Future Improvements
- Use additional machine learning models such as Random Forest or XGBoost for better performance.
- Feature engineering to extract more meaningful variables.
- Hyperparameter tuning to improve model accuracy.

## Contributors


## License
This project is open-source and available under the MIT License.
