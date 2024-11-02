Table of Contents
Introduction
Project Structure
Data Loading and Libraries
Dataset Overview
Exploratory Data Analysis (EDA)
Data Preprocessing
Splitting the Data
Model Building
Training the Model
Model Evaluation
Results
Conclusion
Introduction
The House Price Prediction project aims to estimate house prices based on features such as the number of bedrooms, bathrooms, square footage, and other key attributes. A Random Forest regression model is used for this purpose.

Project Structure
1. Data Loading and Libraries
Libraries used:
numpy, pandas: Data manipulation and processing.
matplotlib, seaborn: Data visualization.
sklearn: For data scaling, model training, and evaluation.
The dataset is loaded from a CSV file using Pandas.
2. Dataset Overview
The dataset contains 21 columns with features such as bedrooms, bathrooms, sqft_living, floors, waterfront view, condition, and price.
After initial exploration, it was confirmed that the data has 21,613 rows with no missing values.
3. Exploratory Data Analysis (EDA)
We used dataset.info() and dataset.describe() to understand data types and statistics.
Visualizations:
Heatmap: Shows correlations between features.
Scatter plots: Highlights relationships between pairs of features.
Insights from this step helped select important features for prediction.
4. Data Preprocessing
The date column was removed as it was not relevant to the prediction task.
Feature Scaling: Applied using StandardScaler to normalize X (features) and y (target) values, optimizing the model's performance.
5. Splitting the Data
The dataset was split into:
Features (X): Independent variables.
Target (y): Dependent variable, price.
An 80-20 split was used for training and testing, ensuring sufficient data for model generalization.
6. Model Building
A Random Forest Regressor was selected for its robustness with non-linear relationships.
Parameters: 100 estimators and a random state of 42 for reproducibility.
7. Training the Model
The model was trained on the training set.
Compatibility issues with input shapes were addressed during this step, ensuring smooth training.
8. Model Evaluation
The model’s performance was evaluated using the r2_score metric.
The model achieved an R² score of approximately 0.854, indicating a high level of accuracy.
Results
The Random Forest model effectively predicts house prices with strong accuracy. This model could be further optimized through parameter tuning or adding additional features.

Conclusion
This project successfully demonstrates the end-to-end workflow of building a regression model for house price prediction. Future improvements may involve exploring alternative algorithms, fine-tuning the Random Forest parameters, or engineering additional features to enhance predictive performance.

