import pandas as pd
from sklearn.impute import SimpleImputer
import joblib

# Step 1: Load merged_data.csv
merged_data = pd.read_csv('merged_data.csv')

# Step 2: Define Features and Target
X_new = merged_data.drop(['OS', 'sampleID'], axis=1, errors='ignore')  # Drop 'OS' and 'sampleID'
y_new = merged_data['OS']  # If you need the target variable for evaluation

# Separate numerical and categorical features
X_num = X_new.select_dtypes(include=['float64', 'int64'])
X_cat = X_new.select_dtypes(include=['object'])

# Impute missing values for numerical data
X_num_imputed = pd.DataFrame(SimpleImputer(strategy='median').fit_transform(X_num), columns=X_num.columns)

# Encode categorical variables with one-hot encoding
X_cat_imputed = pd.get_dummies(X_cat, drop_first=True)

# Combine the imputed numerical and one-hot encoded categorical features into final feature set
X_final_new = pd.concat([X_num_imputed, X_cat_imputed], axis=1)

# Step 3: Load the saved inference pipeline
pipeline_inference = joblib.load('random_forest_inference_pipeline.pkl')

# Step 4: Make predictions with the loaded pipeline
predictions = pipeline_inference.predict(X_final_new)

# Optional: If you want to evaluate the predictions against actual values of 'OS'
from sklearn.metrics import classification_report
print("Classification Report on the full dataset:")
print(classification_report(y_new, predictions))
