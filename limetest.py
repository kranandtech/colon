import pandas as pd
import joblib

# Step 1: Load merged data and saved model
merged_data = pd.read_csv('merged_data.csv')
pipeline_rf = joblib.load('random_forest_inference_pipeline.pkl')

# Step 2: Prepare feature set for SHAP (exclude target and ID columns)
X_explain = merged_data.drop(['OS', 'sampleID'], axis=1, errors='ignore')

# Step 3: One-hot encode X_explain and align it with the training structure
X_explain_encoded = pd.get_dummies(X_explain)

# Get the expected feature names from the scaler
expected_features = pipeline_rf.named_steps['scaler'].get_feature_names_out()

# Identify missing columns, convert to a sorted list, and add them as zero columns
missing_cols = list(set(expected_features) - set(X_explain_encoded.columns))
missing_df = pd.DataFrame(0, index=X_explain_encoded.index, columns=missing_cols)

# Concatenate the missing columns DataFrame to X_explain_encoded and reorder columns to match expected structure
X_explain_aligned = pd.concat([X_explain_encoded, missing_df], axis=1)[expected_features]
