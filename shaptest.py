import pandas as pd
import shap
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

# Step 4: Preprocess X_explain_aligned
X_explain_preprocessed = pipeline_rf.named_steps['scaler'].transform(X_explain_aligned.to_numpy())

# Step 5: Create SHAP explainer for the model
# Extract the trained model from the pipeline
model = pipeline_rf.named_steps['model']
explainer = shap.TreeExplainer(model)

# Step 6: Calculate SHAP values
# Check if shap_values is a list (for multiclass outputs)
shap_values = explainer.shap_values(X_explain_preprocessed)

# Verify the shapes to understand the structure
print("Shape of X_explain_aligned:", X_explain_aligned.shape)
print("Shape of X_explain_preprocessed:", X_explain_preprocessed.shape)
print("Type of shap_values:", type(shap_values))
if isinstance(shap_values, list):
    for i, sv in enumerate(shap_values):
        print(f"Shape of shap_values[{i}]:", sv.shape)

# Ensure that shap_values[1] matches (num_samples, num_features)
# Use shap_values[1] if the focus is on class 1 for binary classification
shap_summary_values = shap_values[1] if isinstance(shap_values, list) else shap_values

# Step 7: Plot SHAP summary plots
# Bar plot of average feature importance
shap.summary_plot(shap_summary_values, X_explain_aligned, plot_type="bar", feature_names=expected_features)

# Detailed summary plot
shap.summary_plot(shap_summary_values, X_explain_aligned, feature_names=expected_features)

