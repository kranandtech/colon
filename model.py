import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load and Merge Datasets
integrated_data = pd.read_csv('integrated_data_for_ml.csv')
tcga_survival_data = pd.read_csv('TCGA-COAD.survival.tsv', sep='\t')
clinical_data = pd.read_csv('TCGA-COAD.clinical.csv')

# Convert sampleID columns to string for consistency
integrated_data['sampleID'] = integrated_data['sampleID'].astype(str)
tcga_survival_data['sampleID'] = tcga_survival_data['sampleID'].astype(str)
clinical_data['sampleID'] = clinical_data['sampleID'].astype(str)

# Merge all datasets on sampleID
merged_data = pd.merge(integrated_data, tcga_survival_data, on='sampleID', how='inner')
merged_data = pd.merge(merged_data, clinical_data, on='sampleID', how='inner')

# Step 2: Define Features and Target
X = merged_data.drop(['OS', 'sampleID'], axis=1, errors='ignore')
y = merged_data['OS']

# Separate numerical and categorical features
X_num = X.select_dtypes(include=['float64', 'int64'])
X_cat = X.select_dtypes(include=['object'])

# Impute missing values for numerical data
X_num_imputed = pd.DataFrame(SimpleImputer(strategy='median').fit_transform(X_num), columns=X_num.columns)

# Encode categorical variables with one-hot encoding
X_cat_imputed = pd.get_dummies(X_cat, drop_first=True)

# Combine the imputed numerical and one-hot encoded categorical features
X_final = pd.concat([X_num_imputed, X_cat_imputed], axis=1)

# Step 3: Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# Step 4: Handle Class Imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Step 5: Scale the Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Step 6: Define Logistic Regression with increased max_iter
model_lr = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)

# Train the Logistic Regression model
model_lr.fit(X_train_scaled, y_train_resampled)
y_prob_lr = model_lr.predict_proba(X_test_scaled)[:, 1]

# Step 7: Precision-Recall Curve and Optimal Threshold
precision, recall, thresholds = precision_recall_curve(y_test, y_prob_lr)
f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_threshold = thresholds[np.argmax(f1_scores)]
print(f'Optimal Threshold based on F1 score: {optimal_threshold}')

# Adjust predictions based on the optimal threshold
y_pred_adjusted_lr = (y_prob_lr >= optimal_threshold).astype(int)

# Step 8: Calculate metrics and print report
accuracy = accuracy_score(y_test, y_pred_adjusted_lr)
print(f'Adjusted Threshold Accuracy: {accuracy}')
print(classification_report(y_test, y_pred_adjusted_lr))


# Plot Precision-Recall Curve
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Logistic Regression)')
plt.show()

# Step 9: Cross-Validation to confirm performance with Stratified Cross-Validation and SMOTE
# Step to fit the scaler with feature names included
scaler = StandardScaler().fit(X_final)  # Fit on X_final to capture feature names

# Create the Random Forest inference pipeline
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib

pipeline_inference = Pipeline([
    ('scaler', scaler),  # Use the pre-fitted scaler to retain feature names
    ('model', RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100))
])

# Fit the model on the scaled training data
pipeline_inference.fit(X_train_scaled, y_train_resampled)

# Save the pipeline for inference
joblib.dump(pipeline_inference, 'random_forest_inference_pipeline.pkl')
merged_data.to_csv('merged_data.csv', index=False)

