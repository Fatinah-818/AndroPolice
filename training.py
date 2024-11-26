import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import numpy as np
import joblib

# Load the dataset
data = pd.read_csv("combined.csv")

# Find and drop highly correlated features
numeric_data = data.select_dtypes(include=np.number)
correlation_matrix = numeric_data.corr().abs()
upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
high_correlation = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.9)]
data.drop(columns=high_correlation, inplace=True)
print(f"Dropped {len(high_correlation)} highly correlated columns.")

# Save the list of dropped highly correlated columns with UTF-8 encoding
with open('high_correlation_columns.txt', 'w', encoding='utf-8') as file:
    for column in high_correlation:
        file.write(f"{column}\n")

# Separate features (X) and label (y)
X = data.iloc[:, :-1]  # All rows, all columns except the last
y = data.iloc[:, -1]   # All rows, only the last column

# Convert object columns to numeric, handling non-numeric values
for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].astype('category').cat.codes

# Fill missing values with column mean for numeric columns only
numeric_cols = X.select_dtypes(include=np.number).columns
X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Remove low-variance features
selector = VarianceThreshold(threshold=0.01)
X = selector.fit_transform(X)

# Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=0.95)  # Retain 95% of variance
X_resampled_pca = pca.fit_transform(X_resampled)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled_pca, y_resampled, test_size=0.05, random_state=42, shuffle=True
)

# Define a hyperparameter grid for Random Forest
param_dist = {
    'n_estimators': [300, 500, 700],
    'max_depth': [15, 25, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
}

# Perform hyperparameter tuning using RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=50,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)
random_search.fit(X_train, y_train)
rf_best_model = random_search.best_estimator_

# Train a StackingClassifier for better accuracy
base_estimators = [
    ('rf', rf_best_model),
    ('xgb', XGBClassifier(random_state=42, eval_metric='mlogloss'))
]
stacking_clf = StackingClassifier(
    estimators=base_estimators,
    final_estimator=RandomForestClassifier(random_state=42)
)

stacking_clf.fit(X_train, y_train)

# Save the trained Stacking Classifier
joblib.dump(stacking_clf, 'random_forest_model.joblib')

# Evaluate the model on the test set
y_pred = stacking_clf.predict(X_test)

# Compute metrics
accuracy_test = accuracy_score(y_test, y_pred)
precision_test = precision_score(y_test, y_pred, average='binary')
recall_test = recall_score(y_test, y_pred, average='binary')
f1_test = f1_score(y_test, y_pred, average='binary')

# Print metrics
print(f"Test Accuracy: {accuracy_test:.4f}")
print(f"Precision: {precision_test:.4f}")
print(f"Recall: {recall_test:.4f}")
print(f"F1 Score: {f1_test:.4f}")
