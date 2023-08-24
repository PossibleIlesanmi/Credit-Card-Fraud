import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load the dataset into a pandas DataFrame
dataset_path = "C:/Users/Possible/Documents/Credit Card Fraud/archive/card_transdata.csv"
df = pd.read_csv(dataset_path)

# Feature selection (adjust this based on your domain knowledge and data analysis)
selected_features = ['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price', 'used_pin_number', 'online_order']

# Use only the selected features for training and validation
X = df[selected_features]
y = df['fraud']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Random Forest model with optimized hyperparameters
model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Evaluate the model's performance
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
roc_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

# Print the evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"AUC-ROC Score: {roc_auc}")
