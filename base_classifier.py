import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import json
import csv

# Load datasets
train_path = "./data/processed/atis_train.csv"
test_path = "./data/processed/atis_test.csv"

try:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
except Exception as e:
    raise Exception(f"Failed to load datasets: {e}")

# Preprocess datasets to handle NaN and empty strings
print("Preprocessing datasets...")
train_df = train_df.dropna(subset=['text'])  # Remove rows with NaN in 'text'
test_df = test_df.dropna(subset=['text'])
train_df = train_df[train_df['text'].str.strip() != '']  # Remove rows with empty strings
test_df = test_df[test_df['text'].str.strip() != '']
train_df = train_df[train_df['intent'].notna()]  # Remove rows with NaN in 'intent'
test_df = test_df[test_df['intent'].notna()]

print(f"Train size after preprocessing: {len(train_df)}, Test size after preprocessing: {len(test_df)}")

# Encode labels
label_encoder = LabelEncoder()
try:
    train_df['intent_encoded'] = label_encoder.fit_transform(train_df['intent'])
    test_df['intent_encoded'] = label_encoder.transform(test_df['intent'])
except Exception as e:
    raise Exception(f"Failed to encode labels: {e}")

# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(train_df['text'])
X_test_tfidf = tfidf_vectorizer.transform(test_df['text'])
y_train = train_df['intent_encoded']
y_test = test_df['intent_encoded']

# Train Logistic Regression
logistic_model = LogisticRegression(multi_class='multinomial', max_iter=1000)
logistic_model.fit(X_train_tfidf, y_train)

# Evaluate Logistic Regression
y_pred_logistic = logistic_model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred_logistic)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_logistic, average='weighted', zero_division=0)

# Print metrics
print("Logistic Regression Accuracy:", accuracy)
print("Logistic Regression Metrics:")
print(f"Precision (weighted): {precision:.4f}")
print(f"Recall (weighted): {recall:.4f}")
print(f"F1-score (weighted): {f1:.4f}")

# Save Logistic Regression metrics to JSON
logistic_metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1
}
metrics_report = {
    "logistic_metrics": logistic_metrics
}
os.makedirs("./metrics", exist_ok=True)
json_path = "./metrics/logistic_metrics_report.json"
try:
    with open(json_path, "w") as f:
        json.dump(metrics_report, f, indent=4)
    print(f"Metrics report saved to {json_path}")
except Exception as e:
    print(f"Failed to save metrics report to JSON: {e}")

# Save Logistic Regression metrics to CSV
csv_path = "./metrics/logistic_metrics_report.csv"
try:
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Accuracy", accuracy])
        writer.writerow(["Precision (weighted)", precision])
        writer.writerow(["Recall (weighted)", recall])
        writer.writerow(["F1-score (weighted)", f1])
    print(f"Metrics report saved to {csv_path}")
except Exception as e:
    print(f"Failed to save metrics report to CSV: {e}")

# Load DistilBERT metrics for comparison
distilbert_json_path = "./metrics/metrics_report.json"
try:
    with open(distilbert_json_path, "r") as f:
        distilbert_metrics_report = json.load(f)
    distilbert_metrics = distilbert_metrics_report.get("trained_metrics", {})
    distilbert_accuracy = distilbert_metrics.get("eval_accuracy", 0.0)
    distilbert_precision = distilbert_metrics.get("eval_precision", 0.0)
    distilbert_recall = distilbert_metrics.get("eval_recall", 0.0)
    distilbert_f1 = distilbert_metrics.get("eval_f1", 0.0)
except Exception as e:
    print(f"Failed to load DistilBERT metrics from {distilbert_json_path}: {e}")
    distilbert_accuracy = distilbert_precision = distilbert_recall = distilbert_f1 = 0.0

# Save comparison to CSV
comparison_csv_path = "./metrics/classifier_comparison.csv"
try:
    with open(comparison_csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Logistic Regression", "DistilBERT"])
        writer.writerow(["Accuracy", f"{accuracy:.4f}", f"{distilbert_accuracy:.4f}"])
        writer.writerow(["Precision (weighted)", f"{precision:.4f}", f"{distilbert_precision:.4f}"])
        writer.writerow(["Recall (weighted)", f"{recall:.4f}", f"{distilbert_recall:.4f}"])
        writer.writerow(["F1-score (weighted)", f"{f1:.4f}", f"{distilbert_f1:.4f}"])
    print(f"Classifier comparison saved to {comparison_csv_path}")
except Exception as e:
    print(f"Failed to save classifier comparison to CSV: {e}")

# Save Logistic Regression model, vectorizer, and label encoder
os.makedirs("./saved_models/logistic_classifier", exist_ok=True)
with open("./saved_models/logistic_classifier/model.pkl", "wb") as f:
    pickle.dump(logistic_model, f)
with open("./saved_models/logistic_classifier/vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf_vectorizer, f)
with open("./saved_models/logistic_classifier/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# Save evaluation results to text file (for compatibility)
os.makedirs("./classification_logs", exist_ok=True)
with open("./classification_logs/logistic_classifier_results.txt", "w") as f:
    f.write(f"Logistic Regression Accuracy: {accuracy}\n")
    f.write("Logistic Regression Metrics:\n")
    f.write(f"Precision (weighted): {precision:.4f}\n")
    f.write(f"Recall (weighted): {recall:.4f}\n")
    f.write(f"F1-score (weighted): {f1:.4f}\n")
    f.write("\nDistilBERT Metrics (from {distilbert_json_path}):\n")
    f.write(f"Accuracy: {distilbert_accuracy:.4f}\n")
    f.write(f"Precision (weighted): {distilbert_precision:.4f}\n")
    f.write(f"Recall (weighted): {distilbert_recall:.4f}\n")
    f.write(f"F1-score (weighted): {distilbert_f1:.4f}\n")