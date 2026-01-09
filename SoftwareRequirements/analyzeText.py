import pandas as pd
import numpy as np
import re
import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Models to try
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('Dataset.csv')

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nColumns:", df.columns.tolist())
print("\nRequirement Type distribution:")
print(df['Requirement Type'].value_counts())
print("\nAuthor distribution:")
print(df['Author'].value_counts())


# Text preprocessing function
def preprocess_text(text):
    if pd.isna(text):
        return ""

    # Convert to string and lowercase
    text = str(text).lower()

    # Remove special characters and extra spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    return text.strip()


# Apply preprocessing
df['Requirement_cleaned'] = df['Requirement'].apply(preprocess_text)
df['Scenario_cleaned'] = df['Scenario'].apply(preprocess_text)

# Combine features for better context
df['Combined_text'] = df['Scenario_cleaned'] + ' ' + df['Requirement_cleaned']

print("\nSample cleaned text:")
print(df[['Requirement', 'Requirement_cleaned']].head())

# Encode target labels
label_encoder = LabelEncoder()
df['Requirement_Type_encoded'] = label_encoder.fit_transform(df['Requirement Type'])

print("\nLabel mapping:")
for i, label in enumerate(label_encoder.classes_):
    print(f"{i}: {label}")

# Split the data
X = df['Combined_text']
y = df['Requirement_Type_encoded']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Create TF-IDF features
tfidf_vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2),
    stop_words='english'
)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print(f"\nTF-IDF feature shape: {X_train_tfidf.shape}")

# Dictionary to store model results
results = {}


# Function to train and evaluate a model
def train_evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    print(f"\n{'=' * 60}")
    print(f"Training {model_name}...")

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Get classification report
    report = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        output_dict=True
    )

    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    ))

    # Store results
    results[model_name] = {
        'model': model,
        'accuracy': accuracy,
        'report': report,
        'predictions': y_pred
    }

    return accuracy


# Try different models
models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    ),
    'Naive Bayes': MultinomialNB(),
    'SVM': SVC(
        kernel='linear',
        probability=True,
        random_state=42,
        class_weight='balanced'
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'
    ),
    'Neural Network': MLPClassifier(
        hidden_layer_sizes=(100, 50),
        max_iter=500,
        random_state=42
    ),
    'XGBoost': XGBClassifier(
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42
    )
}

# Train and evaluate all models
accuracies = {}
for name, model in models.items():
    acc = train_evaluate_model(model, name, X_train_tfidf, X_test_tfidf, y_train, y_test)
    accuracies[name] = acc

# Find best model
best_model_name = max(accuracies, key=accuracies.get)
best_model = results[best_model_name]['model']

print(f"\n{'=' * 60}")
print(f"BEST MODEL: {best_model_name}")
print(f"Best Accuracy: {accuracies[best_model_name]:.4f}")

# Feature importance for interpretable models - FIXED VERSION
print("\nFeature Importance Analysis:")

if best_model_name == 'Logistic Regression':
    print("\nTop 20 most important features for Functional requirements:")
    feature_names = tfidf_vectorizer.get_feature_names_out()
    coef = best_model.coef_[0]

    # Get indices of top features
    top_indices = np.argsort(coef)[-20:]

    for idx in top_indices[::-1]:
        print(f"{feature_names[idx]}: {coef[idx]:.4f}")

elif best_model_name == 'SVM':
    print("\nFor SVM, feature importance is available but in sparse format.")
    print("Here are the non-zero coefficients for Functional requirements:")

    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Get coefficients and convert to array
    if hasattr(best_model, 'coef_'):
        coef_array = best_model.coef_.toarray()[0] if hasattr(best_model.coef_, 'toarray') else best_model.coef_[0]

        # Get indices of features with highest absolute coefficients
        top_indices = np.argsort(np.abs(coef_array))[-20:]

        print("\nTop 20 most influential features (by absolute coefficient):")
        for idx in top_indices[::-1]:
            if coef_array[idx] != 0:
                print(f"{feature_names[idx]}: {coef_array[idx]:.4f}")

elif best_model_name == 'Random Forest':
    print("\nTop 20 most important features (Random Forest):")
    feature_names = tfidf_vectorizer.get_feature_names_out()
    importances = best_model.feature_importances_

    top_indices = np.argsort(importances)[-20:]

    for idx in top_indices[::-1]:
        print(f"{feature_names[idx]}: {importances[idx]:.4f}")

# Confusion matrix for best model
y_pred_best = results[best_model_name]['predictions']
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# Compare model performances
plt.figure(figsize=(10, 6))
models_list = list(accuracies.keys())
accuracy_values = [accuracies[m] for m in models_list]

bars = plt.bar(models_list, accuracy_values, color=['blue', 'green', 'red', 'purple', 'orange', 'brown'])
plt.ylim([0, 1])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xticks(rotation=45, ha='right')

# Add accuracy values on top of bars
for bar, acc in zip(bars, accuracy_values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
             f'{acc:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Analyze misclassifications
misclassified_indices = np.where(y_pred_best != y_test)[0]

print(f"\n{'=' * 60}")
print(f"MISCLASSIFIED EXAMPLES ({len(misclassified_indices)} samples):")

for idx in misclassified_indices[:min(5, len(misclassified_indices))]:  # Show first 5 or less
    print(f"\nExample {idx + 1}:")
    print(f"True Label: {label_encoder.inverse_transform([y_test.iloc[idx]])[0]}")
    print(f"Predicted: {label_encoder.inverse_transform([y_pred_best[idx]])[0]}")
    print(f"Scenario: {df.iloc[X_test.index[idx]]['Scenario']}")
    print(f"Requirement: {df.iloc[X_test.index[idx]]['Requirement']}")


# Function to predict new requirements
def predict_requirement_type(scenario_text, requirement_text):
    # Preprocess
    scenario_clean = preprocess_text(scenario_text)
    requirement_clean = preprocess_text(requirement_text)
    combined = scenario_clean + ' ' + requirement_clean

    # Transform
    combined_tfidf = tfidf_vectorizer.transform([combined])

    # Predict
    prediction_encoded = best_model.predict(combined_tfidf)[0]

    # Get probability if available
    if hasattr(best_model, 'predict_proba'):
        probability = best_model.predict_proba(combined_tfidf)[0]
    else:
        probability = [0, 0]  # Placeholder for models without probability

    # Get results
    predicted_label = label_encoder.inverse_transform([prediction_encoded])[0]

    print(f"\nPrediction Result:")
    print(f"Scenario: {scenario_text}")
    print(f"Requirement: {requirement_text}")
    print(f"\nPredicted Type: {predicted_label}")

    if hasattr(best_model, 'predict_proba'):
        for i, label in enumerate(label_encoder.classes_):
            print(f"Probability for {label}: {probability[i]:.4f}")

    return predicted_label, probability if hasattr(best_model, 'predict_proba') else None


# Test the prediction function with examples
print(f"\n{'=' * 60}")
print("TESTING PREDICTION FUNCTION:")

# Test case 1
print("\nTest Case 1:")
predict_requirement_type(
    "Smart Home System",
    "The system should securely encrypt all user data transmissions"
)

# Test case 2
print("\n\nTest Case 2:")
predict_requirement_type(
    "Health and Fitness Application",
    "Users should be able to track their daily step count"
)

# Test case 3
print("\n\nTest Case 3:")
predict_requirement_type(
    "IoT Application",
    "The application must have high performance and quick response times"
)

# Additional analysis: Common patterns
print(f"\n{'=' * 60}")
print("PATTERN ANALYSIS:")

# Get most common words for each requirement type
from collections import Counter

# Separate functional and non-functional requirements
functional_reqs = df[df['Requirement Type'] == 'Functional']['Requirement_cleaned']
nonfunctional_reqs = df[df['Requirement Type'] == 'Nonfunctional']['Requirement_cleaned']

# Count words
functional_words = ' '.join(functional_reqs).split()
nonfunctional_words = ' '.join(nonfunctional_reqs).split()

functional_word_counts = Counter(functional_words)
nonfunctional_word_counts = Counter(nonfunctional_words)

print("\nTop 10 words in Functional requirements:")
for word, count in functional_word_counts.most_common(10):
    print(f"{word}: {count}")

print("\nTop 10 words in Nonfunctional requirements:")
for word, count in nonfunctional_word_counts.most_common(10):
    print(f"{word}: {count}")

# Save the model and vectorizer (optional)
import pickle

model_data = {
    'model': best_model,
    'vectorizer': tfidf_vectorizer,
    'label_encoder': label_encoder,
    'accuracy': accuracies[best_model_name],
    'best_model_name': best_model_name
}

with open('requirement_classifier.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("\nModel saved as 'requirement_classifier.pkl'")

print(f"\n{'=' * 60}")
print("SUMMARY:")
print(f"Best Model: {best_model_name}")
print(f"Test Accuracy: {accuracies[best_model_name]:.4f}")
print(f"Misclassified Samples: {len(misclassified_indices)} out of {len(X_test)}")
print(f"Error Rate: {len(misclassified_indices) / len(X_test):.4f}")