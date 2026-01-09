import pickle

# Load the trained model
with open('requirement_classifier.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
vectorizer = model_data['vectorizer']
label_encoder = model_data['label_encoder']

# Make predictions
new_text = "application must process data in real time"
new_features = vectorizer.transform([new_text])
prediction = model.predict(new_features)
label = label_encoder.inverse_transform(prediction)
print(f"Requirement Type: {label[0]}")