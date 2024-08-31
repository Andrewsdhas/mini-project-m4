import pickle
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

# Load the Breast Cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Load the trained model
with open('models/trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions
predictions = model.predict(X)

# Evaluate model performance
accuracy = accuracy_score(y, predictions)
print(f"Model Accuracy: {accuracy}")
