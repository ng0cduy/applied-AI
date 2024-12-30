import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import matplotlib.pyplot as plt
import numpy as np

# Ensure NLTK downloads are ready
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load dataset
data = pd.read_csv('Suicide_Detection.csv')
print(data.head())

# Preprocessing function
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()  # Convert to lowercase
        text = ''.join(char for char in text if char.isalnum() or char.isspace())  # Remove special characters
        tokens = word_tokenize(text)  # Tokenize
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Lemmatize and remove stopwords
        return ' '.join(tokens)
    return ''

# Apply preprocessing
data['cleaned_text'] = data['Tweet'].apply(preprocess_text)

# Encode labels
label_encoder = LabelEncoder()
data['Suicide'] = label_encoder.fit_transform(data['Suicide'])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(data['cleaned_text'], data['Suicide'], test_size=0.2, random_state=42)

# Vectorization
vectorizer = CountVectorizer(max_features=5000)
X_train_vectorized = vectorizer.fit_transform(X_train).toarray()
X_test_vectorized = vectorizer.transform(X_test).toarray()

# Visualize word frequencies
word_counts = np.sum(X_train_vectorized, axis=0)
words = vectorizer.get_feature_names_out()
plt.bar(words[:20], word_counts[:20])
plt.xticks(rotation=45)
plt.title("Top 20 Word Frequencies")
plt.show()

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_vectorized, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test_vectorized, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).to(device)

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, activation_fn=nn.ReLU()):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # Binary classification
        self.dropout = nn.Dropout(0.3)  # Add dropout
        self.activation = activation_fn

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Initialize model, loss function, and optimizer
model = SimpleNN(input_size=X_train_tensor.shape[1]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with validation
num_epochs = 25
training_loss, validation_loss = [], []
training_acc, validation_acc = [], []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    # Training accuracy
    _, predicted = torch.max(outputs.data, 1)
    train_accuracy = (predicted == y_train_tensor).sum().item() / y_train_tensor.size(0)
    training_acc.append(train_accuracy)
    training_loss.append(loss.item())

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test_tensor)
        val_loss = criterion(val_outputs, y_test_tensor)
        _, val_predicted = torch.max(val_outputs.data, 1)
        val_accuracy = (val_predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
        validation_loss.append(val_loss.item())
        validation_acc.append(val_accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Acc: {train_accuracy*100:.2f}%, Val Loss: {val_loss.item():.4f}, Val Acc: {val_accuracy*100:.2f}%')

# Plot training and validation curves
plt.plot(training_loss, label="Training Loss")
plt.plot(validation_loss, label="Validation Loss")
plt.title("Loss Curve")
plt.legend()
plt.show()

plt.plot(training_acc, label="Training Accuracy")
plt.plot(validation_acc, label="Validation Accuracy")
plt.title("Accuracy Curve")
plt.legend()
plt.show()

# Prediction function with confidence
def predict(text):
    text = preprocess_text(text)
    vectorized_text = vectorizer.transform([text]).toarray()
    tensor_text = torch.tensor(vectorized_text, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        output = model(tensor_text)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)
    return label_encoder.inverse_transform(predicted.cpu().numpy())[0], confidence.item()

# Test the prediction function
example_text = "I feel so alone and hopeless."
label, confidence = predict(example_text)
print(f'The predicted label is: {label} with confidence {confidence*100:.2f}%')
