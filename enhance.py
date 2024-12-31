import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import sys
from transformers import BertTokenizer
import torch
# Device configuration
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
# print(stop_words)
# Load dataset
data = pd.read_csv('Suicide_Detection.csv')  # Update with your dataset path
# print(data.head())

# Preprocessing function
def preprocess_text_batch(texts):
    """
    Batch process text for faster preprocessing.
    """
    cleaned_texts = []
    for text in texts:
        if isinstance(text, str):
            text = text.lower()
            text = re.sub(r'@\w+|#\w+|http\S+', '', text)  # Remove mentions, hashtags, URLs
            text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters
            tokens = nltk.word_tokenize(text)
            tokens = [word for word in tokens if word not in stop_words]
            cleaned_texts.append(' '.join(tokens))
        else:
            cleaned_texts.append('')
    return cleaned_texts

# Apply preprocessing
batch_size = 1000
batches = [data['Tweet'][i:i+batch_size] for i in range(0, len(data), batch_size)]
data['cleaned_text'] = pd.concat([pd.Series(preprocess_text_batch(batch)) for batch in batches], ignore_index=True)
# data['cleaned_text'] = data['Tweet'].apply(preprocess_text_batch)
# Visualize most frequent words
word_counts = Counter(" ".join(data['cleaned_text']).split())
most_common_words = word_counts.most_common(20)
words, counts = zip(*most_common_words)
sns.barplot(x=counts, y=words)
plt.title("Top 20 Most Frequent Words")
plt.show()

# Encode labels
label_encoder = LabelEncoder()
data['Suicide'] = label_encoder.fit_transform(data['Suicide'])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(data['cleaned_text'], data['Suicide'], test_size=0.2, random_state=42)

# Vectorization
vectorizer = CountVectorizer(max_features=5000)
X_train_vectorized = vectorizer.fit_transform(X_train).toarray()
X_test_vectorized = vectorizer.transform(X_test).toarray()

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_vectorized, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test_vectorized, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).to(device)

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # Assuming binary classification

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# Initialize model, loss function, and optimizer
model = SimpleNN(input_size=X_train_tensor.shape[1]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 25
training_loss = []
validation_loss = []
validation_acc = []

for epoch in range(num_epochs):
    # Training
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test_tensor)
        val_loss = criterion(val_outputs, y_test_tensor)
        _, val_predicted = torch.max(val_outputs.data, 1)
        val_accuracy = (val_predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)

    training_loss.append(loss.item())
    validation_loss.append(val_loss.item())
    validation_acc.append(val_accuracy)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}, Validation Acc: {val_accuracy*100:.2f}%")

# Save the model
torch.save(model.state_dict(), 'model.pth')

# Load the model
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Example prediction
def predict(text):
    text = preprocess_text_batch(text)
    vectorized_text = vectorizer.transform([text]).toarray()
    tensor_text = torch.tensor(vectorized_text, dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(tensor_text)
        _, predicted = torch.max(output.data, 1)
    return label_encoder.inverse_transform(predicted.cpu().numpy())[0]

# Test the prediction function
example_text = "I'm feeling really pressured with my work and I feel lost."
prediction = predict(example_text)
print(f"The predicted label for the text is: {prediction}")

# Plot training and validation loss
plt.plot(training_loss, label="Training Loss", marker='o')
plt.plot(validation_loss, label="Validation Loss", marker='o')
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# Plot validation accuracy
plt.plot(validation_acc, label="Validation Accuracy", marker='o')
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()
