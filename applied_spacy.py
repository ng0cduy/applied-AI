import pandas as pd
import spacy
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
data = pd.read_csv('Suicide_Detection.csv')

# Load SpaCy model with GPU support
spacy.require_gpu()
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

# Preprocessing function with SpaCy
def preprocess_text_spacy(texts):
    """
    Preprocess texts in batches using SpaCy.
    """
    processed_texts = []
    for doc in nlp.pipe(texts, batch_size=1000):
        tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
        processed_texts.append(' '.join(tokens))
    return processed_texts

# Apply preprocessing
data['cleaned_text'] = preprocess_text_spacy(data['Tweet'])

# Visualize most frequent words
word_counts = Counter(" ".join(data['cleaned_text']).split())
most_common_words = word_counts.most_common(20)
words, counts = zip(*most_common_words)
print(words)
# sns.barplot(x=counts, y=words)
# plt.title("Top 20 Most Frequent Words")
# plt.show()

# Encode labels
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
data['Suicide'] = label_encoder.fit_transform(data['Suicide'])

# Split dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data['cleaned_text'], data['Suicide'], test_size=0.2, random_state=42)

# Convert text to vectors using SpaCy
X_train_vectors = [nlp(text).vector for text in X_train]
X_test_vectors = [nlp(text).vector for text in X_test]

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_vectors, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_vectors, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).to(device)

# Define the neural network
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # Binary classification

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# Initialize model, loss function, and optimizer
input_size = X_train_tensor.shape[1]
model = SimpleNN(input_size=input_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 25
training_loss = []
validation_loss = []
validation_acc = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

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
    text_vector = torch.tensor(nlp(text).vector, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(text_vector)
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
