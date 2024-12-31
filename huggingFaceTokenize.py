import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
data = pd.read_csv('Suicide_Detection.csv')
# print(data.head())

# Preprocessing with Hugging Face Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess_with_tokenizer(texts, max_length=128):
    inputs = tokenizer(
        texts.tolist(),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    return inputs.to(device)

# Encode labels
label_encoder = LabelEncoder()
data['Suicide'] = label_encoder.fit_transform(data['Suicide'])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data['Tweet'], data['Suicide'], test_size=0.2, random_state=42
)

# Tokenize datasets
train_inputs = preprocess_with_tokenizer(X_train)
test_inputs = preprocess_with_tokenizer(X_test)

y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).to(device)

# Create Dataset and DataLoader
class TextDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs['input_ids'][idx],
            'attention_mask': self.inputs['attention_mask'][idx],
            'labels': self.labels[idx]
        }

train_dataset = TextDataset(train_inputs, y_train_tensor)
test_dataset = TextDataset(test_inputs, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Define the neural network
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

# Training loop
num_epochs = 3
training_loss = []
validation_loss = []
validation_acc = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            labels=batch['labels'].to(device)
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_train_loss = total_loss / len(train_loader)
    training_loss.append(avg_train_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")

# Evaluation
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for batch in test_loader:
        outputs = model(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device)
        )
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch['labels'].cpu().numpy())

print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))

# Prediction function
def predict(text):
    model.eval()
    inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence, predicted = torch.max(probs, dim=1)
    return label_encoder.inverse_transform(predicted.cpu().numpy())[0], confidence.item()

# Test the prediction function
example_text = "I'm feeling overwhelmed and hopeless."
label, confidence = predict(example_text)
print(f"Predicted label: {label} with confidence: {confidence*100:.2f}%")

# Plot training loss
plt.plot(training_loss, label="Training Loss", marker='o')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
