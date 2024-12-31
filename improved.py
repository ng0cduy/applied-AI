import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
import spacy
import nltk
import matplotlib.pyplot as plt

# Download NLTK resources if not already downloaded
nltk.download('stopwords')

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Load dataset
data = pd.read_csv('Suicide_Detection.csv')  # Update with your dataset path
# print(data.head())

# Preprocessing with SpaCy
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return ' '.join(tokens)

# Apply preprocessing
data['cleaned_text'] = data['Tweet'].apply(preprocess_text)
print(data['cleaned_text'])
# Encode labels
label_encoder = LabelEncoder()
data['Suicide'] = label_encoder.fit_transform(data['Suicide'])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(data['cleaned_text'], data['Suicide'], test_size=0.2, random_state=42)

# Tokenize with Hugging Face tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_data(texts, labels, max_len=128):
    inputs = tokenizer(list(texts), padding=True, truncation=True, max_length=max_len, return_tensors='pt')
    inputs['labels'] = torch.tensor(labels, dtype=torch.long)
    return inputs

train_inputs = tokenize_data(X_train, y_train)
test_inputs = tokenize_data(X_test, y_test)

# Create Dataset and DataLoader
class TextDataset(Dataset):
    def __init__(self, inputs):
        self.inputs = inputs

    def __len__(self):
        return len(self.inputs['input_ids'])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.inputs.items()}

train_dataset = TextDataset(train_inputs)
test_dataset = TextDataset(test_inputs)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)

# Optimizer and loss function
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()
print(model)
# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

# Evaluation
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Classification report
print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))

# Prediction function
def predict(text):
    text = preprocess_text(text)
    inputs = tokenizer([text], padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence, predicted = torch.max(probs, dim=1)
    return label_encoder.inverse_transform(predicted.cpu().numpy())[0], confidence.item()

# Example prediction
example_text = "I feel hopeless and alone."
label, confidence = predict(example_text)
print(f"The predicted label is: {label} with confidence {confidence*100:.2f}%")
