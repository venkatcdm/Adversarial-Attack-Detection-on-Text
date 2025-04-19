

# Define the hybrid model
import torch
import torch.nn as nn
from transformers import BertModel

class AdversarialTextDetector(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=3):
        super(AdversarialTextDetector, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.lstm = nn.LSTM(input_size=768, hidden_size=hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():  # Freeze BERT for contrastive step or tune as needed
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        lstm_out, _ = self.lstm(outputs.last_hidden_state)
        pooled_output = lstm_out[:, -1, :]  # Take the last hidden state

        logits = self.classifier(self.dropout(pooled_output))
        return logits, pooled_output  # logits for classification, pooled_output for contrastive loss


# Prediction function
import torch

# Assuming 'model', 'tokenizer', 'device', and 'test_loader' are defined as in the previous code.

def predict_and_adversarial_probability(model, tokenizer, device, text):
    model.eval()
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=128,  # Same max_length used during training
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits, _ = model(encoding["input_ids"], encoding["attention_mask"])
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    # Adversarial probability (probability of the text being adversarial - class 2)
    adversarial_probability = probabilities[0, 2].item()

    return predicted_class, adversarial_probability