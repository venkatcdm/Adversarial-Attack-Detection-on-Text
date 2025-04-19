import streamlit as st
import torch
from transformers import AutoTokenizer
from model_utils import AdversarialTextDetector, predict_and_adversarial_probability

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Initialize and load the model
model = AdversarialTextDetector()
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

# Streamlit UI
st.title("Text Classification & Adversarial Probability")

user_input = st.text_area("Enter your text:")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        predicted_class, adv_prob = predict_and_adversarial_probability(model, tokenizer, device, user_input)
        label_map = {0: "Human-written", 1: "AI-generated", 2: "Adversarial"}

        st.markdown(f"**Predicted Class:** {label_map.get(predicted_class, 'Unknown')}")
        st.markdown(f"**Adversarial Probability:** {adv_prob:.4f}")
