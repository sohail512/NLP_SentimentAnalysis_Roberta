import streamlit as st
import torch
from transformers import RobertaTokenizer, RobertaModel
import torch.nn as nn

# Load the tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('./tokenizer')

# Define the model architecture
class RobertaClass(nn.Module):
    def __init__(self):
        super(RobertaClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-base")
        self.pre_classifier = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, 5)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

# Load the model
model = RobertaClass()
model.load_state_dict(torch.load(r'C:\Users\Shakir Shaikh\Desktop\Projects DL\Sentiment Analysis\roberta_sentiment_model.pth', map_location=torch.device('cpu')))
model.eval()

# Define sentiment mapping
sentiment_map = {
    0: "Negative",
    1: "Somewhat Negative",
    2: "Neutral",
    3: "Somewhat Positive",
    4: "Positive"
}

# Streamlit App
st.title("Sentiment Analysis App")
st.write("Enter a review to see its sentiment classification.")

# Input text
user_input = st.text_area("Enter your review here:")

if st.button("Classify"):
    if user_input.strip() == "":
        st.write("Please enter a valid review!")
    else:
        # Tokenize the input
        inputs = tokenizer.encode_plus(
            user_input,
            None,
            add_special_tokens=True,
            max_length=52,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        # Predict
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, token_type_ids)
            prediction = torch.argmax(outputs, dim=1).item()

        # Display the sentiment
        st.write(f"Predicted Sentiment: **{sentiment_map[prediction]}**")
