from fastapi import FastAPI
import torch
import random
import pandas as pd
from pydantic import BaseModel
from model import RNN_model
import nltk_utils
import time

# Initialize FastAPI app
app = FastAPI()

# Load dataset
df = pd.read_csv("Symptom2Disease.csv").drop_duplicates()
df.drop("Unnamed: 0", axis=1, inplace=True)

# Train-test split
train_data, test_data = train_test_split(df, test_size=0.15, random_state=42)

# Class names for disease prediction
class_names = {
    0: "Acne", 1: "Arthritis", 2: "Bronchial Asthma", 3: "Cervical spondylosis",
    4: "Chicken pox", 5: "Common Cold", 6: "Dengue", 7: "Dimorphic Hemorrhoids",
    8: "Fungal infection", 9: "Hypertension", 10: "Impetigo", 11: "Jaundice",
    12: "Malaria", 13: "Migraine", 14: "Pneumonia", 15: "Psoriasis",
    16: "Typhoid", 17: "Varicose Veins", 18: "Allergy", 19: "Diabetes",
    20: "Drug Reaction", 21: "GERD", 22: "Peptic Ulcer", 23: "UTI"
}

# Load trained model
model = RNN_model()
model.load_state_dict(torch.load("pretrained_symptom_model.pth", map_location=torch.device("cpu")))
model.eval()

# Disease Advice Dictionary
disease_advice = {
    "Acne": "Maintain a proper skincare routine and avoid touching affected areas.",
    "Arthritis": "Stay active with gentle exercises and consult a rheumatologist.",
    "Bronchial Asthma": "Follow prescribed inhalers, avoid smoke/allergens, and track symptoms.",
    "Cervical spondylosis": "Use ergonomic support, do neck exercises, and consult physiotherapy.",
    "Migraine": "Identify triggers, manage stress, and consult a neurologist.",
    "Diabetes": "Monitor blood sugar, maintain a balanced diet, and follow medication plans.",
    "UTI": "Stay hydrated, take prescribed antibiotics, and maintain hygiene.",
}

# Input Schema for Chatbot Query
class ChatQuery(BaseModel):
    message: str


@app.post("/predict")
async def predict_disease(query: ChatQuery):
    """Predicts disease based on symptoms using NLP processing"""
    transform_text = nltk_utils.vectorizer().transform([query.message])
    transform_text = torch.tensor(transform_text.toarray()).to(torch.float32)

    with torch.no_grad():
        y_logits = model(transform_text)
        pred_prob = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)

    disease = class_names[pred_prob.item()]
    advice = disease_advice.get(disease, "Consult a doctor for professional diagnosis.")
    
    return {"prediction": disease, "advice": advice}


@app.get("/")
async def root():
    return {"message": "Welcome to the Medical Chatbot API"}
