import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np

# Load scaler and label encoders
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))

# Define model
class OutfitIQModel(nn.Module):
    def __init__(self, input_dim):
        super(OutfitIQModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.out(x)
        return x

# Load trained model
input_dim = len(label_encoders)  # number of features
model = OutfitIQModel(input_dim)
model.load_state_dict(torch.load("outfitiq_model.pth"))
model.eval()

# Streamlit interface
st.title("OutfitIQ Price Predictor")
st.write("Predict the price of fashion items based on features.")

# Input features
def get_user_input():
    item_name = st.selectbox("Item Name", list(label_encoders['item_name'].classes_))
    category = st.selectbox("Category", list(label_encoders['category'].classes_))
    color = st.selectbox("Color", list(label_encoders['color'].classes_))
    size = st.selectbox("Size", list(label_encoders['size'].classes_))
    gender = st.selectbox("Gender", list(label_encoders['gender'].classes_))
    season = st.selectbox("Season", list(label_encoders['season'].classes_))
    material = st.selectbox("Material", list(label_encoders['material'].classes_))

    # Encode categorical inputs
    input_data = [
        label_encoders['item_name'].transform([item_name])[0],
        label_encoders['category'].transform([category])[0],
        label_encoders['color'].transform([color])[0],
        label_encoders['size'].transform([size])[0],
        label_encoders['gender'].transform([gender])[0],
        label_encoders['season'].transform([season])[0],
        label_encoders['material'].transform([material])[0]
    ]

    return np.array(input_data).reshape(1, -1)

user_input = get_user_input()

# Scale input
user_input_scaled = scaler.transform(user_input)
user_input_tensor = torch.tensor(user_input_scaled, dtype=torch.float32)

# Prediction
if st.button("Predict Price"):
    with torch.no_grad():
        predicted_price = model(user_input_tensor).item()
        st.success(f"Predicted Price: PKR {predicted_price:.2f}")
