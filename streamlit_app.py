import streamlit as st
import torch
import torch.nn as nn

# Define the model architecture
class FakeJobClassifier(nn.Module):
    def __init__(self, input_size=6, hidden_size=32, num_classes=2):
        super(FakeJobClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Load the model architecture
model = FakeJobClassifier()

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model file path
model_path = r"C:\Users\DELL\Desktop\JOBAPP\model\fake_job_model_full.pth"

# Load model weights
try:
    state_dict = torch.load(model_path, map_location=device)

    # Just load the state dict directly
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    st.success("Model loaded successfully!")

except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# Streamlit UI
st.title('🧠 Fake Job Classification App')
st.write("Enter job features below to predict whether it's **Fake** or **Real**.")

# Input fields
title = st.text_input('Job Title')
company_profile = st.text_area('Company Profile')
description = st.text_area('Job Description')
requirements = st.text_area('Requirements')
benefits = st.text_area('Benefits')

# Feature extraction
def extract_features(title, company_profile, description, requirements, benefits):
    features = [
        len(title),
        len(company_profile),
        len(description),
        len(requirements),
        len(benefits),
        title.count(' ')
    ]
    return torch.tensor(features, dtype=torch.float32)

# Prediction logic
if st.button('Predict'):
    if model:  # Model must be loaded
        with st.spinner('Predicting...'):
            features = extract_features(title, company_profile, description, requirements, benefits)
            features = features.to(device)
            outputs = model(features.unsqueeze(0))  # Add batch dimension
            _, predicted = torch.max(outputs.data, 1)

            if predicted.item() == 1:
                st.error('🚨 This job posting seems **FAKE**!')
            else:
                st.success('✅ This job posting seems **REAL**!')

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and PyTorch")
