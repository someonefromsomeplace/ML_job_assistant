# from flask import Flask, request, jsonify
# import torch
# from transformers import BertTokenizer, BertForSequenceClassification

# app = Flask(__name__)

# # Load model & tokenizer
# MODEL_PATH = "fake_job_model.pth"
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# #model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# class CustomBERTModel(torch.nn.Module):
#     def __init__(self):
#         super(CustomBERTModel, self).__init__()
#         self.bert = BertModel.from_pretrained("bert-base-uncased")
#         self.fc = torch.nn.Linear(768, 1)  # Make sure this matches your training model

#     def forward(self, input_ids, attention_mask):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         return self.fc(outputs.pooler_output)

# # Load the custom model instead of BertForSequenceClassification
# model = CustomBERTModel()

# model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
# model.eval()

# # Prediction function
# def predict_fake_job(description):
#     inputs = tokenizer(description, return_tensors="pt", truncation=True, padding=True, max_length=512)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     logits = outputs.logits
#     prediction = torch.argmax(logits, dim=1).item()
#     return "Fake Job" if prediction == 1 else "Real Job"

# # API route
# @app.route("/predict", methods=["POST"])
# def predict():
#     data = request.get_json()
#     description = data.get("description", "")
#     if not description:
#         return jsonify({"error": "No description provided"}), 400
#     result = predict_fake_job(description)
#     return jsonify({"prediction": result})

# if __name__ == "_main_":
#     app.run(debug=True)



from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertModel  # ✅ FIXED: Added BertModel

app = Flask(__name__)

# Load model & tokenizer
MODEL_PATH = "fake_job_model.pth"
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ✅ FIXED: Custom BERT Model
class CustomBERTModel(torch.nn.Module):
    def __init__(self):
        super(CustomBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc = torch.nn.Linear(768, 1)  # Output 1 neuron (for binary classification)
        self.sigmoid = torch.nn.Sigmoid()  # Convert logits to probabilities

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # Get [CLS] token representation
        return self.sigmoid(self.fc(pooled_output))  # Apply Sigmoid for binary classification

# Load the trained model
model = CustomBERTModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# ✅ FIXED: Predict Function
def predict_fake_job(description):
    inputs = tokenizer(description, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    with torch.no_grad():
        output = model(inputs["input_ids"], inputs["attention_mask"]).squeeze()  # Get scalar prediction

    prediction = "Fake Job ❌" if output.item() > 0.5 else "Real Job ✅"  # Threshold at 0.5
    return prediction

# API route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    description = data.get("description", "").strip()
    
    if not description:
        return jsonify({"error": "No description provided"}), 400
    
    result = predict_fake_job(description)
    return jsonify({"prediction": result})

# ✅ FIXED: Corrected `if __name__`
if __name__ == "__main__":
    app.run(debug=True)
