import torch
import torch.nn.functional as F
from pymongo import MongoClient
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load your fine-tuned model and tokenizer from the saved directory.
model_path = "email_classifier_llm_latest"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)


def classify_email(email_text, max_length=128):
    """
    Tokenizes the input text, performs inference using the model,
    and returns the predicted label along with a confidence score.
    """
    inputs = tokenizer(
        email_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    # Apply softmax to obtain probabilities
    probabilities = F.softmax(logits, dim=1)
    predicted_class_id = logits.argmax(dim=1).item()
    # Get confidence score
    confidence_score = probabilities[0, predicted_class_id].item()

    # Map model output to a label.
    label_mapping = {0: "update", 1: "request"}
    return label_mapping[predicted_class_id], confidence_score


# Connect to MongoDB.
client = MongoClient("mongodb://localhost:27017")
db = client["email_routing"]
collection = db["email_datasets"]

# Iterate over each email document.
for doc in collection.find():
    subject = doc.get("subject", "")
    body = doc.get("body", "")
    # Combine subject and body to create the text input.
    email_text = f"{subject} {body}"

    # Classify the email.
    predicted_label, confidence_score = classify_email(email_text)

    # Update the document with the predicted label and confidence score.
    collection.update_one(
        {"_id": doc["_id"]},
        {"$set": {"predicted_label": predicted_label,
                  "confidence_score": confidence_score}}
    )

    print(
        f"Updated document {doc['_id']} with predicted label: {predicted_label}, confidence score: {confidence_score:.2f}")

print("Document classification and update process completed.")
