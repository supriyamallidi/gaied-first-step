# 🚀 Email Routing

# Core Components

Python 3.11+

PyTorch (Deep Learning Framework)

Transformers Library (Hugging Face)

DistilBERT (Pre-trained NLP Model)

Pandas (Data Handling)

Scikit-learn (Metrics/Evaluation)

## 🎯 Approach

1. Generated email datasets & Attachments with the helper
2. Using one more helper extracted the content of the attachments
3. Using Mongodb to store the extracted content into the collection format.
4. Train the model from the email datasets that generated from step 1
5. Then analysing the emails based on subject + body + attachments content (if any)

## 🎥 Demo

🔗 [Live Demo](#) (if applicable)  
📹 [Video Demo](#) (if applicable)  
🖼️ Screenshots:

![Screenshot 1](link-to-image)

## 💡 Inspiration

What inspired you to create this project? Describe the problem you're solving.

## ⚙️ What It Does

Explain the key features and functionalities of your project.

## 🛠️ How We Built It

A. Taxonomy Design
B. Model Selection

- DistilBERT (Optimized for efficiency)
- 97% of BERT performance
- 40% smaller size
- 60% faster inference

C. Training Methodology

- Transfer learning approach
- Fine-tuning on domain-specific data
- Class-balanced sampling
- Progressive unfreezing (optional)

## 🚧 Challenges We Faced

Describe the major technical or non-technical challenges your team encountered.

## 🏃 How to Run

1. Unzip the attachmments
2. Run the script(GenerateEmailFilesFromJSON.py) to generate the emails with different sets of attachments in json format
3. Run the script(ExtractEmailContentToMongoDB.py) to read/extract the emails (from step 2) and attachments (step 1) contents into the mongodb
4. Run the runner script() to analyze the emails to extract the Request type, Request Subtype, Confidence Score, Extracted fields from email and Intent of the mail.

## 🏗️ Tech Stack

- 🔹 Backend: Python
- 🔹 Database: MongoDB
- 🔹 Other: OpenAI API / Twilio / Stripe

## 👥 Team

- **Supriya Mallidi**
- **Venkat Kanchumarthy**
- **Ramakrishna Dayala**
