# 📩 SMS Spam Detection Web Application

This project presents a Natural Language Processing (NLP)-based web application developed using a Machine Learning model to detect whether an SMS is **spam** or **not spam**. The app is simple, lightweight, and built with Streamlit, making it accessible to both technical and non-technical users for quick message screening.

---

## Objective

The primary goal of this application is to provide an effective tool for classifying SMS messages as spam or not spam using a trained machine learning model. It aims to help users filter out unwanted messages and enhance security against fraud and scams.

---

## Live Demo

Access the deployed application here:  
🔗 [SMS Spam Detection App – Streamlit](https://sms-spam-detection-system-using-nlp-saee-surve.streamlit.app/)

---

## Model Overview

- **Algorithm Used**: Multinomial Naive Bayes
- **Accuracy Achieved**: ~95.74%
- **Input**: Raw SMS text
- **Model Files**:
  - `model.pkl` – Serialized trained classification model
  - `vectorizer.pkl` – Serialized TF-IDF vectorizer used for transforming text

---

## Dataset

- **Source**: [Kaggle - SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Description**: The dataset contains a collection of labeled SMS messages for binary classification:
  - **Labels**: `ham` (not spam), `spam`
  - **Total Samples**: 5,574 SMS messages

---

## Tech Stack

- **Frontend & Deployment**: Streamlit
- **Natural Language Processing**: NLTK
- **Machine Learning**: Scikit-learn
- **Programming Language**: Python
- **Data Handling**: Pandas, NumPy
- **Model Serialization**: Pickle

---

## Repository Contents

- `app.py` – Streamlit web application code
- `model.pkl` – Trained classification model
- `vectorizer.pkl` – Fitted TF-IDF vectorizer
- `requirements.txt` – Python dependencies required for the project

GitHub Repository:  
🔗 [Saee-Surve/SMS-Spam-Detection](https://github.com/Saee-Surve)

---

## How It Works

1. The user enters a text message (SMS) into the input field.
2. The application:
   - Preprocesses the text (lowercasing, tokenization, stopword removal, stemming)
   - Transforms the text into numeric features using TF-IDF vectorization
   - Predicts whether the message is spam or not using a trained ML model
3. The result is displayed on the interface.

---

## Advantages

- High accuracy (~95.74%) on real-world SMS data
- Clean, minimal, and user-friendly interface
- Fast and responsive predictions
- Easy to deploy and extend for more use cases

---

> ⚠️ **Disclaimer**: This tool is for educational and demonstrative purposes only. It should not be used for critical filtering in real-world security systems without additional verification and testing.

---
