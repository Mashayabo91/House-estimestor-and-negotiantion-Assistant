---
title: Rental Price Estimator & Negotiation Assistant
emoji: 🏠
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.44.1"
python_version: "3.10"
app_file: app.py
pinned: false
---

# 🏠 Rental Price Estimator & Negotiation Assistant

An AI-powered rental price estimation and negotiation assistant.

The application uses machine learning to estimate fair rental prices and the Groq API to provide practical negotiation advice.

## Features

- Rental price prediction
- Fair / overpriced / underpriced classification
- AI-powered negotiation advice
- Gradio interface
- Automatic Kaggle dataset download
- XGBoost machine-learning model

## Security

The Groq API key is stored securely as a Hugging Face Space Secret named `GROQ_API_KEY`.