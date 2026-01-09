# SoftwareRequirements-Text-Classification
Python Machine Learning
Software Requirement Classification Model
A machine learning model that classifies software requirements as either Functional or Nonfunctional based on text descriptions. This project demonstrates natural language processing (NLP) techniques for requirement engineering and software development documentation.

📋 Project Overview
This project trains and evaluates multiple machine learning models to automatically classify software requirements. The dataset contains 400 requirements across various software domains, with 200 functional and 200 nonfunctional requirements.

Key Features
Text preprocessing with cleaning and normalization

TF-IDF feature extraction with n-grams

Multiple model comparison (6 different algorithms)

Comprehensive evaluation with visualizations

Prediction API for new requirements

Model persistence for deployment


📚 Related Research
This project implements techniques from:

NLP for Software Engineering

Requirement Classification

Text Mining for Documentation

Machine Learning for SE

Contribution Areas
Add more ML models

Improve text preprocessing

Create web interface

Add more datasets

Implement deep learning approaches

🔧 Code Structure
text
requirement-classifier/
│
├── requirement_classifier.py    # Main training script
├── Dataset.csv                  # Input dataset (not included)
├── requirement_classifier.pkl   # Saved model (generated)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── examples/                    # Example usage scripts
    ├── predict.py              # Prediction example
    └── test_requirements.txt   # Sample test cases
