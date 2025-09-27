# Iris Flower Classification

## Project Overview

This project is a machine learning-powered web application that predicts the species of iris flowers based on their physical characteristics. The application uses a trained Random Forest classifier to identify whether an iris flower belongs to one of three species: Setosa, Versicolor, or Virginica.

## How It Works

The application takes four input parameters:
- **Sepal Length** - Length of the flower's sepal 
- **Sepal Width** - Width of the flower's sepal 
- **Petal Length** - Length of the flower's petal 
- **Petal Width** - Width of the flower's petal 

Based on these measurements, the trained machine learning model predicts which of the three iris species the flower most likely belongs to.

## Technology Stack

- **Backend**: Flask (Python web framework)
- **Machine Learning**: Random Forest Classifier with StandardScaler using Scikit-learn
- **Frontend**: HTML with embedded CSS styling
- **Model Serialization**: Pickle for model storage and loading
- **Data Processing**: NumPy and Pandas

## Project Structure

- `app.py` - Main Flask application with prediction endpoints
- `model_training.py` - Script to train and save the machine learning model
- `templates/index.html` - Web interface for user input and results display
- `model.pkl` - Serialized trained machine learning model
- `iris.csv` - Dataset containing iris flower measurements and classifications
