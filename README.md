# Detecting Spam Messages using NLP
This repository contains a complete project for detecting spam messages using Natural Language Processing (NLP). The application is powered by a Naive Bayes classifier and is deployed using Heroku. It provides an interactive web interface built with Flask, where users can input messages and get real-time predictions on whether they are spam or not.

## Features
### 1. Dataset:
* The project uses a labeled dataset of messages (spam.csv), where messages are classified as either spam or ham (not spam).
### 2. Model:
* A Multinomial Naive Bayes model is trained using features extracted via CountVectorizer (Bag-of-Words model).
* The trained model achieves high accuracy and is saved as a .pkl file for deployment.
### 3. Web Application:
* Built using Flask, the web application offers a user-friendly interface for message input and prediction display.
### 4. Deployment:
* The application is deployed to Heroku.

## Usage Instructions
### 1. Ensure Prerequisites Are Installed:
* Install Python (version 3.9 or above recommended).
* Set up the Heroku CLI.
### 2. Install Dependencies:
* Install the required Python libraries by running:
  - pip install -r requirements.txt
### 3. Train the Model:
* If you wish to retrain the model, run:
  - python nlp_model.py
### 4. Run Locally:
* Start the Flask application:
  - python app.py
* Open a browser and go to http://127.0.0.1:5000/ to test the application.
### 5. Deploy to Heroku:
* Access the application via the Heroku-provided URL.
  
## Technologies Used
* Flask: For creating the web application.
* Heroku: For deploying the application online.
* Python Libraries:
 - pandas: For dataset preprocessing.
 - scikit-learn: For feature extraction and machine learning.
 - pickle: For saving and loading the trained model.
 - gunicorn: For serving the application on Heroku.
