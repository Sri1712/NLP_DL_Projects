# Step 1: Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model
model = load_model('reviews_rnn_model.h5')

# Step 2: Helper Functions
# Function to decode a review from its integer representation back to text
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input review
def preprocess_review(review):
    words = review.lower().split() # Convert review to lowercase and split into words
    encoded_review = [word_index.get(word, 0) for word in words] # Convert words to their corresponding indices based on the word index
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500) # Pad the sequence to ensure it has the same length as the training data
    return padded_review

# Step 3: Predict Sentiment
# Predict the sentiment of a user input review

def predict_sentiment(review):
    processed_review = preprocess_review(review) # Preprocess the review
    prediction = model.predict(processed_review) # Make a prediction using the model
    sentiment = 'Positive' if prediction[0][0] >= 0.5 else 'Negative' # Interpret the prediction
    confidence = prediction[0][0] if prediction[0][0] >= 0.5 else 1 - prediction[0][0]
    return sentiment, confidence


# Streamlit app
import streamlit as st
st.title("IMDB Movie Review Sentiment Analysis")

# Input field for user review
user_review = st.text_area("Enter a movie review:")

# Button to trigger sentiment prediction
if st.button("Predict Sentiment"):
    if user_review.strip() == "":
        st.warning("Please enter a movie review.")
    else:
        preprocess_input = preprocess_review(user_review)
        sentiment, confidence = predict_sentiment(user_review)
        st.success(f"Predicted Sentiment: {sentiment}")
        st.info(f"Confidence Score: {confidence:.2f}")