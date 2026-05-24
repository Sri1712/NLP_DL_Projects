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
    encoded_review = [word_index.get(word, 2) +3 for word in words] # Convert words to their corresponding indices based on the word index
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500, padding = 'pre', truncating= 'pre') # Pad the sequence to ensure it has the same length as the training data
    return padded_review

# Streamlit app
import streamlit as st
st.title("IMDB Movie Review Sentiment Analysis")
st.write('Enter a movie review to classify it as positive or negative.')

# Input field for user review
user_review = st.text_area("Enter a movie review:")

# Button to trigger sentiment prediction
if st.button("Predict Sentiment"):

    preprocessed_input=preprocess_review(user_review)

    ## Make prediction
    prediction=model.predict(preprocessed_input)
    sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'

    # Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write('Please enter a movie review.')
