import pandas as pd
import numpy as np
import streamlit as st
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD


st.title('Classifying Song Lyrics as a Genre')

# Load in previously trained classification model
# Load from file
pkl_model = "log_model_lyrics.pkl"
with open(pkl_model, 'rb') as file:
    LogModel = pickle.load(file)

pkl_bagofwords = "bagofwords_lyrics.pkl"
with open(pkl_bagofwords, 'rb') as file:
    bagofwords = pickle.load(file)

pkl_SVD = "truncatedSVD_lyrics.pkl"
with open(pkl_SVD, 'rb') as file:
    SVD = pickle.load(file)

# Get input on streamlit app
# Square brackets needed for coutn vectorizer to work
input_lyrics = [st.text_area('Input the lyrics of a song',height=400)]

# If statement activates when input is receieved
if input_lyrics:
    def predict(X):
        # Perform opperations on input
        input_count_vectorized = bagofwords.transform(X)
        input_SVD = SVD.transform(input_count_vectorized)
        prediction = LogModel.predict(input_SVD)
        #Convert prediction value to genre name
        if prediction == 0:
            st.write('Predicted song genre: **Rock**')
        elif prediction == 1:
            st.markdown('Predicted song genre: **Pop**')
        elif prediction == 2:
            st.markdown('Predicted song genre: **Alternative/Indie**')
        elif prediction == 3:
            st.markdown('Predicted song genre: **Metal**')
        elif prediction == 4:
            st.markdown('Predicted song genre: **Country**')
        elif prediction == 5:
            st.markdown('Predicted song genre: **Soul**')
        elif prediction == 6:
            st.markdown('Predicted song genre: **Hip-Hop/Rap**')
        elif prediction == 7:
            st.markdown('Predicted song genre: **Electronic/Dance**')

    predict(input_lyrics)

#### Classifaction dictionary
# Rock, Classic Rock, Alternative Rock, Hard Rock, Indie Rock combined = 0  
# Pop = 1  
# Alternative + Indie combined = 2  
# Metal + Heavy metal combined = 3  
# Country = 4  
# Soul = 5  
# Hip-hop + rap = 6  
# Electronic + Dance = 7  

