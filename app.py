import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


ps = PorterStemmer()

def transform_text(sms):
    sms = sms.lower()
    sms = nltk.word_tokenize(sms)

    # This is used to remove all the special characters from the sms which are there in the Dataset
    y = []
    for i in sms:
        if i.isalnum():
            y.append(i)

    sms = y[:]
    y.clear()

    for i in sms:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    sms = y[:]
    y.clear()

    for i in sms:
        y.append(ps.stem(i))

    return " ".join(y)



tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier: By Ayush")

input_sms = st.text_area("Enter The Message")
if st.button('Predict'):

    #1. Preprocessing
    transformed_sms = transform_text(input_sms)
    #2. Vectorize
    vector_input = tfidf.transform([transformed_sms])
    #3. Predict
    result = model.predict(vector_input)[0]
    #4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
