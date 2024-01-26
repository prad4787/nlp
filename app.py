# import os
# os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import pandas as pd
import streamlit as st
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
import joblib
from scipy.sparse import hstack

# Download NLTK stopwords data
nltk.download('stopwords')
nltk.download('punkt')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

def preprocess_abstract_text(df):
    # Step 1: Convert abstract_text to lowercase
    df['abstract_text'] = df['abstract_text'].str.lower()

    # Step 2: Remove HTML tags
    def remove_html_tags(text):
        pattern = re.compile('<.*?>')
        return pattern.sub(r'', text)
    df['abstract_text'] = df['abstract_text'].apply(remove_html_tags)

    # Step 3: Remove punctuation
    exclude = string.punctuation
    def remove_punc(text):
        return text.translate(str.maketrans('', '', exclude))
    df['abstract_text'] = df['abstract_text'].apply(remove_punc)

    # Step 4: Remove stopwords
    def remove_stopwords(text):
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text)
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(filtered_words)
    df['abstract_text'] = df['abstract_text'].apply(remove_stopwords)

    # Step 5: Lemmatize using spaCy
    def lemmatize_text_spacy(text):
        doc = nlp(text)
        lemmatized_tokens = [token.lemma_ for token in doc]
        return ' '.join(lemmatized_tokens)
    df['lemmatized_column'] = df['abstract_text'].apply(lemmatize_text_spacy)

    return df[['lemmatized_column', 'abstract_text', 'line_number', 'total_lines']]

def main():
    st.title("Abstract Level Sentence Classification")

    # Text input for the user to enter the abstract
    abstract_input = st.text_area("Enter the abstract:", "")

    if st.button("Classify"):
        # Create a DataFrame with the abstract lines
        abstract_lines = abstract_input.split(".")
        abstract_lines = [line.strip() for line in abstract_lines if line.strip()]
        df = pd.DataFrame({'abstract_text': abstract_lines})
        df['total_lines'] = len(df)
        df['line_number'] = df.index

        # Preprocess abstract
        df = preprocess_abstract_text(df)

        # Load models
        vectorizer = joblib.load('text_vectorizer.joblib')
        nb_model = joblib.load('naive_bayes.joblib')

        # Feature extraction
        inference_numerical = df[['line_number', 'total_lines']].values
        inference_bow = vectorizer.transform(df['lemmatized_column'])
        inference_final = hstack((inference_numerical, inference_bow), format='csr')

        # Make predictions
        predictions = nb_model.predict(inference_final)
        df['predictions'] = predictions

        # Display results in Streamlit
        st.write("Predictions:")
        st.table(df[['abstract_text', 'predictions']])

if __name__ == "__main__":
    main()
