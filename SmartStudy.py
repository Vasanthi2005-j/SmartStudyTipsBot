import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# App Config
st.set_page_config(page_title="Study Tips Chatbot 📚", page_icon="📚", layout="centered")

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load Study Tips CSV
csv_url = "study_tips.csv"  # Ensure this file is in the same directory or provide full path

try:
    df = pd.read_csv(csv_url, encoding="ISO-8859-1")
except Exception as e:
    st.error(f"Failed to load the CSV file. Error: {e}")
    st.stop()

df = df.fillna("")
df['Question'] = df['Question'].str.lower()
df['Answer'] = df['Answer'].str.lower()

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(df['Question'])

# Gemini API Setup
API_KEY = "AIzaSyAJ1YbZ_vt8PafCFyWf24__lsivzq5GYjg"  # Replace with your actual key
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Matching Function
def find_closest_question(user_query, vectorizer, question_vectors, df):
    query_vector = vectorizer.transform([user_query.lower()])
    similarities = cosine_similarity(query_vector, question_vectors).flatten()
    best_match_index = similarities.argmax()
    best_match_score = similarities[best_match_index]
    if best_match_score > 0.3:
        return df.iloc[best_match_index]['Answer']
    else:
        return None

# UI
st.title("📚 Smart Study Tips Chatbot")
st.write("Ask me how to study, prepare for exams, stay focused, and present your answers well!")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input
if prompt := st.chat_input("Ask your study-related question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    closest_answer = find_closest_question(prompt, vectorizer, question_vectors, df)

    if closest_answer:
        st.session_state.messages.append({"role": "assistant", "content": closest_answer})
        with st.chat_message("assistant"):
            st.markdown(closest_answer)
    else:
        try:
            response = model.generate_content(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
            with st.chat_message("assistant"):
                st.markdown(response.text)
        except Exception as e:
            st.error(f"Sorry, I couldn't generate a response. Error: {e}")
