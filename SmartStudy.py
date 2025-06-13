import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Page Config
st.set_page_config(page_title="📚 Smart Study Tips Chatbot", page_icon="🤖", layout="centered")

# Session Setup
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load CSV
csv_path = "study_tips.csv"  # Ensure this file exists
try:
    df = pd.read_csv(csv_path, encoding="ISO-8859-1")
except Exception as e:
    st.error(f"❌ Failed to load the CSV file. Error: {e}")
    st.stop()

df = df.fillna("")
df['Question'] = df['Question'].str.lower()
df['Answer'] = df['Answer'].str.lower()

# Vectorize
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(df['Question'])

# Gemini Setup
API_KEY = "AIzaSyAJ1YbZ_vt8PafCFyWf24__lsivzq5GYjg"  # Replace this with your key
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Match Closest Answer
def find_closest_question(user_query, vectorizer, question_vectors, df):
    query_vector = vectorizer.transform([user_query.lower()])
    similarities = cosine_similarity(query_vector, question_vectors).flatten()
    best_match_index = similarities.argmax()
    best_match_score = similarities[best_match_index]
    if best_match_score > 0.3:
        return df.iloc[best_match_index]['Answer']
    return None

# Chatbot Header UI
st.markdown("<h1 style='text-align: center;'>🤖 AI-Powered Study Tips Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>💡 Ask me anything about exam prep, time management, stress control, or answer presentation!</p>", unsafe_allow_html=True)
st.divider()

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.markdown(f"🧠 **Answer:**\n\n{message['content']}", unsafe_allow_html=True)
        else:
            st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask your study-related question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    closest_answer = find_closest_question(prompt, vectorizer, question_vectors, df)

    if closest_answer:
        with st.chat_message("assistant"):
            st.markdown(f"🧠 **Based on our study guide:**\n\n{closest_answer}")
        st.session_state.messages.append({"role": "assistant", "content": closest_answer})
    else:
        try:
            response = model.generate_content(prompt)
            answer = response.text

            # Optional: insert educational image
            if "time management" in prompt.lower():
                st.image("https://cdn.pixabay.com/photo/2016/03/31/20/11/hourglass-1297575_1280.png", width=300)
            elif "exam" in prompt.lower():
                st.image("https://cdn.pixabay.com/photo/2016/09/02/22/04/exam-1643316_1280.jpg", width=300)

            with st.chat_message("assistant"):
                st.markdown(f"🤖 **AI Assistant says:**\n\n{answer}")
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"❌ Could not generate a response. Error: {e}")
