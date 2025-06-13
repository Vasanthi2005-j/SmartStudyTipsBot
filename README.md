# 🤖📚 EduMentor: AI-Powered Study Tips Chatbot


An AI-powered Study Tips Chatbot built using **Streamlit** for the web interface, **Pandas** for data processing, **Scikit-learn** for TF-IDF vectorization and similarity computation, and **Google’s Gemini API** for generating personalized responses. This project provides students with real-time, intelligent study advice, including exam tips, time management strategies, and stress control techniques.

---

## 🌟 Features

- **Interactive Chat Interface**: Type your academic queries and receive instant suggestions.
- **Personalized Recommendations**: Study tips tailored to your input using content similarity and generative AI.
- **Content Understanding**: Uses TF-IDF and cosine similarity to understand and match user queries with relevant tips.
- **AI Responses**: Integrated with Google Gemini API to generate natural, contextual replies.
- **Clean UI**: Streamlit-based interface that is simple, fast, and user-friendly.

---

## 🛠️ Tech Stack

| Technology     | Purpose                                      |
|----------------|----------------------------------------------|
| Python         | Core programming language                    |
| Streamlit      | Building the interactive web frontend        |
| Pandas         | Data loading, cleaning, and processing       |
| Scikit-learn   | TF-IDF vectorization and similarity scoring  |
| Google Gemini API | Generative AI for human-like responses    |
| NumPy          | Numerical operations                         |
| Pickle (optional) | Loading pre-trained data/models           |

---

## 🔍 How It Works

1. **Data Preprocessing**:
   - A dataset of categorized study tips is cleaned and vectorized using TF-IDF.

2. **Similarity Matching**:
   - User queries are vectorized and compared to the dataset using **cosine similarity**.

3. **AI-Powered Response**:
   - The most relevant tip is returned directly or enhanced using the **Gemini API** to improve the naturalness of the reply.

---

## 📊 Application Workflow

### 🖥️ Frontend:
- Users type a query (e.g., “How to reduce exam stress?”).
- The chatbot responds with relevant study advice.

### ⚙️ Backend:
- Vectorizes the user query with TF-IDF.
- Matches with stored study tips using cosine similarity.
- Optionally enhances the output with Gemini API for improved language and tone.

---

