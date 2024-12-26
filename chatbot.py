import streamlit as st
import spacy
from spacy.matcher import PhraseMatcher
import subprocess
import sys

# Ensure the SpaCy model is installed
def ensure_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")

# Load SpaCy NLP model
nlp = ensure_model()

# Initialize FAQ dataset
FAQ_DATA = FAQ_DATA = {
    "What is artificial intelligence?": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions.",
    "What are the types of AI?": "The main types of AI are Narrow AI, General AI, and Superintelligent AI.",
    "What is machine learning?": "Machine Learning is a subset of AI that allows machines to learn from data and improve their performance over time without being explicitly programmed.",
    "What is deep learning?": "Deep Learning is a subset of Machine Learning that uses neural networks with three or more layers to analyze various types of data.",
    "What are neural networks?": "Neural Networks are a series of algorithms that mimic the operations of a human brain to recognize relationships between vast amounts of data.",
    "What is natural language processing?": "Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and humans using natural language, allowing computers to understand and respond to text or voice data.",
    "What is supervised learning?": "Supervised Learning is a type of machine learning where the model is trained on labeled data, meaning the algorithm is provided with both input data and the corresponding correct output.",
    "What is unsupervised learning?": "Unsupervised Learning is a type of machine learning where the model is trained on unlabeled data, and the algorithm tries to identify patterns and relationships in the data without explicit instructions on what to look for.",
    "What is reinforcement learning?": "Reinforcement Learning is a type of machine learning where an agent learns by interacting with its environment and receiving feedback in the form of rewards or penalties.",
    "What is overfitting in machine learning?": "Overfitting occurs when a machine learning model learns not only the underlying patterns in the data but also the noise, making it perform poorly on new, unseen data.",
    "What is underfitting in machine learning?": "Underfitting occurs when a machine learning model is too simple to capture the underlying patterns in the data, resulting in poor performance on both training and new data.",
    "What is a convolutional neural network (CNN)?": "A Convolutional Neural Network (CNN) is a class of deep neural networks commonly used for image and video recognition tasks, which uses convolutional layers to automatically learn spatial hierarchies in images.",
    "What is transfer learning?": "Transfer Learning is a technique where a pre-trained model on one task is reused on another related task, allowing the model to leverage knowledge from a different domain to improve learning efficiency.",
    "What is a decision tree?": "A Decision Tree is a machine learning model that makes decisions by following a series of binary questions, representing decisions that lead to an outcome. It's widely used for classification and regression tasks.",
    "What is a random forest?": "A Random Forest is an ensemble learning method that creates a forest of decision trees, where each tree is trained on a random subset of data, and the final prediction is made by aggregating the predictions of all trees.",
}
# Function to find the best matching question
def get_faq_response(user_input):
    doc = nlp(user_input.lower())
    matcher = PhraseMatcher(nlp.vocab)

    # Create a mapping for patterns
    pattern_mapping = {}
    for question in FAQ_DATA:
        pattern_doc = nlp.make_doc(question.lower())
        matcher.add("FAQ", [pattern_doc])
        pattern_mapping[question.lower()] = question

    matches = matcher(doc)
    if matches:
        # Retrieve the most relevant question based on the match
        match_id, start, end = matches[0]
        matched_span = doc[start:end].text.lower()
        matched_question = pattern_mapping.get(matched_span, None)
        if matched_question:
            return FAQ_DATA[matched_question]
    return "I'm sorry, I couldn't find an answer to your question. Please try rephrasing it."

# Streamlit UI setup
st.set_page_config(page_title="FAQ Chatbot", page_icon="🧠", layout="wide")
st.title("🧠 FAQ Chatbot")
st.markdown("""
<style>
    .main {
        background-color: #f5f7fa;
    }
    .stTextInput > div > div > input {
        border: 2px solid #1f77b4;
        border-radius: 10px;
        padding: 10px;
    }
    .stButton button {
        background-color: #1f77b4;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .stButton button:hover {
        background-color: #0d47a1;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.header("Chatbot Settings")
st.sidebar.markdown("Adjust the chatbot settings below:")
theme_color = st.sidebar.color_picker("Pick a theme color:", "#1f77b4")
st.sidebar.markdown(f"<style>.stButton button {{ background-color: {theme_color}; }}</style>", unsafe_allow_html=True)

st.write("**Ask me anything about AI and its concepts!**")

# User input
user_input = st.text_input("Type your question here:")
if st.button("Submit"):
    if user_input:
        response = get_faq_response(user_input)
        st.success(response)
    else:
        st.warning("Please enter a question to get a response.")
