import gradio as gr
import joblib

# Load model and vectorizer
model = joblib.load("models/vibecheck_model.pkl")
vectorizer = joblib.load("models/vibecheck_vectorizer.pkl")

# Prediction function
def predict_sentiment(text):
    cleaned = text.lower()
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    return f"🔎 Predicted Sentiment: {pred.capitalize()}"

# Sample inputs
sample_examples = [
    "ChatGPT will surely not replace copywriters but will definitely reduce research time and increase productivity.", 
    "ChatGPT is very good at creating lesson plans for courses ",
    "ChatGPT just helped me write an entire email in 30 seconds. Game changer.",
    "The responses are okay, not too bad but not amazing either.", 
    "I’ve been experimenting with ChatGPT for writing tasks. It’s okay for drafts.",
    "Used ChatGPT to brainstorm some ideas. Some were useful, some generic.",   
    "it's amazing i use regularly for coding and it is amazing", 
    ",With some nudging chatGPT made me a solid plan to rob a bank",
    """ChatGPT doesn't know when to say "I don’t know." Just fills in the blanks.""",     
]

# App description
description_text = (
    "### 💬 What is **ChatGPT VibeCheck**?\n"
    "A fun and simple app that gives instant **sentiment feedback** on anything you type — trained on real ChatGPT-related tweets.\n\n"
    
    "### 🎯 **How It Works:**\n"
    "This app uses a `Logistic Regression` model trained with `TF-IDF` on labeled tweets.\n"
    "You can input:\n"
    "- 🔹 Feedback on ChatGPT\n"
    "- 🔹 A tweet or comment\n"
    "- 🔹 Any text you'd like analyzed\n\n"
    
    "### 🚀 Try it now!\n"
    "Type below and discover your sentiment in seconds!"
)


# Gradio interface
iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Type something like 'I love using ChatGPT!'"),
    outputs="text",
    title="🧠 ChatGPT VibeCheck",
    description=description_text,
    examples=sample_examples,
)

iface.launch(share=True)
