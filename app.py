import os
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.static_folder = 'static'

# Load model and data
model = load_model('model.h5')
intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def ensure_nltk_resources():
    """Ensure that NLTK resources are available."""
    try:
        nltk.data.find('tokenizers/punkt')
        print("NLTK punkt resource is available.")
    except LookupError:
        print("NLTK punkt resource is missing. Downloading...")
        nltk.download('punkt', download_dir='/opt/render/nltk_data')  # Adjust path if necessary

def clean_up_sentence(sentence):
    """Tokenize and lemmatize the input sentence."""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    """Create a bag of words representation for the input sentence."""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

def predict_class(sentence, model):
    """Predict the class of the input sentence using the trained model."""
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    """Get the response based on the predicted class."""
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            responses = i['responses']
            result = "\n\n".join(responses)
            break
    return result

def chatbot_response(msg):
    """Generate a chatbot response for the input message."""
    try:
        ensure_nltk_resources()  # Ensure NLTK resources are available
        ints = predict_class(msg, model)
        res = getResponse(ints, intents)
        return res
    except Exception as e:
        print(f"Error in chatbot_response: {e}")
        return "Sorry, there was an error."

@app.route("/")
def home():
    """Render the home page."""
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    """Handle the request to get a response from the chatbot."""
    try:
        userText = request.args.get('msg')
        if not userText:
            raise ValueError("No message provided.")
        response = chatbot_response(userText)
        return jsonify(response)
    except Exception as e:
        print(f"Error in get_bot_response: {e}")
        return jsonify("Sorry, there was an error.")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
