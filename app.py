import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import logging
from flask import Flask, render_template, request, jsonify

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

# Initialize lemmatizer and load model
lemmatizer = WordNetLemmatizer()
model = load_model('model.h5')

# Load intents and model data
intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))

# Function to clean up sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Function to create bag of words
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

# Function to predict class
def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Function to get response based on predicted intent
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            responses = i['responses']
            result = "\n\n".join(responses)
            break
    return result

# Function to get chatbot response
def chatbot_response(msg):
    try:
        print(f"Received message: {msg}")  # Debugging line
        ints = predict_class(msg, model)
        res = getResponse(ints, intents)
        return res
    except Exception as e:
        logging.error(f"Error in chatbot_response function: {e}")
        return "Sorry, I didn't understand that."

# Initialize Flask app
app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    try:
        userText = request.args.get('msg')
        if not userText:
            return jsonify({"error": "No message provided"}), 400
        response = chatbot_response(userText)
        return jsonify({"response": response})
    except Exception as e:
        logging.error(f"Error in /get route: {e}")
        return jsonify({"error": "Internal server error"}), 500

# Run the app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
