import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.data import find
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
from flask import Flask, render_template, request, jsonify

# Ensure the NLTK punkt resource is downloaded
nltk.data.path.append('/opt/render/nltk_data')  # Adjust the path if needed
try:
    find('tokenizers/punkt')
except LookupError:
    print("punkt resource not found, downloading...")
    nltk.download('punkt', download_dir='/opt/render/nltk_data')

# Initialize NLTK resources
lemmatizer = WordNetLemmatizer()

# Load model and data
model = load_model('model.h5')
intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))

def clean_up_sentence(sentence):
    try:
        sentence_words = word_tokenize(sentence)
        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words
    except Exception as e:
        print(f"Error in clean_up_sentence: {e}")
        return []

def bow(sentence, words, show_details=True):
    try:
        sentence_words = clean_up_sentence(sentence)
        bag = [0] * len(words)
        for s in sentence_words:
            for i, w in enumerate(words):
                if w == s:
                    bag[i] = 1
                    if show_details:
                        print("found in bag: %s" % w)
        return np.array(bag)
    except Exception as e:
        print(f"Error in bow: {e}")
        return np.array([0] * len(words))

def predict_class(sentence, model):
    try:
        p = bow(sentence, words, show_details=False)
        res = model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
        return return_list
    except Exception as e:
        print(f"Error in predict_class: {e}")
        return []

def getResponse(ints, intents_json):
    try:
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                responses = i['responses']
                result = "\n\n".join(responses)
                return result
        return "Sorry, I don't understand."
    except Exception as e:
        print(f"Error in getResponse: {e}")
        return "Sorry, I don't understand."

def chatbot_response(msg):
    try:
        ints = predict_class(msg, model)
        res = getResponse(ints, intents)
        return res
    except Exception as e:
        print(f"Error in chatbot_response: {e}")
        return "Sorry, there was an error."

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
            raise ValueError("No message provided.")
        response = chatbot_response(userText)
        return jsonify(response)
    except Exception as e:
        print(f"Error in get_bot_response: {e}")
        return jsonify("Sorry, there was an error.")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
