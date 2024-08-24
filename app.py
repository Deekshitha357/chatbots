import logging
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
from flask import Flask, render_template, request, jsonify

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# NLTK setup
nltk.download('popular')
lemmatizer = WordNetLemmatizer()

# Load model and data
model = load_model('model.h5')
intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    logging.debug("found in bag: %s" % w)
    return np.array(bag)

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

def getResponse(ints, intents_json):
    if not ints:
        return "I'm not sure how to respond. Could you please provide more information or contact a doctor?"
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            responses = i['responses']
            result = "\n\n".join(responses)
            return result
    return "Sorry, I don't have information on that."

def chatbot_response(msg):
    try:
        ints = predict_class(msg, model)
        res = getResponse(ints, intents)
        return res  # Return as a plain string
    except Exception as e:
        logging.error(f"Error in chatbot_response: {e}")
        return "An error occurred while processing your request. Please try again later."

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    if not userText:
        logging.warning("No message provided")
        return jsonify({"response": "No message provided"}), 400
    
    response = chatbot_response(userText)
    logging.info(f"User: {userText}, Response: {response}")
    return jsonify({"response": response})  # Ensure response is a JSON object with a "response" key

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
