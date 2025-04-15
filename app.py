from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the model and vectorizer
with open('model (2).pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer (1).pkl', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None

    if request.method == 'POST':
        input_text = request.form['text']
        if input_text.strip():
            # Transform input and predict
            input_vect = vectorizer.transform([input_text])
            pred = model.predict(input_vect)[0]
            sentiment = 'Positive 😊' if pred == 1 else 'Negative 😞'

    return render_template('index.html', sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
