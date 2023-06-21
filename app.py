from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))


@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    text = [str(x) for x in request.form.values()]
    vectorized_text = tfidf_vectorizer.transform(text)
    prediction = model.predict(vectorized_text)
    return render_template("index.html", prediction_text = "{}".format(prediction[0]), text=text[0])

if __name__ == "__main__":
    app.run(debug=True)
