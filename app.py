from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model and vectorizer
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    probabilities = None
    message = ""

    if request.method == "POST":

        message = request.form["message"]

        # Convert message into numbers
        message_bow = vectorizer.transform([message])

        # Predict category
        prediction = model.predict(message_bow)[0]

        # Get probabilities
        probability_values = model.predict_proba(message_bow)[0]

        probabilities = dict(
            zip(model.classes_, probability_values)
        )

        # Highest probability
        confidence = max(probability_values) * 100

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        probabilities=probabilities,
        message=message
    )


if __name__ == "__main__":
    app.run(debug=True)