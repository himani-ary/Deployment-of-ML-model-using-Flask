import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create a Flask app
app = Flask(__name__)

# Load the model for making predictions
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    # Render the index.html template
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Retrieve the input values from the form
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    
    # Make predictions using the loaded model
    prediction = model.predict(features)

    # Render the index.html template with the prediction result
    return render_template("index.html", prediction_text=f"The flower species is {prediction}")


if __name__ == "__main__":
    # Run the Flask app
    app.run(debug=True, port=5000)
