from flask import Flask, request, jsonify, render_template
from helper import analyze
import tensorflow as tf
import keras
import numpy as np

app = Flask(__name__)

model = keras.models.load_model("model.h5")
model._make_predict_function()
graph = tf.get_default_graph()


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if(request.method == "POST"):
        review = request.form['review']
        review_tensor = analyze(review)

        global graph
        with graph.as_default():
            result = model.predict(np.array([review_tensor][0]))[0][0]

        if result <= 0.7:
            decision = "Review is negative"
        else:
            decision = "Review is positive"

        return render_template("form.html", review=review, score=decision)

    else:
        return render_template("form.html", review="")


if(__name__ == "__main__"):
    app.run(debug=True)
