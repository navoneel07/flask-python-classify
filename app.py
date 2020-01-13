from flask import Flask, request, jsonify, render_template

app = Flask(__name__)


@app.route('/predict', methods=['GET', 'POST'])
def predict():

    if(request.method == "POST"):
        review = request.form['review']

        return render_template("form.html", review=review)

    else:
        return render_template("form.html", review="")


if(__name__ == "__main__"):
    app.run(debug=True)
