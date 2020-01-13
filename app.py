from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    data = {"success": False}
    params = request.json
    if(params == None):
        params = request.args
    if (params != None):
        data["response"] = params.get("msg")
        data["success"] = True

    return jsonify(data)


if(__name__ == "__main__"):
    app.run(debug=True)
