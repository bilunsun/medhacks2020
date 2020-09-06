import flask
from flask import Flask, request, render_template

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")

@app.route('/', methods=['POST'])
def classify():
    text = str(request.form.getlist('text')[0])
    gene = str(request.form.getlist('gene')[0])
    mutation = str(request.form.getlist('mutation')[0])
    classification = text
    return render_template("index.html", classification=classification)

app.run()