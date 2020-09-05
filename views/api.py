import flask
from flask import Flask, request, render_template

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")

@app.route('/', methods=['POST'])
def classify():
    userinput = str(request.form.getlist('userinput')[0])
    print(userinput)
    return render_template("index.html")

app.run()