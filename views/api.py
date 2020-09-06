import flask
from flask import Flask, request, render_template
import sys
sys.path.append("..\medhacks2020")
from web_utils import get_single_inference_class
from model import GeneClassifier
import torch

model = GeneClassifier()
model.load_state_dict(torch.load("model.pth"))

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
    classification = get_single_inference_class(text, gene, mutation, model)
    classification = classification.index(max(classification)) + 1
    filler_response = "Your classification is: " + str(classification) + "!"
    return render_template("index.html", classification=filler_response)

app.run()