import os

from flask import Flask, render_template, request, jsonify

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch

# load in hugging face token
hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
if hf_token is None:
    raise ValueError("Missing Hugging Face token. Set HUGGINGFACEHUB_API_TOKEN in your environment.")

# free model good with numerical analysis
model_name = "google/flan-t5-large"

# initialize the model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# create a text generation pipeline
pipe = pipeline(
    "text2text-generation", 
    model=model, 
    tokenizer=tokenizer,
    device=-1 # CPU
)

# create a Flask app
app = Flask(__name__)

# set the template folder with chat formatting
@app.route("/")
def index():
    return render_template('chat.html')

# handle chat requests
@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)

# function to get chat response
def get_Chat_response(prompt):
    output = pipe(prompt, max_new_tokens=200)
    return output[0]['generated_text'].strip()
    
if __name__ == '__main__':
    app.run()
