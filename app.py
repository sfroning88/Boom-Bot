import os

from flask import Flask, render_template, request, jsonify

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# load in hugging face token
hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
if hf_token is None:
    raise ValueError("Missing Hugging Face token. Set HUGGINGFACEHUB_API_TOKEN in your environment.")

# free model good with numerical analysis
model_name = "mistralai/Mistral-7B-Instruct-v0.1"

# tokenize and load the model
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    token=hf_token
)

# create a text generation pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

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
    formatted_prompt = f"""<s>[INST] {prompt.strip()} [/INST]"""
    output = pipe(formatted_prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
    return output[0]['generated_text'].split("[/INST]")[-1].strip()
    
if __name__ == '__main__':
    app.run()
