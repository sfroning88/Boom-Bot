import os
import math
import re
from datetime import datetime

from flask import Flask, render_template, request, jsonify

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pipelines import pipeline
import torch

from werkzeug.utils import secure_filename
import tempfile
import pandas as pd

import matplotlib
matplotlib.use('Agg')

ALLOWED_EXTENSIONS = {'csv', 'xls', 'xlsx'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# load in hugging face token
hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
if hf_token is None:
    raise ValueError("Missing Hugging Face token. Set HUGGINGFACEHUB_API_TOKEN in your environment.")

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
    formatted_prompt = f"<|system|>You are a helpful financial analyst assistant.</s><|user|>{prompt}</s><|assistant|>"
    output = pipe(formatted_prompt, max_new_tokens=1024, do_sample=True, temperature=0.7)
    return output[0]['generated_text'].split('<|assistant|>')[-1].strip()

# function to extract analysis from model output
def extract_analysis(model_output, prompt):
    if model_output.startswith(prompt):
        model_output = model_output[len(prompt):].lstrip()
    # Return more of the response and preserve formatting
    return model_output.strip()

from support.processing import process_file, summarize_values, calculate_working_capital_ratios
from support.plotting import plot_financial_data

# --- chat_upload function ---
@app.route('/chat_upload', methods=['POST'])
def chat_upload():
    msg = request.form.get('msg', '')
    file = request.files.get('file')
    if file:
        periods, nwc = process_file(file)
        # For plotting, extract lists from nwc
        ar_values = [nwc[p][0] for p in periods]
        ap_values = [nwc[p][1] for p in periods]
        inv_values = [nwc[p][2] for p in periods]
        revenue_values = [nwc[p][3] for p in periods]
        ar_ratios = [nwc[p][4] for p in periods]
        ap_ratios = [nwc[p][5] for p in periods]
        inv_ratios = [nwc[p][6] for p in periods]
        wc_cycles = [nwc[p][7] for p in periods]
        prompt = (
            "You are a financial analyst. Based on this working capital data, provide exactly 3 sentences of analysis:\n"
            f"Data: {nwc}\n"
            "Respond with only 3 plain sentences - no numbering, no formatting, no repetition of the prompt. Focus on AR/AP/Inventory trends as % of revenue and practical recommendations."
        )
        model_output = get_Chat_response(prompt)
        analysis = extract_analysis(model_output, prompt)
        plot_path = plot_financial_data(ar_values, ap_values, inv_values, revenue_values, periods)
        return jsonify({'success': True, 'analysis': analysis, 'plot_path': plot_path}), 200
    return get_Chat_response(msg)

@app.route('/chat_followup', methods=['POST'])
def chat_followup():
    msg = request.form.get('msg', '')
    context = request.form.get('context', '')
    # Compose a prompt that includes the previous analysis/context and the new user question
    prompt = (
        f"Previous analysis: {context}\n"
        f"User follow-up question: {msg}\n"
        "As a financial analyst, answer the user's follow-up question in a concise, practical way, referencing the previous analysis if relevant."
    )
    ai_response = get_Chat_response(prompt)
    return jsonify({'success': True, 'response': ai_response}), 200

import sys
if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1].lower() not in ('free', 'premium'):
        print("Usage: python3 app.py [free|premium]")
        sys.exit(1)
    mode = sys.argv[1].lower()
    if mode == 'premium':
        # Placeholder for premium mode logic (e.g., OpenAI embeddings, 4o mini)
        pass
    elif mode == 'free':
        # free model good with numerical analysis
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        # initialize the model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        pipe = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer,
            device=-1 # CPU
        )

        app.run()
