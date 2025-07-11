import os
import math
import re

from flask import Flask, render_template, request, jsonify

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pipelines import pipeline
import torch

from werkzeug.utils import secure_filename
import tempfile
import pandas as pd

ALLOWED_EXTENSIONS = {'csv', 'xls', 'xlsx'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# load in hugging face token
hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
if hf_token is None:
    raise ValueError("Missing Hugging Face token. Set HUGGINGFACEHUB_API_TOKEN in your environment.")

# free model good with numerical analysis
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# initialize the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# create a text generation pipeline
pipe = pipeline(
    "text-generation", 
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
    formatted_prompt = f"<|system|>You are a helpful financial analyst assistant.</s><|user|>{prompt}</s><|assistant|>"
    output = pipe(formatted_prompt, max_new_tokens=1024, do_sample=True, temperature=0.7)
    return output[0]['generated_text'].split('<|assistant|>')[-1].strip()

# function to summarize values
def summarize_values(values):
    if not values:
        return "no data"
    flat = [item for sublist in values for item in sublist]
    # Filter out non-finite values (NaN, inf, -inf)
    flat = [x for x in flat if isinstance(x, (int, float)) and math.isfinite(x)]
    if not flat:
        return "no data"
    def usd(x):
        return f"${x:,.2f}"
    return (
        f"min={usd(round(min(flat), 2))}, "
        f"max={usd(round(max(flat), 2))}, "
        f"mean={usd(round(sum(flat)/len(flat), 2))}"
    )

# function to extract analysis from model output
def extract_analysis(model_output, prompt):
    if model_output.startswith(prompt):
        model_output = model_output[len(prompt):].lstrip()
    # Return more of the response and preserve formatting
    return model_output.strip()

# function to upload file    
@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    filename = file.filename if file.filename is not None else ''
    if file and allowed_file(filename):
        safe_filename = secure_filename(filename)
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, safe_filename)
        file.save(file_path)
        # process file code holder
        return jsonify({'success': True, 'filename': safe_filename}), 200
    else:
        return jsonify({'success': False, 'error': 'File upload failed'}), 400

# function to load file in chat progress
@app.route('/chat_upload', methods=['POST'])
def chat_upload():
    msg = request.form.get('msg', '')
    file = request.files.get('file')
    ap_rows, ar_rows, inv_rows = [], [], []
    ap_values, ar_values, inv_values = [], [], []
    if file:
        filename = file.filename if file.filename is not None else ''
        safe_filename = secure_filename(filename)
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, safe_filename)
        file.save(file_path)
        if filename.lower().endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        for idx, row in df.iterrows():
            row_str = ' '.join([str(x).lower() for x in row.astype(str)])
            # Normalize for matching
            row_str_nopunct = re.sub(r'[^a-z0-9 ]', ' ', row_str)
            # Accounts Receivable matching
            if (
                re.search(r'accounts receivable', row_str_nopunct) or
                re.search(r'\ba r\b', row_str_nopunct) or
                re.search(r'\bar\b', row_str_nopunct) or
                re.search(r'\ba r\b', row_str) or
                re.search(r'\ba r\b', row_str.replace('/', '')) or
                re.search(r'receivable', row_str_nopunct)
            ):
                ar_rows.append(row)
            # Accounts Payable matching
            if (
                re.search(r'accounts payable', row_str_nopunct) or
                re.search(r'\ba p\b', row_str_nopunct) or
                re.search(r'\bap\b', row_str_nopunct) or
                re.search(r'\ba p\b', row_str) or
                re.search(r'\ba p\b', row_str.replace('/', '')) or
                re.search(r'payable', row_str_nopunct)
            ):
                ap_rows.append(row)
            # Inventory matching
            if (
                re.search(r'inventory', row_str_nopunct) or
                re.search(r'stock', row_str_nopunct) or
                re.search(r'raw materials', row_str_nopunct) or
                re.search(r'finished goods', row_str_nopunct)
            ):
                inv_rows.append(row)
        if ap_rows:
            ap_df = pd.DataFrame(ap_rows)
            ap_values = ap_df.select_dtypes(include=['number']).values.tolist()
        if ar_rows:
            ar_df = pd.DataFrame(ar_rows)
            ar_values = ar_df.select_dtypes(include=['number']).values.tolist()
        if inv_rows:
            inv_df = pd.DataFrame(inv_rows)
            inv_values = inv_df.select_dtypes(include=['number']).values.tolist()
        prompt = (
            "Analyze these financial numbers:\n"
            f"Accounts Payable: {summarize_values(ap_values)}\n"
            f"Accounts Receivable: {summarize_values(ar_values)}\n"
            f"Inventory: {summarize_values(inv_values)}\n"
            "For each financial number, provide one sentence of analysis (trends and patterns).\n"
            "Insert a newline between each financial number and its analysis."
        )
        model_output = get_Chat_response(prompt)
        return extract_analysis(model_output, prompt)
    return get_Chat_response(msg)

if __name__ == '__main__':
    app.run()
