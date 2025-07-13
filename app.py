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

import matplotlib
matplotlib.use('Agg')

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
    import numpy as np
    if not values:
        return "no data"
    flat = []
    for sublist in values:
        for item in sublist:
            # Convert to string for parsing
            s = str(item).strip()
            if s in ('', '-', 'nan', 'NaN', 'None'):
                continue
            # Handle parentheses as negatives
            if s.startswith('(') and s.endswith(')'):
                try:
                    num = -float(s[1:-1].replace(',', ''))
                except Exception:
                    continue
            else:
                try:
                    num = float(s.replace(',', ''))
                except Exception:
                    continue
            if math.isfinite(num):
                flat.append(num)
    if not flat:
        return "no data"
    def usd(x):
        return f"${x:,.2f}"
    return (
        f"min={usd(round(np.min(flat), 2))}, "
        f"max={usd(round(np.max(flat), 2))}, "
        f"mean={usd(round(np.mean(flat), 2))}"
    )

# function to extract analysis from model output
def extract_analysis(model_output, prompt):
    if model_output.startswith(prompt):
        model_output = model_output[len(prompt):].lstrip()
    # Return more of the response and preserve formatting
    return model_output.strip()

# function to calculate working capital ratios and trends
def calculate_working_capital_ratios(ar_total, ap_total, inv_total, revenue_total):
    """Calculate working capital ratios and provide trend analysis"""
    if revenue_total <= 0:
        return {
            'ar_ratio': 0,
            'ap_ratio': 0,
            'inv_ratio': 0,
            'working_capital_cycle': 0,
            'analysis': "No revenue data available for ratio calculations."
        }
    
    ar_ratio = (ar_total / revenue_total) * 100
    ap_ratio = (ap_total / revenue_total) * 100
    inv_ratio = (inv_total / revenue_total) * 100
    
    # Basic working capital efficiency indicators
    working_capital_cycle = ar_ratio + inv_ratio - ap_ratio
    
    analysis = []
    if ar_ratio > 20:
        analysis.append(f"High AR ratio ({ar_ratio:.1f}%) suggests potential collection issues.")
    if ap_ratio > 15:
        analysis.append(f"High AP ratio ({ap_ratio:.1f}%) may indicate cash flow pressure.")
    if inv_ratio > 25:
        analysis.append(f"High inventory ratio ({inv_ratio:.1f}%) suggests potential overstocking.")
    if working_capital_cycle > 30:
        analysis.append(f"Long working capital cycle ({working_capital_cycle:.1f}%) indicates cash tied up in operations.")
    
    if not analysis:
        analysis.append("Working capital ratios appear healthy.")
    
    return {
        'ar_ratio': ar_ratio,
        'ap_ratio': ap_ratio,
        'inv_ratio': inv_ratio,
        'working_capital_cycle': working_capital_cycle,
        'analysis': ' '.join(analysis)
    }

def plot_financial_data(ap_values, ar_values, inv_values, revenue_values, periods=None, save_path="static/financial_plot.png"):
    import matplotlib.pyplot as plt
    import numpy as np
    def flatten(vals):
        return [item for sublist in vals for item in sublist] if vals and isinstance(vals[0], list) else vals or []
    ap = flatten(ap_values)
    ar = flatten(ar_values)
    inv = flatten(inv_values)
    rev = flatten(revenue_values)
    n = max(len(ap), len(ar), len(inv), len(rev))
    if not periods:
        periods = [f"Period {i+1}" for i in range(n)]
    def pad(lst):
        return lst + [np.nan] * (n - len(lst))
    ap, ar, inv, rev = map(pad, [ap, ar, inv, rev])
    plt.figure(figsize=(10, 6))
    plt.plot(periods, ap, marker='o', label='Accounts Payable')
    plt.plot(periods, ar, marker='o', label='Accounts Receivable')
    plt.plot(periods, inv, marker='o', label='Inventory')
    plt.plot(periods, rev, marker='o', label='Revenue')
    plt.xlabel('Period')
    plt.ylabel('Amount')
    plt.title('Working Capital Line Items Over Time')
    plt.legend()
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.savefig(save_path)
    plt.close()
    return save_path

# function to load file in chat progress
@app.route('/chat_upload', methods=['POST'])
def chat_upload():
    msg = request.form.get('msg', '')
    file = request.files.get('file')
    ap_rows, ar_rows, inv_rows, revenue_rows = [], [], [], []
    ap_values, ar_values, inv_values, revenue_values = [], [], [], []
    if file:
        filename = file.filename if file.filename is not None else ''
        if not allowed_file(filename):
            return jsonify({'success': False, 'error': 'Invalid file type. Please upload CSV or Excel files only.'}), 400
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
            row_str_nopunct = re.sub(r'[^a-z0-9 ]', ' ', row_str)
            if (
                re.search(r'accounts receivable', row_str_nopunct) or
                re.search(r'\ba r\b', row_str_nopunct) or
                re.search(r'\bar\b', row_str_nopunct) or
                re.search(r'\ba r\b', row_str) or
                re.search(r'\ba r\b', row_str.replace('/', '')) or
                re.search(r'receivable', row_str_nopunct)
            ):
                ar_rows.append(row)
            if (
                re.search(r'accounts payable', row_str_nopunct) or
                re.search(r'\ba p\b', row_str_nopunct) or
                re.search(r'\bap\b', row_str_nopunct) or
                re.search(r'\ba p\b', row_str) or
                re.search(r'\ba p\b', row_str.replace('/', '')) or
                re.search(r'payable', row_str_nopunct)
            ):
                ap_rows.append(row)
            if (
                re.search(r'inventory', row_str_nopunct) or
                re.search(r'stock', row_str_nopunct) or
                re.search(r'raw materials', row_str_nopunct) or
                re.search(r'finished goods', row_str_nopunct)
            ):
                inv_rows.append(row)
            if (
                re.search(r'revenue', row_str_nopunct) or
                re.search(r'sales', row_str_nopunct) or
                re.search(r'income', row_str_nopunct) or
                re.search(r'gross revenue', row_str_nopunct) or
                re.search(r'net revenue', row_str_nopunct) or
                re.search(r'total revenue', row_str_nopunct) or
                re.search(r'turnover', row_str_nopunct)
            ):
                revenue_rows.append(row)
        if ap_rows:
            ap_df = pd.DataFrame(ap_rows)
            ap_values = ap_df.select_dtypes(include=['number']).values.tolist()
        if ar_rows:
            ar_df = pd.DataFrame(ar_rows)
            ar_values = ar_df.select_dtypes(include=['number']).values.tolist()
        if inv_rows:
            inv_df = pd.DataFrame(inv_rows)
            inv_values = inv_df.select_dtypes(include=['number']).values.tolist()
        if revenue_rows:
            revenue_df = pd.DataFrame(revenue_rows)
            revenue_values = revenue_df.select_dtypes(include=['number']).values.tolist()
        revenue_total = sum([sum(sublist) for sublist in revenue_values]) if revenue_values else 0
        ap_total = sum([sum(sublist) for sublist in ap_values]) if ap_values else 0
        ar_total = sum([sum(sublist) for sublist in ar_values]) if ar_values else 0
        inv_total = sum([sum(sublist) for sublist in inv_values]) if inv_values else 0
        ratios = calculate_working_capital_ratios(ar_total, ap_total, inv_total, revenue_total)
        prompt = (
            "You are a financial analyst talking directly to a small business owner.\n"
            f"Revenue: {summarize_values(revenue_values)}\n"
            f"AP: {summarize_values(ap_values)} ({ratios['ap_ratio']:.1f}% of revenue)\n"
            f"AR: {summarize_values(ar_values)} ({ratios['ar_ratio']:.1f}% of revenue)\n"
            f"Inventory: {summarize_values(inv_values)} ({ratios['inv_ratio']:.1f}% of revenue)\n"
            f"Working Capital Cycle: {ratios['working_capital_cycle']:.1f}%\n"
            "Write a 3 sentence analysis of the working capital trends including recommendations for improvement."
        )
        model_output = get_Chat_response(prompt)
        analysis = extract_analysis(model_output, prompt)
        # Generate and save the plot
        plot_path = plot_financial_data(ap_values, ar_values, inv_values, revenue_values)
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

if __name__ == '__main__':
    app.run()
