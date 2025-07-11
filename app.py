import os

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
model_name = "gpt2"

# initialize the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

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
    output = pipe(prompt, max_new_tokens=200)
    return output[0]['generated_text'].strip()

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
    response_parts = []
    ap_rows, ar_rows, inv_rows = [], [], []
    ap_values, ar_values, inv_values = [], [], []
    if file:
        filename = file.filename if file.filename is not None else ''
        if allowed_file(filename):
            safe_filename = secure_filename(filename)
            temp_dir = tempfile.gettempdir()
            file_path = os.path.join(temp_dir, safe_filename)
            file.save(file_path)
            try:
                if filename.lower().endswith('.csv'):
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
                # Find rows for AP, AR, Inventory (case-insensitive, partial match)
                
                for idx, row in df.iterrows():
                    row_str = ' '.join([str(x).lower() for x in row.astype(str)])
                    if 'accounts payable' in row_str or 'ap ' in row_str:
                        ap_rows.append(row)
                    if 'accounts receivable' in row_str or 'ar ' in row_str:
                        ar_rows.append(row)
                    if 'inventory' in row_str:
                        inv_rows.append(row)

                # Collect values for each category
                if ap_rows:
                    ap_df = pd.DataFrame(ap_rows)
                    ap_values = ap_df.select_dtypes(include=['number']).values.tolist()
                if ar_rows:
                    ar_df = pd.DataFrame(ar_rows)
                    ar_values = ar_df.select_dtypes(include=['number']).values.tolist()
                if inv_rows:
                    inv_df = pd.DataFrame(inv_rows)
                    inv_values = inv_df.select_dtypes(include=['number']).values.tolist()
                response_parts.append(f"AP: {ap_values}\nAR: {ar_values}\nInventory: {inv_values}")
            except Exception as e:
                response_parts.append(f"Error parsing file: {e}")
        else:
            response_parts.append("Invalid file type uploaded.")
    if msg:
        chat_response = get_Chat_response(msg)
        response_parts.append(f"Bot: {chat_response}")
    if not response_parts:
        response_parts.append("No message or file received.")
    return '\n'.join(response_parts)

if __name__ == '__main__':
    app.run()
