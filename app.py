import os

from flask import Flask, render_template, request, jsonify

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pipelines import pipeline
import torch
from werkzeug.utils import secure_filename
import tempfile

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
    if msg:
        # Use the existing chat response function
        chat_response = get_Chat_response(msg)
        response_parts.append(f"Bot: {chat_response}")
    if file:
        filename = file.filename if file.filename is not None else ''
        if allowed_file(filename):
            safe_filename = secure_filename(filename)
            temp_dir = tempfile.gettempdir()
            file_path = os.path.join(temp_dir, safe_filename)
            file.save(file_path)
            response_parts.append(f"File '{safe_filename}' received and saved.")
        else:
            response_parts.append("Invalid file type uploaded.")
    if not response_parts:
        response_parts.append("No message or file received.")
    return '\n'.join(response_parts)

if __name__ == '__main__':
    app.run()
