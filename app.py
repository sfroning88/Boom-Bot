from flask import Flask, render_template, request, jsonify

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

model_name = "mistralai/Mistral-7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)

def get_Chat_response(prompt):
    formatted_prompt = f"""<s>[INST] {prompt.strip()} [/INST]"""
    output = pipe(formatted_prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
    return output[0]['generated_text'].split("[/INST]")[-1].strip()
    
if __name__ == '__main__':
    app.run()
