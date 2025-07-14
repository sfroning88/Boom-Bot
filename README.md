# Boom-Bot
Accepts financial statements and conducts analysis on working capital. <br>
Model: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1<br>

To install Python:<br>
https://www.python.org/downloads/<br>

To install Pip:<br>
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py<br>
python3 get-pip.py<br>

To install Libraries:<br>
python3 -m pip install --user -r requirements.txt<br>

To authorize MacOs HuggingFace token:<br>
echo 'export HUGGINGFACEHUB_API_TOKEN=hf_token_value' >> ~/.zshrc<br>
echo 'export OPENAI_API_KEY=openai_key_value' >> ~/.zshrc<br>
source ~/.zshrc<br>

To login to HuggingFace client on shell:<br>
huggingface-cli login<br>

To run the chat bot:<br>
cd [your directory]<br>
python3 app.py [free|premium]
