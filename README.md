# Boom-Bot
Accepts financial statements and conducts analysis on working capital. <br>
Model: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1<br><br>

To install Python:<br>
https://www.python.org/downloads/<br><br>

To install Pip:<br>
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py<br>
python3 get-pip.py<br><br>

To install Flask:<br>
pip3 install -r requirements.txt<br><br>

To authorize MacOs HuggingFace token:<br>
echo 'export HUGGINGFACEHUB_API_TOKEN=hf_token_value' >> ~/.zshrc
source ~/.zshrc

To run the chat bot:<br>
cd [your directory]<br>
python3 app.py<br>
