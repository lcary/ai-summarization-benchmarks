# Adapted from the following URL for macOS compatibility:
# https://colab.research.google.com/github/camenduru/text-generation-webui-colab/blob/main/llama-2-7b-chat.ipynb

cd ~/w/ai-summarization-benchmarks/
brew install aria2

git clone -b v2.5 https://github.com/camenduru/text-generation-webui
pip install -r requirements.txt

cd text-generation-webui/

aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/4bit/Llama-2-7b-chat-hf/resolve/main/model-00001-of-00002.safetensors -d models/Llama-2-7b-chat-hf -o model-00001-of-00002.safetensors
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/4bit/Llama-2-7b-chat-hf/resolve/main/model-00002-of-00002.safetensors -d models/Llama-2-7b-chat-hf -o model-00002-of-00002.safetensors
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/4bit/Llama-2-7b-chat-hf/raw/main/model.safetensors.index.json -d models/Llama-2-7b-chat-hf -o model.safetensors.index.json
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/4bit/Llama-2-7b-chat-hf/raw/main/special_tokens_map.json -d models/Llama-2-7b-chat-hf -o special_tokens_map.json
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/4bit/Llama-2-7b-chat-hf/resolve/main/tokenizer.model -d models/Llama-2-7b-chat-hf -o tokenizer.model
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/4bit/Llama-2-7b-chat-hf/raw/main/tokenizer_config.json -d models/Llama-2-7b-chat-hf -o tokenizer_config.json
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/4bit/Llama-2-7b-chat-hf/raw/main/config.json -d models/Llama-2-7b-chat-hf -o config.json
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/4bit/Llama-2-7b-chat-hf/raw/main/generation_config.json -d models/Llama-2-7b-chat-hf -o generation_config.json

echo "dark_theme: true" > settings.yaml
echo "chat_style: wpp" >> settings.yaml

python server.py --settings settings.yaml --model models/Llama-2-7b-chat-hf --cpu --listen
