Generating Synthetic Clinical Trial Data with AI: Methods, Challenges, and Insights


1. To create/activate python ,venv:
  >python -m venv .venv
  >source .venv/bin/activate.csh

2. To run:
  >python ./src/dm.py


pip install -r requirements.txt



how to tremove trailiong speces:
sed -i -e 's/[[:space:]]*$//g' dm.py

openai: ma_horse@hotmail.com

>python -m pip install --upgrade pip

Local model:
ollama mode list:
ollama list
lama 3	8B	4.7GB	ollama run llama3
Llama 3	70B	40GB	ollama run llama3:70b
Phi 3 Mini	3.8B	2.3GB	ollama run phi3
Phi 3 Medium	14B	7.9GB	ollama run phi3:medium
Gemma 2	9B	5.5GB	ollama run gemma2
Gemma 2	27B	16GB	ollama run gemma2:27b
Mistral	7B	4.1GB	ollama run mistral
Moondream 2	1.4B	829MB	ollama run moondream
Neural Chat	7B	4.1GB	ollama run neural-chat
Starling	7B	4.1GB	ollama run starling-lm
Code Llama	7B	3.8GB	ollama run codellama
Llama 2 Uncensored	7B	3.8GB	ollama run llama2-uncensored
LLaVA	7B	4.5GB	ollama run llava
Solar	10.7B	6.1GB	ollama run solar

curl http://localhost:11434/api/generate -d '{  "model": "mistral",  "prompt":"Why is the sky blue?"}'

curl http://localhost:11434/api/chat -d '{ "model": "llama3",  "messages": [{ "role": "user", "content": "why is the sky blue?" } ]}'

# firewall-cmd --list-all
# firewall-cmd --zone=public --permanent --add-port 8080/tcp
# firewall-cmd --reload

Showing port listen
#netstat -tln | grep 11434

curl https://ollama.ai/install.sh | sh
ollama pull openhermes2.5-mistral
service ollama stop
env OLLAMA_HOST=0.0.0.0:11434 ollama serve
#nohup env OLLAMA_HOST=0.0.0.0:11434 ollama serve