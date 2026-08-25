# Create an isolated Python environment for the project
python -m venv .venv

# Activate the environment
source .venv/bin/activate

# Install backend and classification dependencies
pip install -r requirements.txt

# Install the project locally in editable mode
pip install -e .

Enables commands such as:
classify
evaluate
update-dataset

Copy `.env.example` to `.env` and set provider credentials.

Local Ollama (used by `rule_plus_ollama` in the default `classify` run):

```bash
ollama serve
ollama pull qwen2.5
```

Set `OLLAMA_BASE_URL` and `OLLAMA_MODEL` in `.env` if you do not want the
defaults (`http://localhost:11434/v1` and `qwen2.5`). Without a running
Ollama server, `classify` fails when it reaches `rule_plus_ollama`.
To run only the local strategy: `classify --strategies rule_plus_ollama`.
