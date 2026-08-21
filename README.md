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
