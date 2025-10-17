## Create a new project


### Env with UV

```sh
# Install uv

curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version


## init
uv init
uv venv

# add dependencies
uv add --pre langgraph langchain langchain-openai
uv add --pre langchain-anthropic
uv add "fastapi[standard]"

# add dev dependencies
uv add "langgraph-cli[inmem]" --dev
uv add ipykernel --dev
uv add grandalf --dev

# run the agent
uv run langgraph dev



# install the project
uv pip install -e .
```

