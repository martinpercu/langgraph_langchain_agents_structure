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
uv add langgraph langchain langchain-openai
uv add "langgraph-cli" --dev
uv add langchain-anthropic
uv add "fastapi[standard]"

# add dev dependencies
uv add "langgraph-cli[inmem]" --dev

# jupyter (important)
uv add ipykernel --dev

uv add grandalf --dev

# run the agent
uv run langgraph dev



# install the project
uv pip install -e .

```

#### Web site to visualise 
- https://mermaidviewer.com/editor
Example:
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	model(model)
	tools(tools)
	__end__([<p>__end__</p>]):::last
	__start__ --> model;
	model -.-> __end__;
	model -.-> tools;
	tools -.-> model;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

- This will show the flow we have in DEV.

## .toml
```sh
[tool.setuptools.packages.find]
where = ["src"]
include = ["*"]
```


## Simple Graph
- In notebooks the 02-simple shows a linear connextion with one node
- The agent "simple" is the same


## Messages
- In notebooks 03-messages shows how to concatenate messages. Human and AI messages
- The agent "simple-message" just add the "state" using MessagesState

## LLM
- In notebooks import API keys
- Check the "from langchain.chat_models import init_chat_model". Very useful to declare any LLM doesn't matter if OpenAi Anthropic Gemini whatever. 
- The agent "simple-llm" has the LLM integrated in the node_1.
