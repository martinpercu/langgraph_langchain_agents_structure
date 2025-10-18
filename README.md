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
uv add "langgraph-cli" --dev
uv add --pre langchain-anthropic
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


