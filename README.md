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

## Rag (basic)
- Using a openAI vector store. (we need the VectorStore ID)
- In notebook rag-basic implementing the file_search_tool using ==> bind_tools([file_search_tool]).
- In the agent rag-basic.py is important to understand the basic rag is will not work find if we send all the "history". So we send only the last message. (of course the issues is the model lost the memory of the conversation, this is a basic RAG implementation).

---

# IMPORTANT
#### 6 kind of workflows: 
1. Prompt Chaining 
2. Parallelization
3. Orchestrator-Worker
4. Evaluator-optimizer
5. Routing
6. Agent (usually using tools like a pdf reader)

## Prompt-Chaining
- In the notebook the 2 ways to create the chain. (builder.add_node AND builder.add_sequence)
- In the agent rag-basic-chaining.py just the implementation

## Structured-message
- In the notebook use pydantic to create a class BaseModel. (UserInfo).
- Define the LLM with this schema as output =>
```sh
llm_with_structured_output  = llm.with_structured_output(schema=UserInfo)
```
- In the rag-basic-structured-message just implement this in the def extractor to force the LLM to extract the info and continue sending the info with the defined schema.

---

# IMPORTANT-2
- If we see the langgraph docs how to structure we will se somethin like this ==> <br>
https://docs.langchain.com/oss/python/langgraph/application-structure#python-pyproject-toml<br><br>
my-app/
├── my_agent # all project code lies within here
│   ├── utils # utilities for your graph
│   │   ├── __init__.py
│   │   ├── tools.py # tools for your graph
│   │   ├── nodes.py # node functions for your graph
│   │   └── state.py # state definition of your graph
│   ├── __init__.py
│   └── agent.py # code for constructing your graph
├── .env # environment variables
├── langgraph.json  # configuration file for LangGraph
└── pyproject.toml # dependencies for your project
<br>
- This structure is ok but not great because we will need sooner a lot of nodes to control with differents tools states prompts etc etc. So in the "agents" folder I will use a new folder "support_agent" to add a folder for each node. In each node folder add whatever I need; promts tools whatever.

## Refactor to support agent
- This is just the refactoring what I have to the support support_agent.

## Template Prompt Jinja
- Install Jinja2
```sh
add uv Jinja2
```
- Using Jinja2 Template and PromptTemplate we have great control in the prompts. Check the notebook and see.
- Just a tip for Jinja2 ==> 
```sh
Without name the space above and below this block will be keeped
{% if name %} {% endif %}

Sugin "-" sign 
Without name the space above and below this block disappears

{%- if name %}
{% endif -%}
```


