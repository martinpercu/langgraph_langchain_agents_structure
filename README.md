## 🤖 Agents Structure with Langgraph & LangChain: Advanced Workflows

This repository provides a comprehensive, structured project template for building complex AI agents using **LangGraph** and **LangChain**.

The primary focus is on establishing a robust project structure (`src/agents/support_agent`) and implementing six core types of multi-step agent workflows.

### 🎯 Key Features & Workflows

| Workflow Type | Description |
| :--- | :--- |
| **Basic Agent** | Shows simple linear graphs, message concatenation, and LLM integration (02-03-LLM). |
| **RAG Implementation** | Demonstrates basic RAG using OpenAI vector stores and `bind_tools` (Rag basic). |
| **Prompt Chaining** | Connects nodes sequentially using `builder.add_sequence` (Prompt-Chaining). |
| **Structured Output** | Enforces Pydantic schemas for structured LLM output using `llm.with_structured_output` (Structured-message). |
| **ReAct & Tool Use** | Implements the **Reasoning and Acting (ReAct)** pattern, where the LLM decides which external tool to call (e.g., `get_weather`, `book_appointment`). |
| **Conditional Routing** | An agent makes decisions on where to go next (e.g., `conversation` vs. `booking`) using `builder.add_conditional_edges` (Routing Agent Decision). |
| **Parallelization** | Executes two or more nodes simultaneously and condenses the results (Parallel). |
| **Orchestrator-Worker** | A dedicated node decides which worker nodes to activate using the `Send` mechanism (Orchestrator). |
| **Evaluator-Optimizer** | Implements a feedback loop: an `evaluator_node` judges output and sends feedback back to the `generator_node` for refinement (Evaluator Optimizer). |

### 🛠️ Structure and Infrastructure

* **Structure:** Follows a modular architecture where each agent (e.g., `support_agent`) has dedicated folders for `nodes/`, `routes/`, and `state.py`.
* **Prompting:** Uses **Jinja2** for templating to gain fine-grained control over prompt generation.
* **FastAPI & Database:** Includes a **FastAPI** backend integrated with **PostgreSQL** checkpoints using `langgraph-checkpoint-postgres` for persistent conversation history and dynamic agent creation. 

This repository serves as a guide for engineering scalable, production-ready AI agent systems.



### This will be the final structure to manage agents.

```sh
curso_langchain_agent/
├── src/                                   # main project code
│   ├── agents/                            # agents and workflows
│   │   └── support_agent/                 # customer support agent
│   │       ├── nodes/                     # graph node functions
│   │       ├── routes/                    # routing logic
│   │       ├── state.py                   
│   │       └── support_agent.py           
│   └── api/                               # FastAPI backend
├── notebooks/                             # Jupyter notebooks for examples
├── init-scripts/                          # database initialization scripts
├── .env                                   # environment variables
├── langgraph.json                         # LangGraph configuration
├── docker-compose.yml                     # Docker configuration
├── pyproject.toml                         # project dependencies
└── README.md                              # documentation
```


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
uv add langchain-google-genai
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

# install langgraph for DB postgres
uv add langgraph-checkpoint-postgres

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


## Tools with Re-Act (fromlangchain)
- In the notebook we start with the basic agent function. The LLM will chose witch tool will use. 
- This is the Reason Action (ReAct) concept. Is an LLM (better with raisoning, like Gemini-Pro) use to define witch tool should use.
- To define a tool ==>
```sh
from langchain_core.tools import tool
@tool("get_weather", description="Get the weather of a city")
def get_weather(city: str):
    response = requests.get(f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1")
    data = response.json()
    latitude = data["results"][0]["latitude"]
    longitude = data["results"][0]["longitude"]
    response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true")
    data = response.json()
    response = f"The weather in {city} is {data["current_weather"]["temperature"]}C with {data["current_weather"]["windspeed"]}km/h of wind."
    return response
```
- With this ==>
```sh
llm_with_tools = llm.bind_tools([get_products, get_weather])
response = llm_with_tools.invoke(messages)
response.tool_calls
```
- We get ==>
```sh
[{'name': 'get_weather',
  'args': {'city': 'Rome'},
  'id': 'e99892ea-b023-4c08-b32e-2390c111767e',
  'type': 'tool_call'}]
```
- Note the LLM __understand "capital of Italy"__ to use __"Rome"__
- Also 'get_weather' means that the agent understand he should __use the tool__ "get_weather"


## React Booking
- In the agensts I add react.py just to have the example in the last notebook.
- Now add support_agent new folder "booking". There teh files "prompt" "tools" and "node".
- The node is the agent itself with only one basic mission. Booking and appointment. The tools book_appointment and get_appointment_availability one use to check the availability of the appointment and the other just to book the appointment.
- The LLM will know witch tool use for andwer the conversation.

## Routing Agent Decision
- In notebook the example to have a ==> route_edge (the "routing"). 
- This def will return from "where go" (in this case is randomic). Go to node_2 or go to node_3.
- To make this work we add in the build this ==> 
```sh
builder.add_conditional_edges('node_1', route_edge)
```
- Just as example check we could even start with the conditionals to continue to node 2 or 3. (see the las example in the notebook)
- Now we will add to the support_agent the booking_node (the same we already have using ReAct). In the graph we will see this booking_node as a subgraph because itself has his own graph.
- In the folder routes new folder "intention" here is the logic to define where continue de flow. The logic in the route.py (kind of equivalents as folder nodes.)
- Now we will add to the support_agent the "intention_route" ==> is the DEF in the route.py.
```sh
builder.add_conditional_edges('extractor', intention_route)
```
- *I commented the llm = llm_openai.bind_tools(tools) in conversation_moment because is getting infor from a Vector storeage in Open AI we don't need.*
- The support_agent is working !! Take the decition if use the conversation or the booking.

## Parallel
- In the notebook the example how to send the flow to 2 or 3 node in parallel way.
- You can see in the coder_checker.py a simple agent that will send the same "code" to analize it for 2 differents nodes in parallel. Then both answer return to the "condenser_data" node and return the response.

## Orchestrator
- In notebook just a def wtha random choose. In real life is a node that will decide witch nodes will continue using.
- In the agent orchestartor.py we add ==>
```sh
from langgraph.types import Send
```
- This "Send" is used in the return of the "assign_nodes" function (the same as random above). To define witch nodes will be used.

## Evaluator Optimizer
- In the evaluator_optimizer_joke.py the flow is easy but VERY interesting. 
- First node (generator_node) create a joke BUT depending on the state has 2 differents orderto to create the joke. 
- Second node (evaluator_node) check if the joke if funny or not using the system prompt "guidelines". And if not funny send a feedback to improve the joke.
- The Thrid node route_edge will check if is funny or not. If funny send to END if not funny send again to the first node. Now the first node has also feedback from the first one to "make a better joke". (becareful!!) This process could be an infinite loop if the jokes never is good enought for the 2nd node.

## FastAPi Docker Postgress
- Install ==> langgraph-checkpoint-postgres
```sh
uv add langgraph-checkpoint-postgres
```
- In main.py a fastapi basic.
- In main-2.py fastapi with DB and more
- add a Docker to see the postgre DB
---
### IMPORTANT
- __In support_agent the agent must be create in a dynamic way.__
- In support_agent.py compile the agent inside a function ==> 
```sh
def make_the_agent(config: TypedDict):
    checkpointer = config.get("checkpointer", None)
    builder = StateGraph(State)
    builder.add_node("extractor", extractor)
    builder.add_node("conversation_moment", conversation_moment)
    builder.add_node("booking_node", booking_node)

    builder.add_edge(START, 'extractor')
    builder.add_conditional_edges('extractor', intention_route)
    # builder.add_edge('extractor', 'conversation_moment')
    builder.add_edge('conversation_moment', END)

    agent = builder.compile(checkpointer=checkpointer)
    
    return agent
```
- No in the langgraph.json just chancg ":agent" for ":make_the_agent" (the new funtion)
- Now in /api a new db.py file to connect DB. 
- Here with the @asynccontextmanager def lifestan() says before connect to a DB (FUNDAMENTAL and IMPORTANT)
- The get_checkpointer return the instance of checkpointier if exist. (Otherwise send error).
- All in a dependency "CheckpointerDep" of fastAPI.
- In main-2.py with 
```sh
from api.db import lifespan, CheckpointerDep

app = FastAPI(lifespan=lifespan)
``` 
- The @app.post("/chat/{chat_id}") now has a dependency (the checkpointer:CheckpointerDep)