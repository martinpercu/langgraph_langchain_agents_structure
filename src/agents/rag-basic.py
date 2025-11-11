
from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage
from langchain.chat_models import init_chat_model
import random

llm_openai = init_chat_model("openai:gpt-4o-mini", temperature=0.5)

file_search_tool = {
    "type": "file_search",
    "vector_store_ids": ["vs_67bdae7312d081919021ad0dc0c7ef96"],
}
llm = llm_openai.bind_tools([file_search_tool])


# Ths is the basic dictorio the "shared memory" for agents

class State(MessagesState):
    customer_name: str
    my_age: int


def node_1(state: State):
    new_state: State = {}
    # if state.get("customer_name") is None:
    #     new_state["customer_name"] = "Johny Dolly"
    # else:
    #     new_state["my_age"] = random.randint(18, 48)
    
    history = state["messages"]
    last_message = history[-1]
    ai_message = llm.invoke(last_message.text)
    new_state["messages"] = [ai_message]
    return new_state
    


from langgraph.graph import StateGraph, START, END

builder = StateGraph(State)
builder.add_node("node_1", node_1)

builder.add_edge(START, 'node_1')
builder.add_edge('node_1', END)

agent = builder.compile()