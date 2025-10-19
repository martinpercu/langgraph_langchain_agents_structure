
from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage


# Ths is the basic dictorio the "shared memory" for agents

class State(MessagesState):
    customer_name: str
    my_age: int


def node_1(state: State):
    history = state["messages"]
    if state.get("customer_name") is None:
        return {
            "customer_name": "Johny Dolly"
        }
    else:
        ai_msg = AIMessage(content="Hi, what can I do for you?")
        return {
            "messages": [ai_msg]            
        }


from langgraph.graph import StateGraph, START, END

builder = StateGraph(State)
builder.add_node("node_1", node_1)

builder.add_edge(START, 'node_1')
builder.add_edge('node_1', END)

agent = builder.compile()