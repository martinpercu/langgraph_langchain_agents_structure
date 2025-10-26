from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from typing import Literal
import random
from langgraph.types import Send

class State(MessagesState):
    nodes: list[str]


def orchestrator(state: State):
    nodes = ["node_a", "node_b", "node_c"]
    nodes = random.sample(nodes, random.randint(1, 3))
    return {"nodes": nodes}


def node_a(state: State):
    return state


def node_b(state: State):
    return state


def node_c(state: State):
    return state


def aggregator_condenser_data(state: State):
    return state


def assign_nodes(state: State) -> Literal["node_a", "node_b", "node_c"]:
    nodes = state['nodes']
    # return [Send('node_a', {}), Send('node_b', {}), Send('node_c', {})]
    return [Send(n, {}) for n in nodes]



builder = StateGraph(State)

builder.add_node('orchestrator', orchestrator)
builder.add_node('node_a', node_a)
builder.add_node('node_b', node_b)
builder.add_node('node_c', node_c)
builder.add_node('aggregator_condenser_data', aggregator_condenser_data)

builder.add_edge(START, 'orchestrator')
builder.add_conditional_edges("orchestrator", assign_nodes)
builder.add_edge('node_a', 'aggregator_condenser_data')
builder.add_edge('node_b', 'aggregator_condenser_data')
builder.add_edge('node_c', 'aggregator_condenser_data')
builder.add_edge('aggregator_condenser_data', END)
agent = builder.compile()





