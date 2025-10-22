from langgraph.graph import StateGraph, START, END

from agents.support_agent.state import State
from agents.support_agent.nodes.conversation.node import conversation_moment
from agents.support_agent.nodes.extractor.node import extractor

builder = StateGraph(State)
builder.add_node("conversation_moment", conversation_moment)
builder.add_node("extractor", extractor)

builder.add_edge(START, 'extractor')
builder.add_edge('extractor', 'conversation_moment')
builder.add_edge('conversation_moment', END)

agent = builder.compile()