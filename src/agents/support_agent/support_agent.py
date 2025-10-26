from langgraph.graph import StateGraph, START, END
from typing import TypedDict

from agents.support_agent.state import State
from agents.support_agent.nodes.conversation.node import conversation_moment
from agents.support_agent.nodes.extractor.node import extractor
from agents.support_agent.nodes.booking.node import booking_node
from agents.support_agent.routes.intention.route import intention_route


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


# builder = StateGraph(State)
# builder.add_node("extractor", extractor)
# builder.add_node("conversation_moment", conversation_moment)
# builder.add_node("booking_node", booking_node)

# builder.add_edge(START, 'extractor')
# builder.add_conditional_edges('extractor', intention_route)
# # builder.add_edge('extractor', 'conversation_moment')
# builder.add_edge('conversation_moment', END)

# agent = builder.compile()