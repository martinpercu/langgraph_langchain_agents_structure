from pydantic import BaseModel, Field
from typing import Literal
from langchain.chat_models import init_chat_model

from agents.support_agent.state import State
from agents.support_agent.routes.intention.prompt import SYSTEM_PROMPT

class RouteIntention(BaseModel):
    step: Literal["conversation_moment", "booking_node"] = Field(
        'conversation_moment', description="The next step in the routing process"
    )

llm = init_chat_model("openai:gpt-4o-mini", temperature=0.2)
llm = llm.with_structured_output(schema=RouteIntention)




def intention_route(state: State) -> Literal["conversation_moment", "booking_node"]:
    history = state["messages"]
    print('*'*100)
    print(history)
    print('*'*100)
    schema = llm.invoke([("system", SYSTEM_PROMPT)] + history)
    print('-*-'*40)
    print(schema)
    print('-*-'*40)
    if schema.step is not None:
        return schema.step
    return 'conversation_moment'

