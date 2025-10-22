from agents.support_agent.state import State
from langchain.chat_models import init_chat_model

from pydantic import BaseModel, Field

from agents.support_agent.nodes.extractor.prompt import SYSTEM_PROMPT


class UserInfo(BaseModel):
    """Contact information for a user."""
    name: str = Field(description="The name of the user")
    email: str = Field(description="The email address of the user")
    phone: str = Field(description="The phone number of the user")
    age: str = Field(description="The age of the user")
    sentiment: str = Field(description="The sentiment conversation of the user")


llm_2 = init_chat_model("anthropic:claude-haiku-4-5-20251001", temperature=0)
llm_2_with_structured_output  = llm_2.with_structured_output(schema=UserInfo)

def extractor(state:State):
    history = state["messages"]
    customer_name = state.get("customer_name", None)
    new_state: State = {}
    if customer_name is None or len(history) >= 20:
        schema = llm_2_with_structured_output.invoke([("system", SYSTEM_PROMPT)] + history)
        new_state["customer_name"] = schema.name
        new_state["phone"] = schema.phone
        new_state["my_age"] = schema.age
    return new_state