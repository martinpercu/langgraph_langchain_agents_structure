
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
    phone: str
    my_age: int
    


from pydantic import BaseModel, Field

class UserInfo(BaseModel):
    """Contact information for a user."""
    name: str = Field(description="The name of the user")
    email: str = Field(description="The email address of the user")
    phone: str = Field(description="The phone number of the user")
    age: str = Field(description="The age of the user")
    sentiment: str = Field(description="The sentiment conversation of the user")

llm_2 = init_chat_model("anthropic:claude-haiku-4-5-20251001", temperature=0)
llm_with_structured_output  = llm_2.with_structured_output(schema=UserInfo)
    


def extractor(state:State):
    history = state["messages"]
    customer_name = state.get("customer_name", None)
    new_state: State = {}
    if customer_name is None or len(history) >= 20:
        schema = llm_with_structured_output.invoke(history)
        new_state["customer_name"] = schema.name
        new_state["phone"] = schema.phone
        new_state["my_age"] = schema.age
    return new_state

def conversation_moment(state: State):
    new_state: State = {}
    history = state["messages"]
    last_message = history[-1]
    print(last_message)
    customer_name = state.get("customer_name", 'John Doe')
    system_message = f"You are a helpful assistant that can answer questions about the customer {customer_name}"
    ai_message = llm.invoke([("system", system_message), ("user", last_message.text)])
    new_state["messages"] = [ai_message]
    return new_state
    


from langgraph.graph import StateGraph, START, END

builder = StateGraph(State)
builder.add_node("conversation_moment", conversation_moment)
builder.add_node("extractor", extractor)

builder.add_edge(START, 'extractor')
builder.add_edge('extractor', 'conversation_moment')
builder.add_edge('conversation_moment', END)

agent = builder.compile()