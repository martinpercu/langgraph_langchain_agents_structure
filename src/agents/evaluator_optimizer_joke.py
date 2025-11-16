from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from typing import Literal
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

class State(TypedDict):
    joke: str
    topic: str
    feedback: str
    is_funny: str

class Feedback(BaseModel):
    is_funny: bool = Field(
        description="Decide if the joke is funny or not. Return True if it is, False otherwise.",
    )
    feedback: str = Field(
        description="If the joke is not funny, provide feedback on how to improve it.",
    )
    
    


llm_nano = init_chat_model("openai:gpt-5-nano", temperature=0)
llm_flash = init_chat_model("google_genai:gemini-2.5-flash", temperature=0)
llm_gpt = init_chat_model("openai:gpt-4o-mini", temperature=0.8)

## This fails
# llm_evaluator = llm_flash.with_structured_output(Feedback, temperature=0)

# GOOGLE NOT ACCEPT temperature in the "with_structured_output"
# So use ".bind"
llm_evaluator = llm_flash.bind(temperature=0).with_structured_output(Feedback)


# ## Open AI accept temperature in the "with_structured_output"
# llm = init_chat_model("openai:openai:gpt-4o-mini", temperature=0)
# ll_evaluator = llm.with_structured_output(Feedback, temperature=0)



SYSTEM_PROMPT = """
A funny joke should:
- be more than 2 paragraphs and never more than 4. 
- has an effective punchline. 
- should be quite smart.
"""


def generator_node(state: State):
    feedback = state.get("feedback", None)
    topic = state.get("topic", None)
    if feedback:
        msg = llm_gpt.invoke(f"Write a joke about {topic} but take into account the feedback: {feedback} respond in Spanish")
    else:
        msg = llm_gpt.invoke(f"Write a joke about {topic} respond in Spanish")
    return {"joke": msg.text}


def evaluator_node(state: State):
    joke = state.get("joke", None)
    schema = llm_evaluator.invoke(f"Grade the joke {joke} with the following prompt: {SYSTEM_PROMPT}")
    return {"is_funny": schema.is_funny, "feedback": schema.feedback}


def route_edge(state: State) -> Literal[END, "generator_node"]:
    is_funny = state.get("is_funny", True)
    if is_funny:
        return END
    else:
        return "generator_node"


builder = StateGraph(State)

builder.add_node('generator_node', generator_node)
builder.add_node('evaluator_node', evaluator_node)

builder.add_edge(START, 'generator_node')
builder.add_edge('generator_node', 'evaluator_node')
builder.add_conditional_edges('evaluator_node', route_edge)
agent = builder.compile()