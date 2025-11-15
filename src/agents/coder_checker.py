from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

llm_nano = init_chat_model("openai:gpt-5-nano")
llm_flash = init_chat_model("google_genai:gemini-2.5-flash")
llm_gpt_5_mini = init_chat_model("openai:gpt-5-mini")
## google_genai:gemini-2.5-flash

class SecurityReview(BaseModel):
    vulnerabilities: list[str] = Field(description="The vulnerabilities in the code", default=None)
    riskLevel: str = Field(description="The risk level of the vulnerabilities", default=None)
    suggestions: list[str] = Field(description="The suggestions for fixing the vulnerabilities", default=None)


class MaintainabilityReview(BaseModel):
    concerns: list[str] = Field(description="The concerns about the code", default=None)
    qualityScore: int = Field(description="The quality score of the code from 1 to 10", default=None, ge=1, le=10)
    recommendations: list[str] = Field(description="The recommendations for improving the code", default=None)


class State(TypedDict):
    code: str
    security_check: SecurityReview
    maintainability_check: MaintainabilityReview
    final_review: str


def maintainability_check(state: State):
    code = state['code']
    messages = [
        ("system", "You are an expert in code quality. Focus on code structure, readability, and adherence to best practices."),
        ("user", f"Review this code: {code}")
    ]
    llm_with_structured_output = llm_flash.with_structured_output(MaintainabilityReview)
    schema = llm_with_structured_output.invoke(messages)
    return {
        'maintainability_check': schema
    }

def security_check(state: State):
    code = state['code']
    messages = [
        ("system", "You are an expert in code security. Focus on identifying security vulnerabilities, injection risks, and authentication issues."),
        ("user", f"Review this code: {code}")
    ]
    llm_with_structured_output = llm_nano.with_structured_output(SecurityReview)
    schema = llm_with_structured_output.invoke(messages)
    return {
        'security_check': schema
    }


def condenser_data(state: State):
    security_check = state['security_check']
    maintainability_check = state['maintainability_check']
    messages = [
        ("system", "You are a technical lead summarizing multiple code reviews"),
        ("user", f"Synthesize these code review results into a concise summary with key actions: Security review: {security_check} and Maintainability review: {maintainability_check}")
    ]
    response = llm_nano.invoke(messages)
    return {
        'final_review': response.text
    }


builder = StateGraph(State)

builder.add_node('security_check', security_check)
builder.add_node('maintainability_check', maintainability_check)
builder.add_node('condenser_data', condenser_data)

builder.add_edge(START, 'security_check')
builder.add_edge(START, 'maintainability_check')
builder.add_edge("security_check", "condenser_data")
builder.add_edge("maintainability_check", "condenser_data")
builder.add_edge('condenser_data', END)
agent = builder.compile()