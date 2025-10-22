from agents.support_agent.state import State
from langchain.chat_models import init_chat_model

from agents.support_agent.nodes.conversation.tools import tools 
from agents.support_agent.nodes.conversation.prompt import SYSTEM_PROMPT 


llm_openai = init_chat_model("openai:gpt-4o-mini", temperature=0.5)
llm = llm_openai.bind_tools(tools)



def conversation_moment(state: State):
    new_state: State = {}
    history = state["messages"]
    last_message = history[-1]
    customer_name = state.get("customer_name", 'Johny Dollie')
    ai_message = llm.invoke([("system", SYSTEM_PROMPT), ("user", last_message.text)])
    new_state["messages"] = [ai_message]
    return new_state
    