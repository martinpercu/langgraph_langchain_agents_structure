from langchain.agents import create_agent

from agents.support_agent.nodes.booking.tools import tools
from agents.support_agent.nodes.booking.prompt import prompt_template


booking_node = create_agent(
    model="claude-haiku-4-5-20251001",
    tools=tools,
    system_prompt=prompt_template.format(),
)