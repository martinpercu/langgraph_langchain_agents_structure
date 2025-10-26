from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel

from fastapi import FastAPI
from agents.support_agent.support_agent import make_the_agent

from langchain_core.messages import HumanMessage
from fastapi.responses import StreamingResponse
from api.db import lifespan, CheckpointerDep

app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {"Hello": "World"}


class Message(BaseModel):
    message: str



@app.post("/chat/{chat_id}")
async def chat(chat_id: str, item: Message, checkpointer: CheckpointerDep):
    # CRUD add message
    config = {
        "configurable": {
            "thread_id": chat_id,
        }
    }
    human_message = HumanMessage(content=item.message)
    agent = make_the_agent(config={"checkpointer": checkpointer})
    state = {"messages": [human_message]}
    response = agent.invoke(state, config)
    last_message = response["messages"][-1]
    # CRUD add message
    return last_message.text


@app.post("/chat/{chat_id}/stream")
async def stream_chat(chat_id: str, message: Message, checkpointer: CheckpointerDep):
    human_message = HumanMessage(content=message.message)
    async def generate_response():
        agent = make_the_agent(config={"checkpointer": checkpointer})
        for message_chunk, metadata in agent.stream({"messages": [human_message]}, stream_mode="messages"):
            if message_chunk.content:
                yield f"data: {message_chunk.content}\n\n"

        print(message_chunk.content, end="|", flush=True)

    return StreamingResponse(generate_response(), media_type="text/event-stream")