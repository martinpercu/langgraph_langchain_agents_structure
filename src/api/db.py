import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi import Depends
from typing import Annotated

from langgraph.checkpoint.postgres import PostgresSaver

# DB_URI = os.getenv("DB_URI")
DB_URI = "postgresql://postgres:postgres@localhost:5432/agent_test"

# Global checkpointer instance
_checkpointer: PostgresSaver | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _checkpointer
    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        _checkpointer = checkpointer
        _checkpointer.setup()
        yield
        
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     global _checkpointer
    
#     # Crear el checkpointer SIN context manager
#     _checkpointer = PostgresSaver.from_conn_string(DB_URI)
#     _checkpointer.setup()
    
#     try:
#         yield
#     finally:
#         # Cerrar la conexión al finalizar
#         if _checkpointer is not None:
#             _checkpointer.conn.close()
#             _checkpointer = None


def get_checkpointer() -> PostgresSaver:
    if _checkpointer is None:
        raise RuntimeError("Checkpointer not initialized. Make sure lifespan is running.")
    return _checkpointer

CheckpointerDep = Annotated[PostgresSaver, Depends(get_checkpointer)]