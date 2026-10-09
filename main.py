import uvicorn
from fastapi import FastAPI
from openai import BaseModel
from starlette.middleware.cors import CORSMiddleware

from Isaac_Langchain import Isaac

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,           # Allowed domains
    allow_credentials=True,          # Support cookies/auth headers
    allow_methods=["*"],             # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],             # Allow all headers
)
isaac = Isaac

class Chat(BaseModel):
    user_input: str

@app.post("/isaac")
def chat_isaac(message: Chat):
    input_message = message.user_input
    return isaac.call_isaac(input_message)
