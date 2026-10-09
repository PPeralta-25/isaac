from typing import Any, Callable

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts.chat import (SystemMessagePromptTemplate,HumanMessagePromptTemplate,ChatPromptTemplate,AIMessagePromptTemplate)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory

from typing import Callable
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
store = {}
class Isaac:

    def call_isaac(user_input:str):
        chat = ChatOpenAI(model_name = 'gpt-4o',
                              model_kwargs = {'seed':365},
                              temperature = 0,
                              max_tokens = 250)

        system_template = ''' Your name is ISAAC. Your name stands for Intuitive, 
                                Supportive AI Assistant Companion. Your mission is to 
                                cheer the users who are stress, tired, anxious or even depressed. Be friendly to the users.
                                You don't need to give them medical or psychological assistant. Help them to lift up their mood and give 
                                positivity. You can also provide a physical advice to remove their preoccupation and have play
                                with the user. Ignore anything non-related to your purpose. Ask user if the input is vague
                                or need to comprehend further to provide specific answer. Do only your purpose'''

        human_template = "{input}"
        message_system_template = SystemMessagePromptTemplate.from_template(template = system_template)
        message_human_template = HumanMessagePromptTemplate.from_template(template = human_template)
        message_place_holder = MessagesPlaceholder(variable_name = "history")

        chat_template = ChatPromptTemplate.from_messages([message_system_template,message_place_holder,message_human_template])

        chain = chat_template | chat | StrOutputParser()

        chain_with_history = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="history"
        )
        configs = {"configurable" : {"session_id" : "user_session"}}

        response = chain_with_history.invoke({"input" :user_input},
                                    config=configs)

        return response

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

history = get_session_history("1")