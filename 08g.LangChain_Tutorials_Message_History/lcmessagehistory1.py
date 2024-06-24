# From LangChain tutorials: 
# https://python.langchain.com/v0.2/docs/how_to/qa_sources/

import os
import fitz # Imports pdfs
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv  # Not in tutorial: added to use gitignored .env file

from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory

# Not in tutorial: Sets path to .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '..', 'gitignored', '.env')
load_dotenv(dotenv_path)

# Ensure that environment variables are set
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LANGCHAIN_PROJECT = os.getenv('LANGCHAIN_PROJECT')

os.environ['LANGCHAIN_TRACING_V2'] ='true'
os.environ['LANGCHAIN_ENDPOINT'] ='https://api.smith.langchain.com'
os.environ['LANGCHAIN_PROJECT'] = LANGCHAIN_PROJECT
os.environ['LANGCHAIN_API_KEY'] = LANGCHAIN_API_KEY 
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

model = ChatOpenAI(model="gpt-3.5-turbo")


def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite:///memory.db")


runnable_with_history = RunnableWithMessageHistory(
    model,
    get_session_history,
)


response1 = runnable_with_history.invoke(
    [HumanMessage(content="hi - im bob!")],
    config={"configurable": {"session_id": "1"}},
)


response2 = runnable_with_history.invoke(
    [HumanMessage(content="whats my name?")],
    config={"configurable": {"session_id": "1"}},
)


response3 = runnable_with_history.invoke(
    [HumanMessage(content="whats my name?")],
    config={"configurable": {"session_id": "1a"}},
)


print('\n')
print(f'Resonse1: { response1.content }')
print('\n')
print(f'Resonse2: { response2.content }')
print('\n')
print(f'Resonse3: { response3.content }')
