# From LangChain tutorial: https://python.langchain.com/v0.2/docs/tutorials/chatbot/#managing-conversation-history

import os
from dotenv import load_dotenv # Not in tutorial: added to use gitignored .env file
import logging # Not in tutorial: added to suppress info messages from openAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough


# Not in tutorial: Sets logging level for langchain to WARNING to suppress INFO messages
logging.getLogger('langchain').setLevel(logging.ERROR)

# Not in tutorial: Sets path to .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '..', 'gitignored', '.env')
load_dotenv(dotenv_path)


# Ensure that environment variables are set
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
AICHAT_LANGCHAIN_PROJECT = os.getenv('AICHAT_LANGCHAIN_PROJECT')

os.environ['LANGCHAIN_TRACING_V2'] ='true'
os.environ['LANGCHAIN_ENDPOINT'] ='https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = LANGCHAIN_API_KEY 
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['AICHAT_LANGCHAIN_PROJECT'] = AICHAT_LANGCHAIN_PROJECT

model = ChatOpenAI(model="gpt-3.5-turbo")

# Below is the session history object introduced in lctest2.py
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# Below, 'k' represents the number of prior messages used as context for a given response
# If this is not set, then the number of messages used as context will grow with each subsequent message and will overflow the LLM's context window.
def filter_messages(messages, k=10):
    return messages[-k:]


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = (
    RunnablePassthrough.assign(messages=lambda x: filter_messages(x["messages"]))
    | prompt
    | model
)

messages = [
    HumanMessage(content="hi! I'm bob"), # Msg 1
    AIMessage(content="hi!"), # Msg 2
    HumanMessage(content="I like vanilla ice cream"), # Msg 3
    AIMessage(content="nice"), # Msg 4
    HumanMessage(content="whats 2 + 2"), # Msg 5
    AIMessage(content="4"), # Msg 6
    HumanMessage(content="thanks"), # Msg 7
    AIMessage(content="no problem!"), # Msg 8
    HumanMessage(content="having fun?"), # Msg 9
    AIMessage(content="yes!"), # Msg 10
]


response1 = chain.invoke(
    {
        "messages": messages + [HumanMessage(content="what's my name?")],
        "language": "English",
    }
)


response2 = chain.invoke(
    {
        "messages": messages + [HumanMessage(content="what's my fav ice cream?")],
        "language": "English",
    }
)


print('Response 1: \n'+response1.content+'\n') # Wrong response bc outside k window of 10 messages 
print('Response 2: \n'+response2.content+'\n') # Correct response bc within k window of 10 messages



with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)

config = {"configurable": {"session_id": "abc20"}}

response3 = with_message_history.invoke(
    {
        "messages": messages + [HumanMessage(content="whats my name?")],
        "language": "English",
    },
    config=config
)

print('Response 3: \n'+response3.content+'\n')


response4 = with_message_history.invoke(
    {
        "messages": [HumanMessage(content="whats my favorite ice cream?")],
        "language": "English",
    },
    config=config
)

print('Response 4: \n'+response4.content+'\n')