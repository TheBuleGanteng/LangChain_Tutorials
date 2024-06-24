# From LangChain tutorial: https://python.langchain.com/v0.2/docs/tutorials/chatbot/#prompt-templates

import os
from dotenv import load_dotenv # Not in tutorial: added to use gitignored .env file
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Not in tutorial: Sets path to .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '..', 'gitignored', '.env')
load_dotenv(dotenv_path)


OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY


model = ChatOpenAI(model="gpt-3.5-turbo")


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | model

# Note we are now passing in a dictionary with a messages key where that contains a list of messages.
response = chain.invoke({"messages": [HumanMessage(content="hi! I'm bob")]})

print("response 1: \n"+response.content)


# Below is the session history object introduced in lctest2.py
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# Wrap messages history object above
with_message_history = RunnableWithMessageHistory(chain, get_session_history)

# Set a session history
config = {"configurable": {"session_id": "abc5"}}

response = with_message_history.invoke(
    [HumanMessage(content="Hi! I'm Jim")],
    config=config,
)

print("response 2: \n"+response.content)


response = with_message_history.invoke(
    [HumanMessage(content="What's my name?")],
    config=config,
)

print("response 3: \n"+response.content)



prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | model

# Now, pass in the value for language
response = chain.invoke(
    {"messages": [HumanMessage(content="hi! I'm bob")], "language": "Spanish"}
)

print("response 4: \n"+response.content)



with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)

config = {"configurable": {"session_id": "abc11"}}

response = with_message_history.invoke(
    {"messages": [HumanMessage(content="hi! I'm todd")], "language": "Spanish"},
    config=config,
)

print('response 5: \n'+response.content)

response = with_message_history.invoke(
    {"messages": [HumanMessage(content="whats my name?")], "language": "Spanish"},
    config=config,
)

print('response 6: \n'+response.content)