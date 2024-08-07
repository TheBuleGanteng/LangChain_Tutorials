# From LangChain tutorials: 
# https://python.langchain.com/v0.2/docs/how_to/message_history/#customization
import os
import fitz # Imports pdfs
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv  # Not in tutorial: added to use gitignored .env file

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import ConfigurableFieldSpec


# Not in tutorial: Sets path to .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '..', 'gitignored', '.env')
load_dotenv(dotenv_path)

# Ensure that environment variables are set
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
AICHAT_LANGCHAIN_PROJECT = os.getenv('AICHAT_LANGCHAIN_PROJECT')

os.environ['LANGCHAIN_TRACING_V2'] ='true'
os.environ['LANGCHAIN_ENDPOINT'] ='https://api.smith.langchain.com'
os.environ['AICHAT_LANGCHAIN_PROJECT'] = AICHAT_LANGCHAIN_PROJECT
os.environ['LANGCHAIN_API_KEY'] = LANGCHAIN_API_KEY 
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

model = ChatOpenAI(model="gpt-3.5-turbo")




def get_session_history(user_id: str, conversation_id: str):
    return SQLChatMessageHistory(f"{user_id}--{conversation_id}", "sqlite:///memory.db")



prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're an assistant who speaks in {language}. Respond in 20 words or fewer",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)



runnable = prompt | model


with_message_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="User ID",
            description="Unique identifier for the user.",
            default="",
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id="conversation_id",
            annotation=str,
            name="Conversation ID",
            description="Unique identifier for the conversation.",
            default="",
            is_shared=True,
        ),
    ],
)


response1 = with_message_history.invoke(
    {"language": "italian", "input": "hi im bob!"},
    config={"configurable": {"user_id": "123", "conversation_id": "1"}},
)

response2 = with_message_history.invoke(
    {"language": "italian", "input": "whats my name?"},
    config={"configurable": {"user_id": "123", "conversation_id": "1"}},
)


print('\n')
print(f'Resonse1: { response1.content }')
print('\n')
print(f'Resonse2: { response2.content }')
print('\n')
