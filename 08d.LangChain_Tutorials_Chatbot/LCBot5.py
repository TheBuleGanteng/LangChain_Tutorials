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
os.environ['AICHAT_LANGCHAIN_PROJECT'] = AICHAT_LANGCHAIN_PROJECT
os.environ['LANGCHAIN_API_KEY'] = LANGCHAIN_API_KEY 
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

model = ChatOpenAI(model='gpt-3.5-turbo')

# get_session_history populates the store variable with all the messages in a convo, creating a pool of messages from which we can draw context
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# Below, 'k' represents the number of prior messages used as context for a given response
# If this is not set, then the number of messages used as context will grow with each subsequent message and will overflow the LLM's context window.
def filter_messages(messages, k=10):
    return messages[-k:]


# This is our prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            'You are a helpful assistant. Answer all questions to the best of your ability.',
        ),
        MessagesPlaceholder(variable_name='messages'),
    ]
)

# This takes the basic 'chain = prompt | model' approach and sets the number of previous messages to use as context, as set via the k value above.
chain = (
    RunnablePassthrough.assign(messages=lambda x: filter_messages(x['messages']))
    | prompt
    | model
)

# This sets the message history with the following properties:
# 1. The modified chain that sets the number of prior messages to use as context
# 2. The pool of prior messages stored in get_session_history
# 3. Sets the correct key as 'messages'
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='messages',
)




# Below, we use the .stream method so the user sees the response appear piece-by-piece as it's generated.
config = {'configurable': {'session_id': 'abc15'}}
for r in with_message_history.stream(
    {
        'messages': [HumanMessage(content="hi! I'm todd. tell me a joke")],
        'language': 'English',
    },
    config=config,
):
    print(r.content, end='|')
print('\n')


