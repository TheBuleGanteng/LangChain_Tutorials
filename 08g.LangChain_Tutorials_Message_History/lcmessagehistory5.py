# Adapted from LangChain tutorial: https://python.langchain.com/v0.2/docs/how_to/message_history/#customization
# The function below does the following:
# 1. successfully remembers the conversation
# 2. Sources answers from RAG'ed data provided in the designated folder
# 3. Streams the result
# The function below still needs to do the following:
# 1. Provide the sources in <document name>, <page number> format

import os
import fitz  # Imports pdfs
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv  # Not in tutorial: added to use gitignored .env file

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import ConfigurableFieldSpec

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from sqlalchemy import create_engine
import time

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

model = ChatOpenAI(model="gpt-3.5-turbo", streaming=True)  # Enable streaming 

# Create the SQLAlchemy engine
engine = create_engine("sqlite:///memory.db")

def get_session_history(user_id: str, conversation_id: str):
    # Pass the engine directly
    return SQLChatMessageHistory(f"{user_id}--{conversation_id}", connection=engine)


class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

# Step 1: Upload the documents
docs = []
directory = os.path.join(os.path.dirname(__file__), '.', 'docs_for_upload')
for filename in os.listdir(directory):
    if filename.endswith('.pdf'):
        file_path = os.path.join(directory, filename)
        document = fitz.open(file_path)
        for page_num in range(document.page_count):
            page = document.load_page(page_num)
            text = page.get_text()  
            docs.append(Document(
                page_content=text,
                metadata={'source': filename, 'page_number': page_num + 1}
            ))

# Step 2: Chunk the data
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

# Step 3: Embed all the vector splits
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

# Step 4: Set up a retriever that finds and pulls the relevant data form the vectorstore in response to a user question
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're an assistant who speaks in {language}. Respond in 3 sentences or fewer. Use the following context if relevant: {context}",
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

# Function to retrieve context and invoke the chain
def get_response(input_text, user_id, conversation_id):
    retrieved_docs = retriever.invoke(input_text)
    context = " ".join([doc.page_content for doc in retrieved_docs])
    answer_chunks = [] # Initiate a list to hold the chunks of a given response
    for chunk in with_message_history.stream(
        {
            "language": "english",
            "input": input_text,
            "context": context},
            config={"configurable": {"user_id": user_id, "conversation_id": conversation_id}
        },
    ):
        print(chunk.content, end='', flush=True)
        time.sleep(0.1)  # Adjust the delay as needed

    
print('\nResponse1:\n')
response1 = get_response("hi im bob!", "123", "1")
print('\n')

print('\nResponse2:\n')
response2 = get_response("whats my name?", "123", "1")
print('\n')

print('\nResponse3:\n')
response3 = get_response("What is the best selling beer brand in Vietnam?", "123", "1")
print('\n')

print('\nResponse4:\n')
response4 = get_response("What is the best selling beer brand in Mexico?", "123", "1")    
print('\n')