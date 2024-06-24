
# Adapted from LangChain tutorial: https://python.langchain.com/v0.2/docs/how_to/message_history/#customization
# The function below does the following:
# 2. Sources answers from RAG'ed data provided in the designated folder
# 3. Streams the result
# 4. Provide the sources in <document name>, <page number> format
# 5. successfully remembers the conversation - watch this b/c works inconsistently
# 6. Include logic that omits sources if the information is not from the sources - watch this b/c works inconsistently


# The function below still needs to do the following:
# 1. Limit the number of prior messages used, so as to avoid overflowing the prompt


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

model = ChatOpenAI(model="gpt-3.5-turbo", streaming=True, temperature=0.9)  # Enable streaming 
source_cutoff = 1 # Used to ignore the first source, which seems to improve results
context_available_phrase = 'According to the documents listed below,'
#context_available_instructions = f'Use prior messages in this conversation and context to answer your question. If your answer comes from context (but not from prior messages), begin your answer with the following phrase: "{context_available_phrase}". Otherwise, follow system instructions. '
context_available_instructions = f'Use prior messages in this conversation and context to answer your question. For each response, if the answer was sourced from the uploaded documents, then your answer should begin with the following phrase: { context_available_phrase }. Otherwise, follow system instructions.'


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


system_prompt = (
    "You're an assistant who speaks in {language}." 
    "Respond in 3 sentences or fewer." 
    "Use the following context if relevant: {context}"
    "If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
)



prompt = ChatPromptTemplate.from_messages(
    [
        
        ("system", system_prompt),
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
    answer_chunks = []  # Initiate a list to hold the chunks of a given response
    sources = []
    for chunk in with_message_history.stream(
        {
            "language": "english",
            "input": context_available_instructions + input_text,
            "context": context
        },
        config={"configurable": {"user_id": user_id, "conversation_id": conversation_id}},
    ):
        print(chunk.content, end='', flush=True)
        answer_chunks.append(chunk.content)
        time.sleep(0.1)  # Adjust the delay as needed
    
    # Check if answer used the sources via the presence of context_available_phrase
    answer_chunks = "".join(answer_chunks)
    if context_available_phrase in answer_chunks:
        # Affix the source for all the chunks
        sources = [f"{doc.metadata['source']}, page {doc.metadata['page_number']}" for doc in retrieved_docs]
        # De-duplicate sources
        deduplicated_sources = []
        for source in sources:
            if source not in deduplicated_sources:
                deduplicated_sources.append(source)
        # Print a list of sources
        print('\n\n')
        print('Sources:')
        for source in deduplicated_sources[source_cutoff:]:
            print(source)

    
print('\nResponse1:\n')    
response1 = get_response("Hi, my name is Bob", "123", "1")
print('\n')

print('\nResponse2:\n')
response2 = get_response("What is my name?", "123", "1")
print('\n')

print('\nResponse3:\n')
response3 = get_response("What is the best selling beer brand in Vietnam?", "123", "1")
print('\n')

print('\nResponse4:\n')
response4 = get_response("What is the best selling beer brand in Mexico?", "123", "1")    
print('\n')

print('\nResponse5:\n')
response4 = get_response("What is the second best selling beer brand in Vietnam?", "123", "1")    
print('\n')

print('\nResponse6:\n')
response6 = get_response("What was my last question?", "123", "1")
print('\n')
