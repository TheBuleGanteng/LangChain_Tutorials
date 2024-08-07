import os
import fitz  # Imports pdfs
from langchain_openai import ChatOpenAI
import bs4
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv  # Not in tutorial: added to use gitignored .env file
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage

# Sets path to .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '..', 'gitignored', '.env')
load_dotenv(dotenv_path)

# Ensure that environment variables are set
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
AICHAT_LANGCHAIN_PROJECT = os.getenv('AICHAT_LANGCHAIN_PROJECT')

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = LANGCHAIN_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['AICHAT_LANGCHAIN_PROJECT'] = AICHAT_LANGCHAIN_PROJECT

# Model set
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
source_cutoff = 1

class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


# Below is the session history object introduced in lctest2.py
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# Below, 'k' represents the number of prior messages used as context
def filter_messages(messages, k=20):
    return messages[-k:]



# Step 1: Upload the documents    
docs = []
#print(f'running aichat_chat/helpers, generate_response() ... docs is: {docs}')

# 1b. Load documents from the 'docs_for_upload' directory
directory = os.path.join(os.path.dirname(__file__), '.', 'docs_for_upload')
for filename in os.listdir(directory):
    if filename.endswith('.pdf'):
        file_path = os.path.join(directory, filename)
        document = fitz.open(file_path)
        for page_num in range(document.page_count):  # This makes each page in each pdf a document, allowing the metadata to include both the file name and the page number.
            page = document.load_page(page_num)
            text = page.get_text()
            docs.append(Document(
                page_content=text,
                metadata={'source': filename, 'page_number': page_num + 1}  # The +1 addresses zero indexing
            ))



# Step 2: Chunk the data
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True  # Splits the docs into chunks of 1000 characters with 200 characters of overlap between chunks. The overlap helps mitigate the possibility of separating a statement from important context related to it.
)
splits = text_splitter.split_documents(docs)



class CustomOpenAIEmbeddings(OpenAIEmbeddings):
    def embed_query(self, query):
        if isinstance(query, dict):
            query = query['user_input']
        #print(f"embed_query input: {query}")
        result = super().embed_query(query)
        #print(f"embed_query result: {result}")
        return result

    def embed_documents(self, documents):
        #print(f"embed_documents input: {documents}")
        result = super().embed_documents(documents)
        #print(f"embed_documents result: {result}")
        return result

# Use CustomOpenAIEmbeddings instead of OpenAIEmbeddings
vectorstore = Chroma.from_documents(documents=splits, embedding=CustomOpenAIEmbeddings())



#vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()



# Step 3. Incorporate the retriever into a question-answering chain.
# The system_prompt is a static template that provides initial instructions to the language model. It sets the context and expectations for the assistant's behavior. This prompt typically includes guidelines on how the assistant should respond, what tone to use, and specific phrases to include based on the context provided. In this case, it includes instructions on how to handle retrieved context and general knowledge responses.
system_prompt = (
    "You are an assistant for question-answering tasks. "
    #"Whenever possible, use context answer the question,"
    #"If your information comes from retrieved context, include the phrase 'According to the documents listed below,' "
    #"If your information does not come from retrieved context, include the phrase 'While the uploaded documents do not contain this information, according to my general knowledge,' "
    "Keep your answer concise and limited to three sentences maximum."
    "\n\n"
    "{context}"
)


# The ChatPromptTemplate (prompt) is more dynamic and incorporates multiple elements, including the system_prompt and placeholders for actual user inputs and messages. The ChatPromptTemplate is constructed from a list of messages, where each message can be a system message (like the system_prompt), human message (user inputs), or placeholders for message history.
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ('human', '{user_input}')
    ]
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
#print(f"Messages provided to ChatPromptTemplate: {messages}")

class CustomRunnablePassthrough(RunnablePassthrough):
    def invoke(self, input, *args, **kwargs):
        #print(f"CustomRunnablePassthrough input: {input}")
        if isinstance(input, HumanMessage):
            input = input.content
        elif isinstance(input, dict) and 'messages' in input:
            input = input['messages']
        #print(f"Processed input: {input}")
        result = super().invoke(input, *args, **kwargs)
        #print(f"CustomRunnablePassthrough result: {result}")
        return result


# Ensure that all required inputs are correctly passed into the chain
rag_chain = (
    {"context": retriever, "messages": CustomRunnablePassthrough(), "user_input": CustomRunnablePassthrough()} # The retriever is responsible for finding and returning relevant documents based on the input query. In your code, this is implemented using Chroma and OpenAIEmbeddings to create a vector store and retriever.
    | chat_prompt # Defines the structure and content of the prompt that will be sent to the language model. It includes both static instructions (system_prompt) and dynamic placeholders for messages.
    | llm
    | StrOutputParser() # plucks the string content out of the LLM's output message
)


filtered_messages = filter_messages(messages)
user_input = "What is my name?"
#print(f'filtered_messages is: { filtered_messages }')


# Invoke the chain with the input data
try:
    result = rag_chain.invoke({
        "messages": filtered_messages,
        "user_input": user_input
    })
except Exception as e:
    print(f"Error occurred: {e}")
    raise


# Print the answer
print('Answer: \n' + result + '\n')

"""
# Extract and print the sources
sources = result['context']
source_list = []
for doc in sources:
    source = f'{ doc.metadata["source"] }, page: { doc.metadata["page_number"] }'
    source_list.append(source)
source_list = source_list[source_cutoff:]
print('Sources: \n' + '\n'.join(source_list) + '\n')
"""