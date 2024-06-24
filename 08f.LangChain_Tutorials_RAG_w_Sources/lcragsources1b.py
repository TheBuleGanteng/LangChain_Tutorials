# From LangChain tutorials: 
# https://python.langchain.com/v0.2/docs/how_to/qa_sources/

import os
import fitz # Imports pdfs
from langchain_openai import ChatOpenAI
import bs4
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv  # Not in tutorial: added to use gitignored .env file

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder



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

llm = ChatOpenAI(model="gpt-3.5-turbo")



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







# Step 1: Upload the documents    
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()


# Step 2: Chunk the data
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200, 
    add_start_index=True  # Splits the docs into chunks of 1000 characters with 200 characters of overlap between chunks. The overlap helps mitigate the possibility of separating a statement from important context related to it.
)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()



# Step 3. Incorporate the retriever into a question-answering chain.
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
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


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
        ("assistant", messages),
    ]
)


question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

result = rag_chain.invoke({"input": "What is my name?"})

print('Answer: \n'+result['answer']+'\n')
print('Sources: \n'+str(result['context'])+'\n')