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

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")


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


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

result = rag_chain.invoke({"input": "What is my name?"})

print('Answer: \n'+result['answer']+'\n')
print('Sources: \n'+str(result['context'])+'\n')