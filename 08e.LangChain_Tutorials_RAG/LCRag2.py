# From LangChain tutorial: https://python.langchain.com/v0.2/docs/tutorials/rag/#indexing-split

import os
import getpass
import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv # Not in tutorial: added to use gitignored .env file
import logging # Not in tutorial: added to suppress info messages from openAI

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

#model = ChatOpenAI(model='gpt-3.5-turbo')


# Step 1: Load, chunk and index the contents of a website.
# Here, we use DocumentLoaders, which are objects that load in data from a source and return a list of Documents
# A Document is an object with some page_content (str) and metadata (dict).
# Only keep post title, headers, and content from the full HTML.
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

result1= len(docs[0].page_content)

print('Result 1: \n'+str(result1)+'\n')
print('Result2: \n'+docs[0].page_content[:500]+'\n')


# Step 2: Chunk the data
# Splits the documents into chunks of 1000 characters with 200 characters of overlap between chunks. The overlap helps mitigate the possibility of separating a statement from important context related to it. 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

result3 = len(all_splits)
result4 = len(all_splits[0].page_content)
result5 = all_splits[10].metadata

print('Result 3: \n'+str(result3)+'\n')
print('Result 4: \n'+str(result4)+'\n')
print('Result 5: \n'+str(result5)+'\n')