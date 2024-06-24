import os
import fitz  # Imports pdfs
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

# Sets path to .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '..', 'gitignored', '.env')
load_dotenv(dotenv_path)

# Ensure that environment variables are set
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LANGCHAIN_PROJECT = os.getenv('LANGCHAIN_PROJECT')

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = LANGCHAIN_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['LANGCHAIN_PROJECT'] = LANGCHAIN_PROJECT

# Model set
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
source_cutoff = 1

class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

# Step 1: Upload the documents    
docs = []
print(f'running aichat_chat/helpers, generate_response() ... docs is: {docs}')

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
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# Step 3. Incorporate the retriever into a question-answering chain.
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Whenever possible, use the following pieces of retrieved context to answer the question "
    "If your information comes from retrieved context, include the phrase 'According to the documents listed below,' "
    "If your information does not come from retrieved context, include the phrase 'While the uploaded documents do not contain this information, according to my general knowledge,' "
    "Be sure to include the exact words Use three sentences maximum and keep the "
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

result = rag_chain.invoke({"input": "What is the most popular beer in Vietnam?"})

print('Answer: \n' + result['answer'] + '\n')

# Extract and print the sources
sources = result['context']
source_list = []
for doc in sources:
    source = f'{ doc.metadata["source"] }, page: { doc.metadata["page_number"] }'
    source_list.append(source)
source_list = source_list[source_cutoff:]
print('Sources: \n' + '\n'.join(source_list) + '\n')
