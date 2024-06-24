# From LangChain tutorials: 
# https://python.langchain.com/v0.2/docs/tutorials/rag/#retrieval-and-generation-generate

import os
import getpass
import bs4
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv # Not in tutorial: added to use gitignored .env file
import fitz # Imports pdfs

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

model = ChatOpenAI(model='gpt-3.5-turbo')

# This question is referenced throughout
question = 'How old is Kevin Bacon?'
context_index = 1

# Step 1: Load, chunk and index the contents of a website.
# Here, we use DocumentLoaders, which are objects that load in data from a source and return a list of Documents
# A Document is an object with some page_content (str) and metadata (dict).
# Only keep post title, headers, and content from the full HTML.


# Define a simple Document class for use in this script
class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

# Initialize an empty list to hold the document contents
docs = []

# Load documents from the 'docs_for_upload' directory
directory = 'docs_for_upload'
for filename in os.listdir(directory):
    if filename.endswith('.pdf'):
        file_path = os.path.join(directory, filename)
        document = fitz.open(file_path)
        for page_num in range(document.page_count): # This makes each page in each pdf a document, allowing the metadata to include both the file name and the page number.
            page = document.load_page(page_num)
            text = page.get_text()
            docs.append(Document(
                page_content=text,
                metadata={'source': filename, 'page_number': page_num + 1} # The +1 addresses zero indexing
            ))


result1 = len(docs[0].page_content)
result2 = docs[0].page_content[:50]

print('Result 1: \n'+str(result1)+'\n')
print('Result2: \n'+result2+'\n')



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



# Step 3: Embed all the vector spits
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())



# Step 4: Set up a retriever that finds and pulls the relevant data form the vectorstore in response to a user question 
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

retrieved_docs = retriever.invoke(question)

result6 = len(retrieved_docs)
result7 = retrieved_docs[0].page_content


print('Result 6: \n'+str(result6)+'\n')
print('Result 7: \n'+str(result7)+'\n')



# Step 5:  Pulls a new prompt from the following link: https://smith.langchain.com/hub/rlm/rag-prompt?organizationId=a8c8552e-2aed-5c7b-bc31-30379d8a81c5
prompt = hub.pull("rlm/rag-prompt")
example_messages = prompt.invoke(
    {"context": "filler context", "question": "filler question"}
).to_messages()

result8= example_messages
result9= example_messages[0].content


print('Result 8: \n'+str(result8)+'\n')
print('Result 9: \n'+str(result9)+'\n')



# Step 6: Use the LCEL Runnable protocol to define the chain in order to:
# 2a. pipe together components and functions in a transparent way
# 2b. automatically trace our chain in LangSmith
# 2c. get streaming, async, and batched calling out of the box.

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# retriever | format_docs passes the question through the retriever, generating Document objects, and then to format_docs to generate strings.
# RunnablePassthrough() passes through the input question unchanged.
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model # Tells it to use the model defined above
    | StrOutputParser() # plucks the string content out of the LLM's output message
)


print('Result 10: \n')
for chunk in rag_chain.stream(question):
    print(chunk, end="", flush=True)
print('\n')



# This version uses built-in convenience functions to abstract away the preceding steps
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question." 
    "If you don't know the answer, say that you don't know. In that case, your response must include the term 'don't'."
    "Use three sentences maximum and keep the answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(model, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

response = rag_chain.invoke({"input": question})

result11 = response 
print('Result 11: \n'+str(result11)+'\n')

result12 = response["answer"]
print('Result 12: \n'+str(result12)+'\n')

# Below, 'citation' looks inside the response dict, specifically the first context listed (hence [0]), and returns the source 
if "don't" in response['answer']:
    result13 = response['answer']
else: 
    citation = f" (Source: {response['context'][context_index].metadata['source']}, Page: {response['context'][context_index].metadata['page_number']})"
    result13 = response['answer'] + citation
print('Result 13: \n'+str(result13)+'\n')


# Same as result 12 above, but the answer is streamed.
print('Result 14: \n')
context_metadata = response['context'][context_index].metadata 
answer_chunks = [] # First we accumulate all the answer chunks into a coherent answer and then we list the citation 
for chunk in rag_chain.stream({"input": question}):
    if "answer" in chunk:
        answer_chunks.append(chunk["answer"])
citation = f" (Source: {response['context'][context_index].metadata['source']}, Page: {response['context'][context_index].metadata['page_number']})"
joined_response = "".join(answer_chunks)

if "don't" in joined_response:
    print(joined_response)
else: 
    citation = f" (Source: {context_metadata['source']}, Page: {context_metadata['page_number']})"
    print(joined_response + citation)
print('\n')

