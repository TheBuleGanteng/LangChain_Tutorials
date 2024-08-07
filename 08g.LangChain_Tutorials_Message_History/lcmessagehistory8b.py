
from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)
import os
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv
import fitz  # PyMuPDF

# Load environment variables from .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '..', 'gitignored', '.env')
load_dotenv(dotenv_path)

# Ensure that environment variables are set
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
AICHAT_LANGCHAIN_PROJECT = os.getenv('AICHAT_LANGCHAIN_PROJECT')

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['AICHAT_LANGCHAIN_PROJECT'] = AICHAT_LANGCHAIN_PROJECT
os.environ['LANGCHAIN_API_KEY'] = LANGCHAIN_API_KEY 
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Initialize OpenAI API

# Load documents from the folder and extract text from PDFs
def load_documents():
    folder_path = os.path.join(os.path.dirname(__file__), 'docs_for_upload')
    documents = []
    file_names = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.pdf'):
            file_path = os.path.join(folder_path, file_name)
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            text = text.strip()
            if text:  # Only add non-empty documents
                documents.append(text)
                file_names.append(file_name)
    print(f"Loaded {len(documents)} documents.")
    return documents, file_names

# Create a TF-IDF matrix and FAISS index
def create_faiss_index(documents):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    tfidf_array = tfidf_matrix.toarray()

    index = faiss.IndexFlatL2(tfidf_array.shape[1])
    index.add(np.ascontiguousarray(tfidf_array))
    return index, vectorizer

# Search documents using FAISS
def search_documents(query, index, vectorizer, documents, file_names, top_k=5):
    query_vec = vectorizer.transform([query]).toarray()
    distances, indices = index.search(query_vec, top_k)
    results = [(documents[i], file_names[i]) for i in indices[0]]
    return results

# Generate a response using OpenAI API
def generate_response(query, results):
    context = "\n\n".join([f"{file_name}:\n{doc}" for doc, file_name in results])
    prompt = f"Answer the following question based on the provided documents if possible. If the documents do not contain the information, provide a general answer:\n\nDocuments:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ])

    return response.choices[0].message.content.strip()

# Check if the response is based on the provided documents
def is_response_from_documents(response, results):
    for doc, file_name in results:
        if doc in response:
            return True
    return False

# Main function to run the RAG assistant
def rag_assistant(query):
    documents, file_names = load_documents()
    if not documents:
        return "No documents found or documents are empty."

    index, vectorizer = create_faiss_index(documents)
    results = search_documents(query, index, vectorizer, documents, file_names)
    response = generate_response(query, results)

    if is_response_from_documents(response, results):
        sources = ", ".join([file_name for _, file_name in results])
        response = f"According to the documents below,\n{response}\n\nSources: {sources}"

    return response


# Example usage
if __name__ == "__main__":
    query = "Hello, my name is Bob"  # Replace with your actual question
    response = rag_assistant(query)
    print("Response:", response)