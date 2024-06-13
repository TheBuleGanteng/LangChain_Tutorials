import os
import fitz  # PyMuPDF
import numpy as np
from openai import OpenAI
client = OpenAI()
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Alternate import
#import openai
# Set your OpenAI API key
openai.api_key = 'your-api-key-here'


# Initialize an empty list to hold the document contents
documents = []
metadata = []

# Load documents from the 'docs_for_upload' directory
directory = 'docs_for_upload'
for filename in os.listdir(directory):
    if filename.endswith('.pdf'):
        file_path = os.path.join(directory, filename)
        document = fitz.open(file_path)
        text = ""
        for page_num in range(document.page_count):
            page = document.load_page(page_num)
            text += page.get_text()
            metadata.append({'document': filename, 'page': page_num + 1})
        documents.append(text)

        #print(f"Loaded document: {filename}, Page: {page_num + 1}")

# Convert documents to embeddings
vectorizer = TfidfVectorizer(stop_words='english')
doc_embeddings = vectorizer.fit_transform(documents).toarray()

# Create a FAISS index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

# Define a function to query the index and retrieve top-N documents
def retrieve_documents(query, top_n=5):
    query_embedding = vectorizer.transform([query]).toarray()
    distances, indices = index.search(query_embedding, top_n)
    retrieved = [(documents[idx], distances[0][i], metadata[idx]) for i, idx in enumerate(indices[0])]
    
    # Debug print
    #for doc, _, meta in retrieved:
    #    print(f"Retrieved document: {meta['document']}, Page: {meta['page']}")

    return retrieved

# Function to count tokens
def count_tokens(text):
    return len(text.split())

# Combine retrieved documents with the query and use OpenAI's API to generate a response
def generate_response(query, retrieved_docs, temperature=0.7):
    context_parts = []
    for doc, _, meta in retrieved_docs:
        context_parts.append(f"Document: {meta['document']} (Page: {meta['page']})\n{doc}")
    context = "\n\n".join(context_parts)
    context = context[:50].strip()
    
    # Debug print
    #print(f"Constructed context: {context[:500]}...")  # Printing only the first 500 characters for brevity
    
    max_context_tokens = 4000  # Adjust based on model's max tokens and response length

    # Truncate context if necessary
    if count_tokens(context) > max_context_tokens:
        truncated_context = []
        current_tokens = 0
        for doc, _, meta in retrieved_docs:
            doc_tokens = count_tokens(doc)
            if current_tokens + doc_tokens > max_context_tokens:
                break
            truncated_context.append(f"Document: {meta['document']} (Page: {meta['page']})\n{doc}")
            current_tokens += doc_tokens
        context = "\n\n".join(truncated_context)

        # Debug print
        #print(f"Truncated context: {context[:500]}...")  # Printing only the first 500 characters for brevity

    messages = [
        {"role": "system", "content": "You are a helpful assistant proving in-depth market research."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=150,
        temperature=temperature
    )

    response_text = response.choices[0].message.content.strip()

    # Compare response with document contents
    response_embedding = vectorizer.transform([response_text]).toarray()
    similarity = cosine_similarity(response_embedding, doc_embeddings)
    max_similarity = np.max(similarity)

    # Define a threshold for similarity
    threshold = 0.4

    # Check if response is similar to document contents
    if max_similarity < threshold:
        response_text += "\n\nWarning: this information does not come from the documents provided and must be verified before use."
    else:
        response_text = response_text + "\n\n" + context
    
    # Append the context to the response
    return response_text

# Example usage
query = "What is per-capita consumption of beer in Ho Chi Minh City?"
retrieved_docs = retrieve_documents(query)
response = generate_response(query, retrieved_docs, temperature=0.7)
print(response)
