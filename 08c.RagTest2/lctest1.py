# Notes:
# File below is based on the following tutorial: https://youtu.be/nE2skSRWTTs?si=oLVFdDz1AX2e_GRd
# Updates:
# 1. Updated 'from langchain import HuggingFaceHub' to 'from langchain_huggingface import HuggingFaceEndpoint' due to depreciation
# 2. Updated 'from langchain import PromptTemplate' to 'from langchain_core.prompts import PromptTemplate' due to depreciation
# 3. Updated 'from langchain.chains import LLMChain' to 'from langchain.chains import RunnableSequence' due
# 3. Updated from 'model_kwargs={"temperature":1e-10}' to 'temperature=1e-10' due to depreciation
import os
from dotenv import load_dotenv # Not in tutorial: added to use gitignored .env file
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import LLMChain


# Not in tutorial: Sets path to .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '..', 'gitignored', '.env')
load_dotenv(dotenv_path)

HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN') 
os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACEHUB_API_TOKEN

# initialize HuggingFace LLM
flan_t5 = HuggingFaceEndpoint(
    repo_id="google/flan-t5-large",
    temperature=1e-10, # Temp is (0-1, with 1e-10 effectively zero)
    max_new_tokens=250,  # Represents length on response, must be 1-250.
)

# build prompt template for simple question-answering
template = """Question: {question}

Answer: """
prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(
    prompt=prompt,
    llm=flan_t5
)

question = "Which NFL team won the Super Bowl in the 2010 season?"

print(llm_chain.invoke(question))