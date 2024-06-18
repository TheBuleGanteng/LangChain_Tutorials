# From LangChain tutorial: https://python.langchain.com/v0.2/docs/tutorials/chatbot/#quickstart

from dotenv import load_dotenv # Not in tutorial: added to use gitignored .env file
from langchain_openai import ChatOpenAI
import os
from langchain_core.messages import HumanMessage, AIMessage


# Not in tutorial: Sets path to .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '..', 'gitignored', '.env')
load_dotenv(dotenv_path)


OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

model = ChatOpenAI(model="gpt-3.5-turbo")


response = model.invoke(
    [
        HumanMessage(content="Hi! I'm Bob"),
        AIMessage(content="Hello Bob! How can I assist you today?"),
        HumanMessage(content="What's my name?"),
    ]
)

# Print the response
print(response.content)