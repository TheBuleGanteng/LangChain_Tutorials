# Link to tutorial: https://youtu.be/NsVnUz7sp_Y?si=UTdE9-NlMrSxt6Uz
# Note: This version is adapted for use with SQLITE/Django

import os
import sys
import django

# Add the project root to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')) # Assumes this file is located in project_folder/<app>/helpers/helpers.py
sys.path.insert(0, project_root)

# Set the settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'aichat.settings') # This project's name was aichat

# Setup Django
django.setup()

# Rest of your imports and code
import ast
import base64
import io
import json
import operator
from functools import partial
from typing import Annotated, List, Literal, Optional, Sequence, TypedDict

import pandas as pd
from IPython.display import display
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from matplotlib.pyplot import imshow
from PIL import Image

# Setting up environment variables for OpenAI
import getpass
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Setup database connection using Django settings
from django.db import connection

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.7
)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

class RawToolMessage(ToolMessage):
    """
    Customized Tool message that lets us pass around the raw tool outputs (along with string contents for passing back to the model).
    """
    raw: dict
    """Arbitrary (non-string) tool outputs. Won't be sent to model."""
    tool_name: str
    """Name of tool that generated output."""

# Tool schema for querying SQL db using Django ORM
class CreateDfFromSql(BaseModel):
    """Execute a SQLite SELECT statement and use the results to create a DataFrame with the given column names."""
    select_query: str = Field(..., description="A SQLite SELECT statement.")
    df_columns: List[str] = Field(
        ..., description="Ordered names to give the DataFrame columns."
    )
    df_name: str = Field(
        ..., description="The name to give the DataFrame variable in downstream code."
    )

# Tool schema for writing Python code
class PythonShell(BaseModel):
    """Execute Python code that analyzes the DataFrames that have been generated. Make sure to print any important results."""
    code: str = Field(
        ...,
        description="The code to execute. Make sure to print any important results.",
    )

system_prompt = """\
You are an expert at SQLite and Python. You have access to a SQLite database with the following tables

{table_info}

Given a user question related to the data in the database, \
first get the relevant data from the table as a DataFrame using the CreateDfFromSql tool. Then use the \
PythonShell to do any analysis required to answer the user question."""

# Extract table info from the database
def get_table_info():
    with connection.cursor() as cursor:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
    table_info = "\n".join([table[0] for table in tables])
    return table_info

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt.format(table_info=get_table_info())),
        ("placeholder", "{messages}"),
    ]
)

def call_model(state: AgentState) -> dict:
    """Call model with tools passed in."""
    messages = []

    chain = prompt | llm.bind_tools([CreateDfFromSql, PythonShell])
    messages.append(chain.invoke({"messages": state["messages"]}))

    return {"messages": messages}

def execute_sql_query(state: AgentState) -> dict:
    """Execute the latest SQL queries."""
    messages = []

    for tool_call in state["messages"][-1].tool_calls:
        if tool_call["name"] != "CreateDfFromSql":
            continue

        # Execute SQL query
        with connection.cursor() as cursor:
            cursor.execute(tool_call["args"]["select_query"])
            res = cursor.fetchall()

        # Convert result to Pandas DataFrame
        df_columns = tool_call["args"]["df_columns"]
        df = pd.DataFrame(res, columns=df_columns)
        df_name = tool_call["args"]["df_name"]

        # Add tool output message
        messages.append(
            RawToolMessage(
                f"Generated dataframe {df_name} with columns {df_columns}",  # What's sent to model.
                raw={df_name: df},
                tool_call_id=tool_call["id"],
                tool_name=tool_call["name"],
            )
        )

    return {"messages": messages}

def _upload_dfs_to_repl(state: AgentState) -> str:
    """
    Upload generated dfs to code intepreter and return code for loading them.

    Note that code intepreter sessions are short-lived so this needs to be done
    every agent cycle, even if the dfs were previously uploaded.
    """
    df_dicts = [
        msg.raw
        for msg in state["messages"]
        if isinstance(msg, RawToolMessage) and msg.tool_name == "CreateDfFromSql"
    ]
    name_df_map = {name: df for df_dict in df_dicts for name, df in df_dict.items()}

    # Data should be uploaded as a BinaryIO.
    # Files will be uploaded to the "/mnt/data/" directory on the container.
    for name, df in name_df_map.items():
        buffer = io.StringIO()
        df.to_csv(buffer)
        buffer.seek(0)
        repl.upload_file(data=buffer, remote_file_path=name + ".csv")

    # Code for loading the uploaded files.
    df_code = "import pandas as pd\n" + "\n".join(
        f"{name} = pd.read_csv('/mnt/data/{name}.csv')" for name in name_df_map
    )
    return df_code

def _repl_result_to_msg_content(repl_result: dict) -> str:
    """
    Display images with including them in tool message content.
    """
    content = {}
    for k, v in repl_result.items():
        # Any image results are returned as a dict of the form:
        # {"type": "image", "base64_data": "..."}
        if isinstance(repl_result[k], dict) and repl_result[k]["type"] == "image":
            # Decode and display image
            base64_str = repl_result[k]["base64_data"]
            img = Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))
            display(img)
        else:
            content[k] = repl_result[k]
    return json.dumps(content, indent=2)

def execute_python(state: AgentState) -> dict:
    """
    Execute the latest generated Python code.
    """
    messages = []

    df_code = _upload_dfs_to_repl(state)
    last_ai_msg = [msg for msg in state["messages"] if isinstance(msg, AIMessage)][-1]
    for tool_call in last_ai_msg.tool_calls:
        if tool_call["name"] != "PythonShell":
            continue

        generated_code = tool_call["args"]["code"]
        repl_result = repl.execute(df_code + "\n" + generated_code)

        messages.append(
            RawToolMessage(
                _repl_result_to_msg_content(repl_result),
                raw=repl_result,
                tool_call_id=tool_call["id"],
                tool_name=tool_call["name"],
            )
        )
    return {"messages": messages}
