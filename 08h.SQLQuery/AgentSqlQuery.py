# Link to tutorial: https://youtu.be/NsVnUz7sp_Y?si=UTdE9-NlMrSxt6Uz

import getpass
import os


import ast
import base64
import io
import json
import operator
from functools import partial
from typing import Annotated, List, Literal, Optional, Sequence, TypedDict

import pandas as pd
from IPython.display import display
from langchain_azure_dynamic_sessions import SessionsPythonREPLTool
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from matplotlib.pyplot import imshow
from PIL import Image




os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass("Azure OpenAI API key")
os.environ["AZURE_OPENAI_ENDPOINT"] = getpass.getpass("Azure OpenAI endpoint")

AZURE_OPENAI_DEPLOYMENT_NAME = getpass.getpass("Azure OpenAI deployment name")
SESSIONS_POOL_MANAGEMENT_ENDPOINT = getpass.getpass(
    "Azure Container Apps dynamic sessions pool management endpoint"
)
SQL_DB_CONNECTION_STRING = getpass.getpass("PostgreSQL connection string")

db = SQLDatabase.from_uri(SQL_DB_CONNECTION_STRING)

llm = AzureChatOpenAI(
    deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME, openai_api_version="2024-02-01"
)

repl = SessionsPythonREPLTool(
    pool_management_endpoint=SESSIONS_POOL_MANAGEMENT_ENDPOINT
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


# Tool schema for querying SQL db
class create_df_from_sql(BaseModel):
    """Execute a PostgreSQL SELECT statement and use the results to create a DataFrame with the given column names."""

    select_query: str = Field(..., description="A PostgreSQL SELECT statement.")
    # We're going to convert the results to a Pandas DataFrame that we pass
    # to the code intepreter, so we also have the model generate useful column and
    # variable names for this DataFrame that the model will refer to when writing
    # python code.
    df_columns: List[str] = Field(
        ..., description="Ordered names to give the DataFrame columns."
    )
    df_name: str = Field(
        ..., description="The name to give the DataFrame variable in downstream code."
    )


# Tool schema for writing Python code
class python_shell(BaseModel):
    """Execute Python code that analyzes the DataFrames that have been generated. Make sure to print any important results."""

    code: str = Field(
        ...,
        description="The code to execute. Make sure to print any important results.",
    )


system_prompt = f"""\
You are an expert at PostgreSQL and Python. You have access to a PostgreSQL database \
with the following tables

{db.table_info}

Given a user question related to the data in the database, \
first get the relevant data from the table as a DataFrame using the create_df_from_sql tool. Then use the \
python_shell to do any analysis required to answer the user question."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("placeholder", "{messages}"),
    ]
)


def call_model(state: AgentState) -> dict:
    """Call model with tools passed in."""
    messages = []

    chain = prompt | llm.bind_tools([create_df_from_sql, python_shell])
    messages.append(chain.invoke({"messages": state["messages"]}))

    return {"messages": messages}




def execute_sql_query(state: AgentState) -> dict:
    """Execute the latest SQL queries."""
    messages = []

    for tool_call in state["messages"][-1].tool_calls:
        if tool_call["name"] != "create_df_from_sql":
            continue

        # Execute SQL query
        res = db.run(tool_call["args"]["select_query"], fetch="cursor").fetchall()

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
        if isinstance(msg, RawToolMessage) and msg.tool_name == "create_df_from_sql"
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
        if tool_call["name"] != "python_shell":
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