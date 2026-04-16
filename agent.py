import pandas as pd
import json
from typing import Annotated
from typing_extensions import TypedDict
import os
import matplotlib.pyplot as plt
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from data_loader import load_dataframes

# Ensure the output directory exists
os.makedirs("outputs", exist_ok=True)

# 1. Load Data & Schema
print("Booting up Agent Environment...")
dfs = load_dataframes()

with open("schema.json", "r") as f:
    db_schema = json.load(f)

# 2. Define the Execution Tool
@tool
def execute_pandas_code(code: str) -> str:
    """
    Executes Python pandas code to analyze the dataset.
    You have access to a dictionary named 'dfs' containing the dataframes. 
    You MUST assign your final numerical or string answer to a variable named 'result'.
    """
    clean_code = code.replace("```python", "").replace("```", "").strip()
    print(f"\n[Agent is running code...]\n{clean_code}\n")
    
    try:
        local_env = {"dfs": dfs, "pd": pd, "plt": plt}
        exec(clean_code, {}, local_env)
        
        if 'result' in local_env:
            output = str(local_env['result'])
            print(f"[Tool Output]: {output}")
            return output
        else:
            return "Execution Error: You forgot to assign the final answer to the 'result' variable."
    except Exception as e:
        error_msg = f"Python Error: {type(e).__name__}: {str(e)}"
        print(f"[{error_msg}]")
        return error_msg

# --- LANGGRAPH SETUP ---

# 3. Define the State (Holds the conversation history)
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 4. Initialize Local LLM & Bind Tools
llm = ChatOllama(model="llama3.1", temperature=0)
tools = [execute_pandas_code]
llm_with_tools = llm.bind_tools(tools)

# 5. Define the Agent Node
def chatbot(state: State):
    """The main reasoning engine."""
    sys_msg = SystemMessage(content=f"""You are an autonomous Senior Data Analyst.
A dictionary of pandas DataFrames named `dfs` is already loaded in memory.

Here is the database schema:
{json.dumps(db_schema, indent=2)}

CRITICAL TOOL RULES:
1. You have a native tool called `execute_pandas_code`. Use the system's tool-calling API to trigger it.
2. NEVER type `execute_pandas_code(...)` in your text response. 
3. The code you send to the tool MUST assign the final answer to a variable named `result`.

VISUALIZATIONS:
4. If the user asks for a plot or chart, use `matplotlib.pyplot` (imported as `plt`).
5. Save the figure exactly like this: `plt.savefig('outputs/current_plot.png', bbox_inches='tight')`. Do not use plt.show().
6. If you save a plot, your python code MUST also include: `result = "Plot saved successfully"`.
7. When the tool returns success, your final text response to the user MUST include the exact phrase: [PLOT_GENERATED].
""")
    # Prepend the system prompt to the conversation history
    messages = [sys_msg] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# 6. Build and Compile the Graph
graph_builder = StateGraph(State)

# Add nodes
graph_builder.add_node("agent", chatbot)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# Add edges (The Routing Logic)
graph_builder.add_edge(START, "agent")
# tools_condition checks if the LLM decided to use a tool. If yes, go to 'tools'. If no, go to END.
graph_builder.add_conditional_edges("agent", tools_condition)
graph_builder.add_edge("tools", "agent")

app = graph_builder.compile()

if __name__ == "__main__":
    print("\n--- Agentic Loop Ready ---")
    
    # Let's ask it a real question that requires writing code
    user_query = "What are the names of the columns in the sellers table?"
    print(f"\nUser: {user_query}")
    
    # Run the graph
    events = app.stream(
        {"messages": [HumanMessage(content=user_query)]},
        stream_mode="values"
    )
    
    for event in events:
        if "messages" in event:
            # Print the latest message in the state
            event["messages"][-1].pretty_print()