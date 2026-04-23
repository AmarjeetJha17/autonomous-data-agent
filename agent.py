import pandas as pd
import json
from typing import Annotated
from typing_extensions import TypedDict
import os
import shutil
import matplotlib
matplotlib.use('Agg')
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

# Regenerate schema to ensure enriched columns are included
from data_loader import generate_schema
db_schema = generate_schema(dfs)
with open("schema.json", "w") as f:
    json.dump(db_schema, f, indent=4)

# 2. Define the Execution Tool
@tool
def execute_pandas_code(code: str) -> str:
    """
    Executes Python pandas code to analyze the dataset.
    You have access to a dictionary named 'dfs' containing the dataframes.
    A compatibility alias named 'df' is also available and points to the same dictionary.
    You MUST assign your final numerical or string answer to a variable named 'result'.
    WARNING: The 'code' parameter MUST be purely valid python syntax. DO NOT include markdown, explanations, or English text.
    """
    clean_code = code.replace("```python", "").replace("```", "").strip()
    print(f"\n[Agent is running code...]\n{clean_code}\n")
    
    try:
        # Reset plotting state per execution so old figures never leak into a new request.
        plt.close("all")
        pre_existing_figures = set(plt.get_fignums())

        local_env = {
            "dfs": dfs,
            "df": dfs,
            "pd": pd,
            "plt": plt,
            "get_table_schema": get_table_schema,
            "os": os,
        }
        exec(clean_code, {}, local_env)
        
        if 'result' in local_env:
            output = str(local_env['result'])

            # If model saved a chart but left result as an Axes/object,
            # honor explicit file_path variable when it points to an existing file.
            if not output.startswith("Plot saved successfully at "):
                explicit_path = local_env.get("file_path")
                if isinstance(explicit_path, str) and os.path.exists(explicit_path):
                    output = f"Plot saved successfully at {explicit_path}"

            # Recovery path: if plotting code produced a figure but forgot to save,
            # persist the latest figure to a known path for the frontend.
            figure_numbers = [
                fig_num for fig_num in plt.get_fignums() if fig_num not in pre_existing_figures
            ]
            if figure_numbers and not output.startswith("Plot saved successfully at "):
                os.makedirs("outputs", exist_ok=True)
                fallback_plot_path = os.path.join("outputs", "current_plot.png")
                plt.figure(figure_numbers[-1]).savefig(fallback_plot_path, bbox_inches="tight")
                output = f"Plot saved successfully at {fallback_plot_path}"

            # If model saved a plot outside outputs/, move it so the UI can always render it.
            if output.startswith("Plot saved successfully at "):
                raw_path = output.replace("Plot saved successfully at ", "", 1).strip()
                normalized_path = raw_path.strip('"\'')

                if normalized_path and not normalized_path.startswith("outputs") and os.path.exists(normalized_path):
                    os.makedirs("outputs", exist_ok=True)
                    destination_path = os.path.join("outputs", os.path.basename(normalized_path))
                    shutil.move(normalized_path, destination_path)
                    output = f"Plot saved successfully at {destination_path}"
                elif normalized_path and normalized_path.startswith("outputs") and not os.path.exists(normalized_path):
                    # Some model generations report outputs/... but save in cwd; recover by basename.
                    fallback_source = os.path.basename(normalized_path)
                    if os.path.exists(fallback_source):
                        os.makedirs("outputs", exist_ok=True)
                        destination_path = os.path.join("outputs", os.path.basename(fallback_source))
                        shutil.move(fallback_source, destination_path)
                        output = f"Plot saved successfully at {destination_path}"

            print(f"[Tool Output]: {output}")
            plt.close("all")
            return output
        else:
            plt.close("all")
            return "Execution Error: You forgot to assign the final answer to the 'result' variable."
    except KeyError as e:
        plt.close("all")
        col_name = str(e).strip("'\"[]")
        # Provide helpful hints for common column name errors
        if 'product_category' in col_name.lower():
            hint = " HINT: For product categories in order_items, use 'product_category_name_english' column."
            return f"Column Error: {str(e)}.{hint}"
        else:
            return f"Column Error: {str(e)}"
    except Exception as e:
        plt.close("all")
        error_msg = f"Python Error: {type(e).__name__}: {str(e)}"
        print(f"[{error_msg}]")
        return error_msg

@tool
def get_table_schema(table_name: str) -> str:
    """
    Returns the columns, data types, and sample data for a given table.
    Use this tool to understand the structure of a table before querying it.
    """
    if table_name in db_schema:
        return json.dumps(db_schema[table_name], indent=2)
    else:
        return f"Error: Table '{table_name}' not found. Available tables: {list(db_schema.keys())}"

# --- LANGGRAPH SETUP ---

# 3. Define the State (Holds the conversation history)
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 4. Initialize Local LLM & Bind Tools
llm = ChatOllama(model="llama3.1", temperature=0)
tools = [execute_pandas_code, get_table_schema]
llm_with_tools = llm.bind_tools(tools)

# 5. Define the Agent Node
def chatbot(state: State):
    """The main reasoning engine."""
    table_names = list(db_schema.keys())
    sys_msg = SystemMessage(content=f"""You are an autonomous Senior Data Analyst.
A dictionary of pandas DataFrames named `dfs` is already loaded in memory.
The available tables (DataFrames) are: {table_names}

IMPORTANT TABLE ENRICHMENT:
- The 'order_items' table has been enriched with English product categories
- Use column 'product_category_name_english' for product category analysis
- Avoid 'product_category_name_translation' or 'product_category' - these don't exist

WORKFLOW:
1. For unknown column names: Call get_table_schema('table_name') first
2. Write Python code to analyze data using execute_pandas_code tool
3. Always assign final result to a 'result' variable before returning
4. For plots: save to outputs/ directory and include [PLOT_GENERATED: path] in response
5. Return plain English text, not code or JSON

Code Guidelines:
- Pure Python only, no markdown or comments
- For plots: plt.savefig("outputs/filename.png", bbox_inches='tight'); result = "Plot saved to..."
- Only generate plots when user explicitly asks for visualization
- For text responses: Keep answer concise and in plain English
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