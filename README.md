# 📊 Autonomous Data Analyst Agent

An AI-powered conversational data analyst that lets you explore and visualize an e-commerce dataset using plain English. Ask questions like *"What are the top 5 product categories by sales?"* and get instant answers with auto-generated charts — all running **100% locally** with no API keys required.

![Python](https://img.shields.io/badge/Python-3.13+-blue?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Agent_Framework-orange)
![Ollama](https://img.shields.io/badge/Ollama-Llama_3.1-purple?logo=meta)
![Streamlit](https://img.shields.io/badge/Streamlit-Chat_UI-red?logo=streamlit)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green?logo=fastapi)

---

## ✨ Features

- **Natural Language Queries** — Ask questions in plain English, the agent writes and executes pandas code automatically
- **Auto-Generated Visualizations** — Request charts and plots; the agent generates them using Matplotlib
- **Fully Local & Private** — Powered by Llama 3.1 via Ollama, your data never leaves your machine
- **Agentic Reasoning** — Built with LangGraph's stateful agent architecture with tool-calling capabilities
- **Rich E-Commerce Dataset** — Pre-loaded with the Olist Brazilian E-Commerce dataset (9 tables, 100k+ orders)

---

## 🏗️ Architecture

```
┌─────────────────┐     HTTP      ┌─────────────────┐     LangGraph     ┌─────────────────┐
│                 │  ──────────▶  │                 │  ─────────────▶   │                 │
│   Streamlit UI  │     /chat     │  FastAPI Server  │    Agent Loop     │   Llama 3.1     │
│   (app.py)      │  ◀──────────  │  (api.py)        │  ◀─────────────   │   (Ollama)      │
│                 │   response    │                 │    reasoning      │                 │
└─────────────────┘               └────────┬────────┘                   └─────────────────┘
                                           │
                                           │ tool call
                                           ▼
                                  ┌─────────────────┐
                                  │ execute_pandas   │
                                  │ _code()          │
                                  │                  │
                                  │ Sandboxed exec   │
                                  │ with DataFrames  │
                                  └─────────────────┘
```

### How It Works

1. You type a natural language question in the Streamlit chat interface
2. The question is sent to the FastAPI backend via a POST request
3. The LangGraph agent passes the question along with the database schema to Llama 3.1
4. The LLM reasons about the question and generates pandas code, invoking the `execute_pandas_code` tool
5. The tool executes the code in a sandboxed environment with access to all DataFrames
6. For plots, the figure is saved to `outputs/current_plot.png` and rendered in the UI
7. For text answers, the result is returned directly to the chat

---

## 📁 Project Structure

```
autonomous-data-agent/
├── agent.py               # LangGraph agent with tool-calling & LLM setup
├── api.py                 # FastAPI server exposing /chat endpoint
├── app.py                 # Streamlit chat UI frontend
├── data_loader.py         # Loads & enriches CSV datasets into DataFrames
├── schema.json            # Auto-generated schema (columns, types, sample data)
├── main.py                # Entry point (placeholder)
├── pyproject.toml         # Project config & dependencies (uv)
├── data/                  # Olist e-commerce CSV files (gitignored)
│   ├── olist_customers_dataset.csv
│   ├── olist_orders_dataset.csv
│   ├── olist_order_items_dataset.csv
│   ├── olist_order_payments_dataset.csv
│   ├── olist_order_reviews_dataset.csv
│   ├── olist_products_dataset.csv
│   ├── olist_sellers_dataset.csv
│   ├── olist_geolocation_dataset.csv
│   └── product_category_name_translation.csv
├── outputs/               # Generated plots saved here
└── .venv/                 # Virtual environment
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.13+**
- **[uv](https://docs.astral.sh/uv/)** — Fast Python package manager
- **[Ollama](https://ollama.com/)** — Local LLM runtime

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/autonomous-data-agent.git
cd autonomous-data-agent
```

### 2. Install Dependencies

```bash
uv sync
```

### 3. Download the Dataset

Download the [Olist Brazilian E-Commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) from Kaggle and place all CSV files in the `data/` directory.

### 4. Pull the LLM Model

```bash
ollama pull llama3.1
```

### 5. Generate the Schema

This extracts column names, data types, and sample data from the CSVs so the LLM understands the dataset structure:

```bash
uv run python data_loader.py
```

### 6. Start the Application

You need **three terminals** running simultaneously:

**Terminal 1 — Ollama (LLM Server):**
```bash
ollama run llama3.1
```

**Terminal 2 — FastAPI Backend:**
```bash
uv run uvicorn api:app --reload
```

**Terminal 3 — Streamlit Frontend:**
```bash
uv run streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## 💬 Example Queries

| Query | Type |
|-------|------|
| *What is the shape of the orders table?* | Data inspection |
| *How many unique customers are in the dataset?* | Aggregation |
| *What are the top 5 product categories by total sales volume?* | Analysis |
| *Plot a bar chart of the top 5 product categories by total sales volume* | Visualization |
| *What is the average delivery time in days?* | Calculation |
| *Which state has the most sellers?* | Grouping |

---

## 📊 Dataset

The project uses the **[Olist Brazilian E-Commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)**, which contains ~100k orders from 2016–2018 across multiple Brazilian marketplaces.

| Table | Rows | Description |
|-------|------|-------------|
| `customers` | 99,441 | Customer profiles and locations |
| `orders` | 99,441 | Order status and timestamps |
| `order_items` | 112,650 | Items in each order (enriched with product categories) |
| `order_payments` | 103,886 | Payment method and installment details |
| `order_reviews` | 99,224 | Customer review scores and comments |
| `products` | 32,951 | Product attributes and category info |
| `sellers` | 3,095 | Seller locations |
| `geolocation` | 1,000,163 | Brazilian zip code coordinates |
| `product_category_name_translation` | 71 | Portuguese → English category mapping |

> **Note:** During loading, the `order_items` table is automatically enriched with product category names (both Portuguese and English) by merging through the `products` and `product_category_name_translation` tables. This simplifies queries involving product categories.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| [LangGraph](https://langchain-ai.github.io/langgraph/) | Stateful agent orchestration with tool-calling |
| [LangChain](https://www.langchain.com/) | LLM abstractions and message handling |
| [Ollama](https://ollama.com/) | Local LLM inference (Llama 3.1) |
| [FastAPI](https://fastapi.tiangolo.com/) | REST API backend |
| [Streamlit](https://streamlit.io/) | Interactive chat UI |
| [Pandas](https://pandas.pydata.org/) | Data manipulation and analysis |
| [Matplotlib](https://matplotlib.org/) | Chart and plot generation |
| [uv](https://docs.astral.sh/uv/) | Dependency management |

---

## ⚙️ Configuration

- **LLM Model** — Change the model in `agent.py` line 59:
  ```python
  llm = ChatOllama(model="llama3.1", temperature=0)
  ```
  You can swap in any Ollama-compatible model (e.g., `mistral`, `codellama`, `llama3.1:70b`).

- **API Port** — The FastAPI server runs on port `8000` by default. Modify in the uvicorn command or `api.py`.

- **Streamlit Port** — Defaults to `8501`. The API URL is configured in `app.py` line 6.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
