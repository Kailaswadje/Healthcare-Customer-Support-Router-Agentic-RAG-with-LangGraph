# 🏥 Healthcare Customer Support Router — An Agentic RAG System with LangGraph

An end-to-end **Router Agentic RAG** system for a healthcare provider: every customer query is **classified by department**, checked for **sentiment**, and then **routed** down one of three paths — a grounded RAG answer from the right part of the knowledge base, an escalation form for a human support agent, or an **emergency form for the on-call medical team** when the message signals distress.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.3.18-1C3C3C)
![LangChain](https://img.shields.io/badge/LangChain-0.3.20-1C3C3C)
![OpenAI](https://img.shields.io/badge/OpenAI-gpt--4o--mini-412991?logo=openai&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6F00)
![Domain](https://img.shields.io/badge/Domain-Healthcare-2E8B57)

---

## 📌 Overview

A plain RAG chatbot answers every message the same way: retrieve, then generate. That is the wrong behaviour for healthcare support. A billing question needs a quick factual answer; an angry customer needs a human; someone who says *"I can't breathe"* needs a doctor, not a paragraph from the FAQ.

This project builds that judgement into the system as a **LangGraph state machine**:

1. **Categorise** the query → Billing, Appointments, Records or Insurance
2. **Analyse sentiment** → Positive, Neutral, Negative or Distress
3. **Route** on the result:
   - Positive / Neutral → **department-filtered RAG answer**
   - Negative → **escalation form → human support agent**
   - Distress → **emergency form → on-call medical team**

Both classification steps use **structured output** with `Literal` types, so routing never depends on parsing free text.

---

## 🏗️ Architecture

```
                         customer_query
                               │
                               ▼
                    ┌─────────────────────┐
                    │  categorize_inquiry │  → Billing / Appointments / Records / Insurance
                    └─────────────────────┘
                               │
                               ▼
                 ┌──────────────────────────┐
                 │ analyze_inquiry_sentiment│  → Positive / Neutral / Negative / Distress
                 └──────────────────────────┘
                               │
                       determine_route()
          ┌────────────────────┼─────────────────────┐
     Negative              Distress           Positive / Neutral
          ▼                    ▼                      ▼
 accept_user_input_    accept_user_input_    generate_department_
   escalation              oncall                 response
          ▼                    ▼               (Chroma + metadata
 escalate_to_human_    escalate_to_oncall_      filter + LLM)
       agent                 team                     │
          └────────────────────┴──────────┬───────────┘
                                          ▼
                                         END
```

---

## 🔬 Part 1 — The Knowledge Base and Vector Store

```python
with open("./healthcare_db.json") as f:
    knowledge_base = json.load(f)

processed_docs = [Document(page_content=d["text"], metadata=d["metadata"]) for d in knowledge_base]

kbase_db = Chroma.from_documents(
    documents=processed_docs,
    collection_name="knowledge_base",
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    collection_metadata={"hnsw:space": "cosine"},
    persist_directory="./knowledge_base",
)
```

Every document keeps its **`category` metadata** (`billing`, `appointments`, `medical_records`, `insurance`). That metadata is what makes department routing possible later — retrieval can be restricted to one department instead of searching the whole knowledge base.

### Retriever with a Score Threshold
```python
kbase_search = kbase_db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.3},
)
```
`similarity_score_threshold` returns **nothing** when no chunk is similar enough, instead of always returning the top 3. That empty result is what lets the answer step say "I don't know" honestly.

### A Lesson the Notebook Tests Explicitly
The notebook runs the same query with `{'category': 'billing'}` and `{'category': 'Billing'}`. **Metadata filters are case-sensitive** — the capitalised version returns nothing. This is why the response node maps the LLM's `"Billing"` label to the stored lowercase `"billing"` before filtering.

---

## 🔬 Part 2 — The Agent State

```python
class CustomerSupportAgentState(TypedDict):
    customer_query: str
    query_category: str
    query_sentiment: str
    escalation_cust_info: dict
    oncall_cust_info: dict
    final_response: str

class QueryCategory(BaseModel):
    categorized_topic: Literal["Billing", "Appointments", "Records", "Insurance"]

class QuerySentiment(BaseModel):
    sentiment: Literal["Positive", "Neutral", "Negative", "Distress"]
```

The `TypedDict` is the shared memory every node reads and writes. The two Pydantic models are **output contracts** for the LLM: with `Literal`, the model can only return one of the allowed labels, so the router never receives an unexpected string.

---

## 🔬 Part 3 — The Seven Nodes

| Node | What it does |
|---|---|
| `categorize_inquiry` | LLM + `QueryCategory` structured output → department label |
| `analyze_inquiry_sentiment` | LLM + `QuerySentiment` structured output → sentiment label |
| `generate_department_response` | Maps label → metadata filter, retrieves from Chroma, answers with `prompt \| llm` |
| `accept_user_input_escalation` | ipywidgets form (name, number, issue) for negative sentiment |
| `escalate_to_human_agent` | Confirms the details back and hands off to a human agent |
| `accept_user_input_oncall` | ipywidgets emergency form (name, number) for distress |
| `escalate_to_oncall_team` | Confirms the details back and hands off to the on-call doctors |

### Department-Filtered RAG
```python
if categorized_topic.lower() == "billing":
    metadata_filter = {"category": "billing"}
...
kbase_search.search_kwargs["filter"] = metadata_filter
relevant_docs = kbase_search.invoke(query)
```
The answer prompt includes an explicit fallback: *"In case there is no knowledge base information or you do not know the answer just say: Apologies I was not able to answer your question, please reach out to +1-xxx-xxxx"* — the same retrieval-grounding discipline used throughout this portfolio's RAG work.

### Human-in-the-Loop Forms
The escalation nodes use **`ipywidgets`** for the form and **`jupyter-ui-poll`** to wait for the submit click without freezing the notebook. In production these two nodes are where you would plug in email, WhatsApp, paging or ticketing APIs — the notebook marks those points clearly.

---

## 🔬 Part 4 — Routing and Compiling the Graph

```python
def determine_route(support_state) -> str:
    if support_state["query_sentiment"] == "Negative":
        return "accept_user_input_escalation"
    elif support_state["query_sentiment"] == "Distress":
        return "accept_user_input_oncall"
    elif support_state["query_category"] in ["Billing", "Appointments", "Records", "Insurance"]:
        return "generate_department_response"

customer_support_graph.add_conditional_edges("analyze_inquiry_sentiment", determine_route, [...])
compiled_support_agent = customer_support_graph.compile(checkpointer=MemorySaver())
```

Two design points:

- **Sentiment is checked before category.** An angry or distressed customer is escalated even if the question itself is a simple billing one — safety and service recovery come before answering.
- **`MemorySaver` + `thread_id`** keeps a separate state per customer session (`uid = 's1'`, `uid = 'jon007'` in the tests).

---

## 🧪 Test Scenarios Run in the Notebook

| Query | Expected route |
|---|---|
| "I am unable to breathe properly, need help" | Distress → emergency form → on-call team |
| "how do I get my invoice?" | Billing → RAG answer |
| "how could I update my policy?" | Insurance → RAG answer |
| "Which doctors are available?" | Appointments → RAG answer |
| "I'm fed up with the portal hanging, need help ASAP" | Negative → escalation form → human agent |
| "I'm really fed up with your services... I will ensure to cascade this msg to the public" | Negative → escalation form → human agent |
| "how can you ensure that my record is safe in your system?" | Records → RAG answer |

---

## 🗂️ Repository Structure

```
healthcare-support-router-agentic-rag/
├── Project_Build_a_Healthcare_Customer_Support_Router_Agentic_RAG_System.ipynb
├── requirements.txt       # Dependencies
├── .gitignore             # Keeps secrets, data and the vector DB out of git
├── .env.example           # Template for required environment variables
└── README.md              # This documentation
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- An [OpenAI API key](https://platform.openai.com/api-keys)
- Jupyter (the forms use `ipywidgets`, so run it in Jupyter or Colab, not a plain script)

### Installation

```bash
git clone https://github.com/Kailaswadje/healthcare-support-router-agentic-rag.git
cd healthcare-support-router-agentic-rag

pip install -r requirements.txt

jupyter notebook Project_Build_a_Healthcare_Customer_Support_Router_Agentic_RAG_System.ipynb
```

The notebook downloads `healthcare_db.json` with `gdown`; a manual Google Drive link is in the notebook as a fallback.

> ⚠️ The notebook asks for the API key with `getpass()` — keep it that way. Clear outputs before pushing: the notebook embeds several large base64 images and the forms store whatever was typed into them.

---

## 🧠 Key Takeaways

- **Routing is what makes RAG "agentic"** — the system decides *how* to respond before deciding *what* to say
- **Structured output removes guesswork** — `Literal` types turn the LLM into a reliable classifier the router can trust
- **Sentiment before category** is a deliberate safety ordering for a healthcare setting
- **Metadata filters are case-sensitive** — the notebook proves it, and the code maps labels to stored values to avoid silent empty results
- **Score-threshold retrieval + an explicit fallback line** keeps answers grounded when the knowledge base has nothing relevant
- **Human-in-the-loop nodes are integration points** — forms here, APIs in production

---

## ⚠️ Honest Limitations

- **Not a clinical tool.** The distress route only collects contact details and promises a callback; a real deployment must also show emergency numbers (e.g. 999 in the UK) immediately.
- **Sentiment is LLM-judged.** A distressed message phrased calmly could be classed as Neutral; the distress route needs evaluation on real examples before use.
- **The forms are notebook-only.** `ipywidgets` won't run in a web app; a production version needs LangGraph's `interrupt()` or a front end.

---

## 🔮 Possible Extensions

- [ ] Replace the ipywidgets forms with LangGraph `interrupt()` for a front-end-agnostic human-in-the-loop
- [ ] Show emergency contact numbers in the distress response before any form
- [ ] Evaluate category and sentiment accuracy on a labelled test set; measure distress recall separately
- [ ] Add DeepEval faithfulness checks on the department RAG answers
- [ ] Send escalations to a real channel (email, Slack, WhatsApp) instead of only displaying them
- [ ] Wrap the graph in a Streamlit chat UI

---

## 👤 Author

**Kailas Wadje**
MSc Data Science & AI, University of Liverpool

- GitHub: [@Kailaswadje](https://github.com/Kailaswadje)
- LinkedIn: [linkedin.com/in/kwadaje](https://www.linkedin.com/in/kwadaje/)

---

⭐ If this showed you how routing turns RAG into an agent, consider giving it a star!
