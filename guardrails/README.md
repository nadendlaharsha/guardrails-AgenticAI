# 🛡️ Production AI Guardrails with LangChain & Groq

A comprehensive, production-ready framework for implementing deterministic and model-based guardrails in Agentic AI workflows. Built using **LangChain**, **LangGraph**, and **Groq** for high-speed, cost-effective, and safe AI execution.

---

## 📌 Architecture Overview

```
                        [ User Input ]
                               │
               ┌───────────────▼───────────────┐
               │ 1. Deterministic Input Hook   │ ──(Violates Policy)──► [ Immediate 400 Refusal ]
               │    (Regex / Keyword Blocker)  │                         (Zero LLM Token Cost)
               └───────────────┬───────────────┘
                               │ (Clean / Allowed)
               ┌───────────────▼───────────────┐
               │ 2. PII Sanitization (In)      │
               │    - Email: [REDACTED]        │
               │    - Credit Card: ****-5100   │
               │    - API Keys: Blocked        │
               └───────────────┬───────────────┘
                               │
               ┌───────────────▼───────────────┐
               │ 3. Core Agent Execution (LLM) │
               │    (Groq: Qwen / LLaMA / OSS) │
               └───────────────┬───────────────┘
                               │ (Tool Calling)
           ┌───────────────────┴───────────────────┐
           ▼                                       ▼
  [ Safe Tool: Search ]              [ Sensitive Tool: Delete/Email ]
           │                                       │
     (Auto-executes)                       ┌───────▼───────────────┐
           │                               │ 4. Human-In-The-Loop  │
           │                               │    (Pause & Review)   │
           │                               └───────┬───────────────┘
           │                                       │ (Approved)
           └───────────────────┬───────────────────┘
                               │
               ┌───────────────▼───────────────┐
               │ 5. Model-Based Output Guard   │ ──(Fails Evaluation)──► [ Safe Fallback Message ]
               │    (LLM-as-a-Judge Hook)      │
               └───────────────┬───────────────┘
                               │ (Approved)
                        [ User Output ]
```

---

## 🚀 Key Guardrail Patterns Implemented

### 1. Deterministic vs. Model-Based Guardrails
* **Deterministic (Rule-based)**: Uses regex patterns and fast string matching. Zero token cost, sub-millisecond execution, completely predictable.
* **Model-Based (LLM-as-a-Judge)**: Evaluates semantic nuances, tone, prompt injections, and contextual intent using fast safety classifiers (`qwen/qwen3.8-27b`, `openai/gpt-oss-safeguard-20b`).

### 2. Built-in PII Detection Middleware (`PIIMiddleware`)
Automatically intercepts user messages before they ever touch the model:
* **Email Redaction**: Replaces sensitive emails with `[REDACTED_EMAIL]`.
* **Payment Masking**: Masks credit cards to prevent data exfiltration (`****-****-****-5100`).
* **Secret Detection**: Raises an immediate security error if secrets like OpenAI/Groq API keys (`sk-...`) are entered.

### 3. Human-In-The-Loop (HITL) Guardrails (`HumanInTheLoopMiddleware`)
* Intercepts dangerous actions before execution (e.g. database deletions, external emails).
* Halts the agent execution graph with state persistence (`InMemorySaver`).
* Resumes execution seamlessly upon receiving an explicit approval (`approve`) or rejection (`reject`) command.

### 4. Input & Output Lifecycle Hooks (`AgentMiddleware`)
* **`before_agent`**: Intercepts requests prior to agent reasoning. Instantly terminates unsafe prompts with `jump_to="end"`, preventing unnecessary token generation.
* **`after_agent`**: Validates the model's generated text with a secondary safety judge model before streaming the answer to the end-user.

---

## 🛠️ Tech Stack & Prerequisites

* **Python 3.11+**
* **LangChain & LangGraph** (StateGraph, Middleware, Pregel engine)
* **Groq SDK & LangChain-Groq** (Low-latency inference)
* **python-dotenv**

### Dependencies

```bash
uv pip install langchain langchain-groq langchain-openai python-dotenv
# or
pip install -r requirements.txt
```

---

## ⚙️ Environment Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY="gsk_..."
OPENAI_API_KEY="sk-..."  # Optional
```

---

## 🧪 Quickstart Examples

### 1. PII Masking & Redaction

```python
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langchain_core.tools import tool

@tool
def customer_lookup(query: str) -> str:
    """Look up customer records."""
    return f"Customer record located for: {query}"

agent = create_agent(
    model="groq:qwen/qwen3.8-27b",
    tools=[customer_lookup],
    middleware=[
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
        PIIMiddleware("api_key", detector=r"sk-[a-zA-Z0-9]{32}", strategy="block", apply_to_input=True),
    ],
)
```

### 2. Human-In-The-Loop Middleware

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

hitl_agent = create_agent(
    model="groq:qwen/qwen3.8-27b",
    tools=[send_email, delete_records, search_web],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": True,       # Requires review
                "delete_records": True,   # Requires review
                "search_web": False,      # Auto-approved
            }
        )
    ],
    checkpointer=InMemorySaver(),
)

# Step 1: Trigger action (pauses before execution)
config = {"configurable": {"thread_id": "session_001"}}
hitl_agent.invoke({"messages": [{"role": "user", "content": "Delete records from db"}]}, config=config)

# Step 2: Resume with decision
hitl_agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config=config)
```

### 3. Pre-Agent Hook (Zero-Cost Content Filter)

```python
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langgraph.runtime import Runtime

class ContentFilterMiddleware(AgentMiddleware):
    def __init__(self, banned_keywords: list[str]):
        super().__init__()
        self.banned_keywords = [kw.lower() for kw in banned_keywords]

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime):
        content = state["messages"][0].content.lower()
        if any(kw in content for kw in self.banned_keywords):
            return {
                "messages": [{"role": "assistant", "content": "Policy violation: request blocked."}],
                "jump_to": "end"
            }
        return None
```

---

## 📈 Performance & Cost Optimization

| Guardrail Type | Latency Overhead | Token Cost | Use Case |
|---|---|---|---|
| **Deterministic (`before_agent`)** | `< 1ms` | **0 tokens** | Keyword/IP blocking, rate limits |
| **PII Middleware** | `< 2ms` | **0 tokens** | Email & payment card sanitization |
| **Model-Based (`after_agent`)** | `~200ms` (via Groq) | Minimal | Toxicity, Hallucination, Compliance |
| **Human-In-The-Loop** | Asynchronous | **0 tokens** | High-risk DB mutations & external transactions |

---

## 📄 License
MIT License
