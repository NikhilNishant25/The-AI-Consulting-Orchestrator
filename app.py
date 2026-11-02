import streamlit as st
from openai import OpenAI
from tavily import TavilyClient

SYSTEM_PROMPT = (
    "You are an elite MBB strategy consultant. Take the provided raw market research "
    "and the client's objective, and draft a highly structured Go-To-Market strategy. "
    "Your output must be strictly MECE (Mutually Exclusive, Collectively Exhaustive). "
    "Use first-principles thinking. Format the output with the following headers: "
    "1. Executive Summary, 2. The 3Cs Analysis (Company, Competitor, Customer), "
    "3. Market Realities & Friction Points, 4. The Wedge Strategy (Actionable GTM steps), "
    "5. Unit Economics & Risks. Do not use generic fluff. Be highly analytical, concise, "
    "and data-driven."
)

st.set_page_config(page_title="The AI Consulting Orchestrator", page_icon="🧭", layout="wide")


def run_research(tavily_key: str, client_name: str, industry: str, objective: str) -> str:
    """Agent 1: query Tavily for market dynamics, competitors and news."""
    tavily = TavilyClient(api_key=tavily_key)
    queries = [
        (f"{industry} market size growth trends dynamics {objective}", "general"),
        (f"{industry} top competitors market share pricing {client_name}", "general"),
        (f"{industry} latest news developments", "news"),
    ]

    blocks = []
    for query, topic in queries:
        try:
            response = tavily.search(
                query=query,
                topic=topic,
                search_depth="advanced",
                max_results=5,
            )
        except Exception as exc:  # keep going if one query fails
            blocks.append(f"### Query: {query}\n(Search failed: {exc})")
            continue

        lines = [f"### Query: {query}"]
        for item in response.get("results", []):
            title = item.get("title", "Untitled")
            url = item.get("url", "")
            content = (item.get("content") or "").strip()
            lines.append(f"- **{title}** ({url})\n  {content}")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def run_structuring(openai_key: str, client_name: str, industry: str, objective: str, research: str) -> str:
    """Agent 2: turn raw research into a GTM strategy with gpt-4o-mini."""
    client = OpenAI(api_key=openai_key)
    user_message = (
        f"Client Name: {client_name}\n"
        f"Industry: {industry}\n"
        f"Strategic Objective: {objective}\n\n"
        f"Raw Market Research:\n{research}"
    )
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
    )
    return completion.choices[0].message.content or ""


# ---------- Sidebar ----------
with st.sidebar:
    st.header("🔐 API Keys")
    openai_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    tavily_key = st.text_input("Tavily API Key", type="password", placeholder="tvly-...")
    st.caption("Keys stay in your browser session and are never stored or logged.")

# ---------- Main ----------
st.title("🧭 The AI Consulting Orchestrator")
st.write(
    "Automates junior-analyst workflows: one agent scrapes live market data, "
    "another structures it into a MECE Go-To-Market strategy."
)

col1, col2 = st.columns(2)
with col1:
    client_name = st.text_input("Client Name", placeholder="e.g., Acme Foods")
    industry = st.text_input("Industry", placeholder="e.g., Packaged snacks (FMCG)")
with col2:
    objective = st.text_area(
        "Strategic Objective",
        placeholder="e.g., Market entry into Tier 2 cities",
        height=122,
    )

if st.button("Generate Strategy", type="primary"):
    if not openai_key or not tavily_key:
        st.error("Please enter both your OpenAI and Tavily API keys in the sidebar.")
    elif not (client_name.strip() and industry.strip() and objective.strip()):
        st.error("Please fill in Client Name, Industry and Strategic Objective.")
    else:
        try:
            with st.status("Agent 1: Researching...", expanded=True) as status:
                research = run_research(
                    tavily_key, client_name.strip(), industry.strip(), objective.strip()
                )
                status.update(label="Agent 2: Structuring...")
                strategy = run_structuring(
                    openai_key, client_name.strip(), industry.strip(), objective.strip(), research
                )
                status.update(label="Strategy complete", state="complete", expanded=False)

            st.session_state["strategy"] = strategy
            st.session_state["client_name"] = client_name.strip()
        except Exception as exc:
            st.session_state.pop("strategy", None)
            st.error(f"Something went wrong: {exc}")

if st.session_state.get("strategy"):
    st.divider()
    st.markdown(st.session_state["strategy"])

    file_stub = st.session_state.get("client_name", "client").lower().replace(" ", "_") or "client"
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "⬇️ Download as .md",
            data=st.session_state["strategy"],
            file_name=f"{file_stub}_gtm_strategy.md",
            mime="text/markdown",
        )
    with dl2:
        st.download_button(
            "⬇️ Download as .txt",
            data=st.session_state["strategy"],
            file_name=f"{file_stub}_gtm_strategy.txt",
            mime="text/plain",
        )
