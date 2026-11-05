import re

import streamlit as st
from openai import OpenAI
from tavily import TavilyClient

# Gemini is called through Google's OpenAI-compatible endpoint, so the `openai`
# library is reused and only the base URL and key change.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Used only if the live model list cannot be fetched.
FALLBACK_MODELS = ["gemini-flash-latest", "gemini-2.5-flash", "gemini-2.5-flash-lite"]

# Model names containing any of these are not text-generation chat models.
EXCLUDED_TOKENS = (
    "image", "tts", "embedding", "embed", "live", "audio", "aqa", "imagen",
    "veo", "robotics", "computer-use", "learnlm", "vision", "exp", "dialog",
)
MAX_ATTEMPTS = 8  # how many candidate models to try before giving up

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


def get_secret(name: str):
    """Read a key from Streamlit Secrets. Returns None if the secret (or the whole
    secrets file) is missing or empty, instead of crashing the app."""
    try:
        value = st.secrets[name]
    except Exception:  # KeyError, missing secrets.toml, etc.
        return None
    value = str(value).strip() if value is not None else ""
    return value or None


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


def rank_models(model_ids):
    """Order Gemini text models best-first: newest version, then Pro > Flash > Flash-Lite,
    then stable before preview. Models without a version (e.g. *-latest aliases) go last."""
    tiers = {"pro": 0, "flash": 1, "flash-lite": 2}
    scored = []
    for raw in model_ids:
        name = raw.split("/")[-1].lower()
        if not name.startswith("gemini"):
            continue
        if any(token in name for token in EXCLUDED_TOKENS):
            continue
        match = re.match(r"^gemini-(\d+(?:\.\d+)?)-(pro|flash-lite|flash)", name)
        if match:
            version, tier = float(match.group(1)), tiers[match.group(2)]
        else:
            version = 0.0
            tier = 0 if "pro" in name else (2 if "lite" in name else 1)
        is_preview = 1 if "preview" in name else 0
        scored.append(((-version, tier, is_preview, name), raw.split("/")[-1]))
    scored.sort(key=lambda item: item[0])

    ordered, seen = [], set()
    for _, name in scored:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def get_candidate_models(client: OpenAI):
    """Ask Gemini which models this key can see, ranked best-first."""
    try:
        ids = [m.id for m in client.models.list()]
        ranked = rank_models(ids)
        if ranked:
            return ranked
    except Exception:
        pass
    return FALLBACK_MODELS


def is_bad_key_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        getattr(exc, "status_code", None) == 401
        or "api key not valid" in text
        or "api_key_invalid" in text
    )


def run_structuring(gemini_key: str, client_name: str, industry: str, objective: str, research: str):
    """Agent 2: turn raw research into a GTM strategy using the best Gemini model
    that works for this key (free tier or paid). Returns (text, model_used)."""
    client = OpenAI(api_key=gemini_key, base_url=GEMINI_BASE_URL)
    user_message = (
        f"Client Name: {client_name}\n"
        f"Industry: {industry}\n"
        f"Strategic Objective: {objective}\n\n"
        f"Raw Market Research:\n{research}"
    )

    candidates = get_candidate_models(client)

    # Try the model that worked last time first, so repeat runs are fast.
    last_good = st.session_state.get("model_used")
    if last_good and last_good in candidates:
        candidates = [last_good] + [m for m in candidates if m != last_good]

    errors = []
    for model in candidates[:MAX_ATTEMPTS]:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.3,
            )
            text = completion.choices[0].message.content or ""
            if text.strip():
                return text, model
            errors.append(f"{model}: empty response")
        except Exception as exc:
            if is_bad_key_error(exc):
                raise RuntimeError("Your Gemini API key looks invalid. Please check it.") from exc
            errors.append(f"{model}: {str(exc)[:160]}")

    raise RuntimeError(
        "No Gemini model worked for this key. Models tried:\n- " + "\n- ".join(errors)
    )


# ---------- Load keys from Streamlit Secrets ----------
gemini_key = get_secret("GEMINI_API_KEY")
tavily_key = get_secret("TAVILY_API_KEY")
missing = [
    name
    for name, value in (("GEMINI_API_KEY", gemini_key), ("TAVILY_API_KEY", tavily_key))
    if not value
]

# ---------- Main ----------
st.title("🧭 The AI Consulting Orchestrator")
st.write(
    "Automates junior-analyst workflows: one agent scrapes live market data, "
    "another structures it into a MECE Go-To-Market strategy."
)

st.info(
    "🎥 **Want a quick walkthrough?** "
    "[Watch the 90-Second Demo Video](https://www.loom.com/share/501a9de485794d8686dc4b97a70a1a1e) "
    "to see how this tool automates live strategy research for real clients."
)

if missing:
    st.error(
        "App is not configured: missing secret(s) " + ", ".join(f"`{m}`" for m in missing) + "."
    )
    st.markdown(
        "**Fix (Streamlit Community Cloud):** open your app, click **Settings → Secrets**, "
        "paste the following with your real keys, and save:"
    )
    st.code('GEMINI_API_KEY = "AIza..."\nTAVILY_API_KEY = "tvly-..."', language="toml")
    st.markdown(
        "**Fix (running locally):** create `.streamlit/secrets.toml` in your project folder "
        "with the same two lines, then restart the app."
    )
    st.stop()

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
    if not (client_name.strip() and industry.strip() and objective.strip()):
        st.error("Please fill in Client Name, Industry and Strategic Objective.")
    else:
        try:
            with st.status("Agent 1: Researching...", expanded=True) as status:
                research = run_research(
                    tavily_key, client_name.strip(), industry.strip(), objective.strip()
                )
                status.update(label="Agent 2: Structuring...")
                strategy, model_used = run_structuring(
                    gemini_key,
                    client_name.strip(),
                    industry.strip(),
                    objective.strip(),
                    research,
                )
                status.update(label="Strategy complete", state="complete", expanded=False)

            st.session_state["strategy"] = strategy
            st.session_state["model_used"] = model_used
            st.session_state["client_name"] = client_name.strip()
        except Exception as exc:
            st.session_state.pop("strategy", None)
            st.error(f"Something went wrong: {exc}")

if st.session_state.get("strategy"):
    st.divider()
    st.caption(f"Generated with Gemini model: {st.session_state.get('model_used', 'unknown')}")
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
