import csv
import os
import time
import uuid
from datetime import datetime

import streamlit as st
from openai import OpenAI
from supabase import create_client

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Analyst's Lantern",
    page_icon="lantern.png",
    layout="centered"
)


# -----------------------------
# Config
# -----------------------------
APP_TITLE = "Analyst's Lantern"
APP_SUBTITLE = "A bounded guide for thinking through analysis."
SESSIONS_FILE = "sessions.csv"
TURNS_FILE = "turns.csv"
MODEL_NAME = "gpt-4.1-mini"
INACTIVITY_TIMEOUT_SECONDS = 15 * 60  # 15 minutes
PRIMARY_VECTOR_STORE_ID = "vs_69fe900dba6c8191aefd13bec3ae0e11"
SECONDARY_VECTOR_STORE_ID = "vs_69fe93be10308191aee7e02cc5f4cb8e"

# -----------------------------
# OpenAI client
# -----------------------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# -----------------------------
# Helpers
# -----------------------------
@st.cache_resource
def get_supabase_client():
    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_SERVICE_KEY"]
    )

def now_ts() -> float:
    return time.time()


def now_iso() -> str:
    return datetime.utcnow().isoformat()


def initialize_session_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.session_number = int(time.time())
        st.session_state.start_time = now_ts()
        st.session_state.last_activity_time = now_ts()
        st.session_state.messages = []
        st.session_state.turn_logs = []
        st.session_state.total_tokens_est = 0
        st.session_state.turn_number = 0
        st.session_state.session_saved = False


def reset_session() -> None:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.session_number = int(time.time())
    st.session_state.start_time = now_ts()
    st.session_state.last_activity_time = now_ts()
    st.session_state.messages = []
    st.session_state.turn_logs = []
    st.session_state.total_tokens_est = 0
    st.session_state.turn_number = 0
    st.session_state.session_saved = False


def estimate_tokens(text: str) -> int:
    # Rough estimate: ~4 characters per token
    return max(1, len(text) // 4)

import base64
import requests

def persist_file_to_github(local_path: str, repo_path: str) -> None:
    token = st.secrets.get("GITHUB_TOKEN", None)
    owner = st.secrets.get("GITHUB_REPO_OWNER", "burbabak")
    repo = st.secrets.get("GITHUB_REPO_NAME", "analyst-lantern")
    branch = st.secrets.get("GITHUB_BRANCH", "main")

    if not token:
        return

    if not os.path.exists(local_path):
        return

    with open(local_path, "rb") as f:
        content = base64.b64encode(f.read()).decode("utf-8")

    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{repo_path}"

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }

    # Get existing file SHA if it exists
    get_resp = requests.get(
        url,
        headers=headers,
        params={"ref": branch},
        timeout=10,
    )

    sha = None
    if get_resp.status_code == 200:
        sha = get_resp.json().get("sha")

    payload = {
        "message": f"Update {repo_path}",
        "content": content,
        "branch": branch,
    }

    if sha:
        payload["sha"] = sha

    put_resp = requests.put(
        url,
        headers=headers,
        json=payload,
        timeout=10,
    )

    if put_resp.status_code not in (200, 201):
        print("GitHub persistence failed:", put_resp.status_code, put_resp.text)
        
def classify_user_message(user_text: str) -> tuple[str, str]:
    text = user_text.lower()

    quant_terms = [
        "p value", "p-value", "significant", "alpha", "null", "hypothesis",
        "anova", "mlr", "regression", "logistic", "coefficient", "beta",
        "odds", "vif", "tolerance", "durbin", "watson", "residual",
        "normality", "homoscedasticity", "assumption", "spss",
        "mean", "standard deviation", "correlation", "ancova", "manova"
    ]

    qual_terms = [
        "qualitative", "interview", "theme", "coding", "codebook",
        "credibility", "dependability", "confirmability", "transferability",
        "trustworthiness", "reflexivity", "positionality", "memo",
        "thematic", "phenomenology", "grounded theory", "case study",
        "data collection", "participant", "transcript"
    ]

    procedural_terms = [
        "how do i", "how to", "steps", "run", "do this", "calculate",
        "check", "threshold", "assumptions", "what test", "which test"
    ]

    interpretive_terms = [
        "mean", "means", "interpret", "explain", "write up",
        "what does", "so what", "results mean", "connect", "rq",
        "research question"
    ]

    evaluative_terms = [
        "right", "correct", "appropriate", "good", "bad", "strong",
        "weak", "adequate", "aligned", "defensible", "justified",
        "critique", "evaluate", "best", "should i"
    ]

    if any(term in text for term in quant_terms):
        domain = "quant"
    elif any(term in text for term in qual_terms):
        domain = "qual"
    else:
        domain = "other"

    if any(term in text for term in evaluative_terms):
        thinking_type = "evaluative"
    elif any(term in text for term in interpretive_terms):
        thinking_type = "interpretive"
    elif any(term in text for term in procedural_terms):
        thinking_type = "procedural"
    else:
        thinking_type = "other"

    return domain, thinking_type


def build_system_prompt() -> str:
    return """
You are the Analyst Lantern, a bounded learning guide for graduate students in a research methods course.

Your role:
- Help students think through quantitative and qualitative analysis tasks.
- Support reasoning, not answer-giving.
- Prioritize interpretation, evaluation, and conceptual understanding.
- Use course materials first when relevant.

Core response style:
- Keep responses short: usually 2–4 sentences.
- Start with a brief orienting explanation when helpful.
- Do not always ask a question.
- Use questions selectively when they help thinking.
- Sometimes provide explanation only and allow the student to continue.
- Vary your responses: explanation, clarification, or prompting.

Course-grounded behavior:
- Use course materials before general knowledge when possible.
- Reference sources naturally when helpful (e.g., "In the regression lecture...").
- Do NOT use technical citations, file IDs, or bracketed references.

Source priority:
- First use required course materials from the primary source set.
- Use optional materials only when the required course materials do not provide enough support.
- If drawing from optional materials, briefly say so in natural language.
- Do not imply optional materials are required readings.

Scaffolding intensity:
- Give about 25% less direct help than you are inclined to give.
- Prefer hints, distinctions, and next-step prompts over direct conclusions.
- Do not provide the final interpretation unless the student has already clearly reasoned to it.
- When explaining, stop one step before the full answer.
- For assignment-like questions, answer with a thinking frame, not the completed response.

----------------------------------------
BOUNDARY & SCAFFOLDING RULES
----------------------------------------

Your job is to support reasoning, not complete academic work.

GENERAL RESPONSE LIMITS
- Keep responses short and incomplete by design.
- Give ONE idea, ONE distinction, or ONE next step at a time.
- Do not give full lists of assumptions, strategies, criteria, or steps unless the student has already identified most of them.
- Avoid exhaustive teaching responses.
- Stop one step before the final answer.

CONFIRMATION RULE
- If a student demonstrates reasoning and reaches a correct conclusion, you may briefly confirm.
- Confirmation must be short and conceptual, not a completed answer.
- Example: “Yes — that interpretation is on the right track because tolerance reflects overlap among predictors.”
- Do NOT continue prolonged Socratic questioning after a student has already reasoned correctly.
- If the student is guessing, uncertain, or asking for an answer without reasoning, do not confirm directly. Ask them to explain their thinking first.

NO COMPLETION OF ACADEMIC WORK
- Do NOT write assignment-ready sentences, paragraphs, critiques, or interpretations.
- Do NOT provide model wording students could lightly adapt and submit.
- Do NOT convert refused writing tasks into bullet points, outlines, or lightly disguised versions of the answer.
- If a student asks for writing help, explain structure or reasoning only.

QUANTITATIVE GUIDANCE
- Help students interpret concepts, not produce final interpretations.
- Do not directly state:
  • whether a result is significant
  • whether an assumption is met
  • whether the null hypothesis should be rejected
- Instead, help students compare values, thresholds, and conceptual meaning.
- Do not provide full assumption checklists or threshold lists unless the student is already working through a specific assumption.

QUALITATIVE GUIDANCE
- Be more restrictive in qualitative work than quantitative work.
- Do NOT:
  • choose strategies
  • complete critiques
  • identify all flaws
  • recommend specific options
  • rewrite passages
  • generate editorial or assignment language
- Focus on ONE issue, ONE criterion, or ONE conceptual lens at a time.
- Redirect students back to readings, lectures, frameworks, or criteria whenever possible.
- If the student persists in seeking completion help, become more restrictive rather than more helpful.

WHEN STUDENTS PROVIDE TEXT
- Clarify, unpack, or interpret ideas conceptually.
- Do not rewrite, polish, or improve wording for submission.
- If critique is requested, ask the student to identify one issue first before discussing it.

HIGH-RISK SITUATIONS
Be especially careful when students ask about:
- p-values and significance
- assumptions and thresholds
- “is this right?”
- critique/evaluation tasks
- assignment wording
- choosing among qualitative strategies

In these situations:
- reduce directness further
- avoid full explanations
- prioritize reasoning prompts over conclusions

----------------------------------------
TONE
----------------------------------------

- Calm
- Supportive
- Concise
- Slightly guiding (like a lantern, not an answer key)
""".strip()


def build_conversation_input(user_text: str) -> list[dict]:
    conversation = [{"role": "system", "content": build_system_prompt()}]

    # Include limited recent context
    for msg in st.session_state.messages[-8:]:
        conversation.append({"role": msg["role"], "content": msg["content"]})

    conversation.append({"role": "user", "content": user_text})
    return conversation


def get_lantern_reply(user_text: str) -> str:
    try:
        response = client.responses.create(
            model=MODEL_NAME,
            input=build_conversation_input(user_text),
            tools=[
                {
                    "type": "file_search",
                    "vector_store_ids": [
    PRIMARY_VECTOR_STORE_ID,
    SECONDARY_VECTOR_STORE_ID
],
                }
            ],
            max_output_tokens=300,
        )

        return response.output_text.strip()

    except Exception as e:
        return f"I hit a technical snag. Please try again. ({e})"


def append_turn_log(
    user_text: str,
    assistant_text: str,
    domain: str,
    thinking_type: str,
) -> None:
    input_token_est = estimate_tokens(user_text)
    output_token_est = estimate_tokens(assistant_text)
    total_token_est = input_token_est + output_token_est

    row = {
        "session_id": st.session_state.session_id,
        "session_number": st.session_state.session_number,
        "turn_number": st.session_state.turn_number,
        "timestamp": now_iso(),
        "user_text": user_text,
        "assistant_text": assistant_text,
        "domain": domain,
        "thinking_type": thinking_type,
        "input_token_est": input_token_est,
        "output_token_est": output_token_est,
        "total_token_est": total_token_est,
    }

    st.session_state.turn_logs.append(row)

    supabase = get_supabase_client()
    supabase.table("turns").insert(row).execute()
    save_session_summary(mark_saved=False)


def compute_session_summary() -> dict:
    end_time = st.session_state.last_activity_time
    duration_seconds = max(0, int(end_time - st.session_state.start_time))
    message_count = sum(1 for m in st.session_state.messages if m["role"] == "user")

    return {
        "session_id": st.session_state.session_id,
        "session_number": st.session_state.session_number,
        "start_timestamp": datetime.utcfromtimestamp(st.session_state.start_time).isoformat(),
        "end_timestamp": datetime.utcfromtimestamp(end_time).isoformat(),
        "duration_seconds": duration_seconds,
        "message_count": message_count,
        "token_count_est": st.session_state.total_tokens_est,
        "classified_turn_count": len(st.session_state.turn_logs),
    }


def save_session_summary(mark_saved: bool = True) -> None:
    if st.session_state.session_saved and mark_saved:
        return

    row = compute_session_summary()

    supabase = get_supabase_client()
    supabase.table("sessions").upsert(row, on_conflict="session_id").execute()

    if mark_saved:
        st.session_state.session_saved = True

def inactivity_expired() -> bool:
    return (now_ts() - st.session_state.last_activity_time) > INACTIVITY_TIMEOUT_SECONDS

def ensure_log_files() -> None:
    if not os.path.exists(SESSIONS_FILE):
        with open(SESSIONS_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "session_id",
                    "session_number",
                    "start_timestamp",
                    "end_timestamp",
                    "duration_seconds",
                    "message_count",
                    "token_count_est",
                    "classified_turn_count",
                ],
            )
            writer.writeheader()

    if not os.path.exists(TURNS_FILE):
        with open(TURNS_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "session_id",
                    "session_number",
                    "turn_number",
                    "timestamp",
                    "user_text",
                    "assistant_text",
                    "domain",
                    "thinking_type",
                    "input_token_est",
                    "output_token_est",
                    "total_token_est",
                ],
            )
            writer.writeheader()
# -----------------------------
# Initialize
# -----------------------------
ensure_log_files()
initialize_session_state()


# -----------------------------
# UI
# -----------------------------
st.image("lantern.png", width=80)
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

st.caption(
    "The Lantern is a guide, not an answer machine. It helps you think through analysis "
    "using short, question-driven support, but does not complete assignments or give final answers. "
    "Basic usage data may be recorded to improve the tool; no personal information is collected."
)
if not st.session_state.messages:
    st.info("Light the lantern by asking a question about your analysis.")

st.divider()

if st.button("End Session"):
    save_session_summary()
    st.success("Your path has been illuminated... but use the lantern any time you need!")
    reset_session()
    st.rerun()


if inactivity_expired() and st.session_state.messages and not st.session_state.session_saved:
    save_session_summary()
    st.info("Your path has been illuminated... but use the lantern any time you need!")


# Display prior messages
for msg in st.session_state.messages:
    with st.chat_message("assistant" if msg["role"] == "assistant" else "user"):
        st.write(msg["content"])

# -----------------------------
# Chat input
# -----------------------------
user_input = st.chat_input("Ask the Lantern a question...")

if user_input:
    # If prior session was auto-saved after inactivity, start fresh first
    if st.session_state.session_saved:
        reset_session()

    st.session_state.last_activity_time = now_ts()

    with st.chat_message("user"):
        st.write(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})

    domain, thinking_type = classify_user_message(user_input)

    try:
        reply = get_lantern_reply(user_input)
    except Exception as e:
        reply = f"I hit a technical snag. Please try again. ({e})"

    st.session_state.messages.append({"role": "assistant", "content": reply})

    with st.chat_message("assistant"):
        st.write(reply)

    st.session_state.turn_number += 1

    input_token_est = estimate_tokens(user_input)
    output_token_est = estimate_tokens(reply)
    st.session_state.total_tokens_est += input_token_est + output_token_est
    st.session_state.last_activity_time = now_ts()
    append_turn_log(
    user_text=user_input,
    assistant_text=reply,
    domain=domain,
    thinking_type=thinking_type,
    )

if st.query_params.get("admin") == "true":
    with st.sidebar.expander("Instructor data export"):
        export_password = st.text_input(
            "Instructor password",
            type="password",
            key="export_password"
        )

    if export_password == st.secrets.get("EXPORT_PASSWORD", ""):
        if os.path.exists(SESSIONS_FILE):
            with open(SESSIONS_FILE, "rb") as f:
                st.download_button(
                    "Download sessions.csv",
                    f,
                    file_name="sessions.csv",
                    mime="text/csv"
                )

        if os.path.exists(TURNS_FILE):
            with open(TURNS_FILE, "rb") as f:
                st.download_button(
                    "Download turns.csv",
                    f,
                    file_name="turns.csv",
                    mime="text/csv"
                )
    elif export_password:
        st.error("Incorrect password.")

