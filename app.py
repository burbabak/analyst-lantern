import csv
import os
import time
import uuid
from datetime import datetime

import streamlit as st
from openai import OpenAI


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Analyst Lantern", page_icon="🕯️", layout="centered")


# -----------------------------
# Config
# -----------------------------
APP_TITLE = "🕯️ Analyst Lantern"
APP_SUBTITLE = "A bounded guide for thinking through analysis."
LOG_FILE = "lantern_log.csv"
MODEL_NAME = "gpt-4.1-mini"
INACTIVITY_TIMEOUT_SECONDS = 15 * 60  # 15 minutes
VECTOR_STORE_ID = "vs_69d12923dfc081919ea0b7d992b6092a" 

# -----------------------------
# OpenAI client
# -----------------------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# -----------------------------
# Helpers
# -----------------------------
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
        st.session_state.session_saved = False


def reset_session() -> None:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.session_number = int(time.time())
    st.session_state.start_time = now_ts()
    st.session_state.last_activity_time = now_ts()
    st.session_state.messages = []
    st.session_state.turn_logs = []
    st.session_state.total_tokens_est = 0
    st.session_state.session_saved = False


def estimate_tokens(text: str) -> int:
    # Rough estimate: ~4 characters per token
    return max(1, len(text) // 4)


def classify_user_message(user_text: str) -> tuple[str, str]:
    prompt = f"""
Classify the student's message in two ways.

1. Domain:
- quant
- qual
- other

2. Thinking type:
- procedural
- interpretive
- other

Definitions:
- procedural = asks about steps, tests, mechanics, what to run, how to do it
- interpretive = asks what findings mean, how to explain results, how to connect results to a research question or argument
- other = anything else

Return ONLY this format:
domain,type

Student message:
{user_text}
""".strip()

    try:
        response = client.responses.create(
            model=MODEL_NAME,
            input=prompt,
            max_output_tokens=10,
        )
        text = response.output_text.strip().lower()
        parts = [p.strip() for p in text.split(",")]
        if len(parts) == 2:
            domain, thinking_type = parts
            if domain not in {"quant", "qual", "other"}:
                domain = "other"
            if thinking_type not in {"procedural", "interpretive", "other"}:
                thinking_type = "other"
            return domain, thinking_type
    except Exception:
        pass

    return "other", "other"


def build_system_prompt() -> str:
    return """
You are the Analyst Lantern, a bounded learning guide for graduate students in a research methods course.

Your role:
- Help students think through quantitative and qualitative analysis tasks.
- Support reasoning, not answer-giving.
- Prioritize interpretation, evaluation, and conceptual understanding.

Strict behavior rules:
- Keep responses short: usually 2–4 sentences.
- Start with a brief orienting explanation (1–2 sentences), then ask a guiding question.
- Do not jump straight to a question without framing.

- When course content is relevant, use it to support reasoning, but do not deliver full answers or complete explanations.
- Do not reproduce full textbook-style answers, even if the source material contains them.

- If a student provides text or an idea, help interpret, clarify, or break it down rather than repeating it.
- Expand slightly when needed to support understanding, but do not over-explain.

- Ask questions frequently, but do not rely only on questions. Combine brief explanation + question.

- If you are about to give a full answer, step back and convert it into a partial explanation followed by a question.
- For qualitative work, help students prioritize and reason, but do not identify all flaws, fill gaps, or fix grammar.
- Keep responses short: usually 2–4 sentences.

- Start with a brief orienting explanation when helpful.

- Do not always ask a question. Use questions selectively when they will help the student think.
- Sometimes provide a short explanation only, especially if the student needs space to process or continue directing the conversation.

- Vary your response style:
  • explanation only
  • explanation + question
  • clarification
  • prompting reflection

- Avoid asking a question at the end of every response.

- When a student seems to be thinking or directing the conversation, follow their lead rather than redirecting with a question.

- When course content is relevant, use it to support reasoning, but do not deliver full answers or complete explanations.

- If a student provides text, help interpret or clarify it rather than repeating it.

- If you are about to give a full answer, step back and provide a partial explanation instead.

- When using course materials, occasionally reference the source in a natural way, such as:
  • “In the ANOVA reading…”
  • “In the lecture on regression…”
  • “One of the course examples shows…”

- Do not use technical citation markers or symbols.
- Do not include file IDs or bracketed references.
- Keep references brief and natural, only when helpful.

Adaptive guidance:
- If the student has not shown thinking, ask a question first.
- If the student shows partial understanding, confirm briefly and nudge them forward.
- Do not remain in endless questioning. Mix brief guidance with questions.
- If you are about to give a direct answer too quickly, convert it into a guiding question.

Tone:
- calm
- supportive
- concise
- slightly mythic, like a lantern guiding someone through darkness
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
                    "vector_store_ids": [VECTOR_STORE_ID],
                }
            ],
            max_output_tokens=180,
        )

        return response.output_text.strip()

    except Exception as e:
        return f"I hit a technical snag. Please try again. ({e})"


def append_turn_log(domain: str, thinking_type: str) -> None:
    st.session_state.turn_logs.append(
        {
            "timestamp": now_iso(),
            "domain": domain,
            "thinking_type": thinking_type,
        }
    )


def compute_session_summary() -> dict:
    end_time = st.session_state.last_activity_time
    duration_seconds = max(0, int(end_time - st.session_state.start_time))
    message_count = sum(1 for m in st.session_state.messages if m["role"] == "user")

    turn_logs = st.session_state.turn_logs
    total_classified_turns = len(turn_logs)

    quant_count = sum(1 for t in turn_logs if t["domain"] == "quant")
    qual_count = sum(1 for t in turn_logs if t["domain"] == "qual")
    procedural_count = sum(1 for t in turn_logs if t["thinking_type"] == "procedural")
    interpretive_count = sum(1 for t in turn_logs if t["thinking_type"] == "interpretive")
    other_count = sum(1 for t in turn_logs if t["thinking_type"] == "other")

    return {
        "session_id": st.session_state.session_id,
        "session_number": st.session_state.session_number,
        "start_timestamp": datetime.utcfromtimestamp(st.session_state.start_time).isoformat(),
        "end_timestamp": datetime.utcfromtimestamp(end_time).isoformat(),
        "duration_seconds": duration_seconds,
        "message_count": message_count,
        "token_count_est": st.session_state.total_tokens_est,
        "classified_turn_count": total_classified_turns,
        "quant_count": quant_count,
        "qual_count": qual_count,
        "procedural_count": procedural_count,
        "interpretive_count": interpretive_count,
        "other_count": other_count,
    }


def save_session_summary() -> None:
    if st.session_state.session_saved:
        return

    row = compute_session_summary()
    file_exists = os.path.exists(LOG_FILE)

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
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
                "quant_count",
                "qual_count",
                "procedural_count",
                "interpretive_count",
                "other_count",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    st.session_state.session_saved = True


def inactivity_expired() -> bool:
    return (now_ts() - st.session_state.last_activity_time) > INACTIVITY_TIMEOUT_SECONDS


# -----------------------------
# Initialize
# -----------------------------
initialize_session_state()


# -----------------------------
# UI
# -----------------------------
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

with st.expander("What this tool does"):
    st.write(
        "The Lantern is a guide, not an answer machine. It helps you think through "
        "analysis questions using short, question-driven support."
    )

col1, col2 = st.columns(2)
with col1:
    if st.button("End Session"):
        save_session_summary()
        st.success("Session saved.")
with col2:
    if st.button("Start New Session"):
        reset_session()
        st.rerun()

if inactivity_expired() and st.session_state.messages and not st.session_state.session_saved:
    save_session_summary()
    st.info("Your previous session was saved after inactivity. A new session will begin when you continue.")


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

    domain, thinking_type = classify_user_message(user_input)
    append_turn_log(domain, thinking_type)

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.total_tokens_est += estimate_tokens(user_input)

    with st.chat_message("user"):
        st.write(user_input)

    try:
        reply = get_lantern_reply(user_input)
    except Exception as e:
        reply = f"I hit a technical snag. Please try again. ({e})"

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.session_state.total_tokens_est += estimate_tokens(reply)
    st.session_state.last_activity_time = now_ts()

    with st.chat_message("assistant"):
        st.write(reply)