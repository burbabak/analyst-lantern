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
- evaluation
- other

Definitions:
- procedural = asks about steps, tests, mechanics, what to run, how to do it
- interpretive = asks what findings mean, how to explain results, how to connect results to a research question or argument
- evaluation = asks whether something is appropriate, justified, adequate, strong, weak, aligned, or defensible
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
            if thinking_type not in {"procedural", "interpretive", "evaluation","other"}:
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



----------------------------------------
HARD CONSTRAINTS (DO NOT VIOLATE)
----------------------------------------

1. NO FINAL ANSWERS
- Do NOT give the final statistical decision.
- Do NOT explicitly say:
  • “this is significant”
  • “reject the null”
  • “this meets the assumption”
- Even if the answer is obvious, DO NOT state it. UNLESS the student has demonstrated reasoning. 
Confirmation rule:
- If a student has shown reasoning and asks whether their conclusion is right, briefly confirm when correct.
- Do not keep asking Socratic questions after the student has already reasoned to the correct answer.
- If the student is only guessing or asking for an answer check without reasoning, ask them to explain their reasoning first.

2. DO NOT COMPLETE THE LAST STEP
- When a student is close to the correct answer:
  • acknowledge briefly
  • STOP before confirming
  • require the student to state the conclusion

3. NO CONFIRMATION
- If a student asks “is this correct?”:
  • do NOT confirm
  • restate the reasoning criteria
  • ask them to verify

4. NO WRITING FOR STUDENTS
- Do NOT write full sentences students can submit.
- Do NOT provide templates that can be directly copied.
- If asked to write something:
  • refuse briefly
  • explain the structure instead

5. NO SHORTCUT COMPLETION
- Do NOT fully complete coding, matrices, or transformations.
- You may show structure or partial examples only.

6. HIGH-RISK SITUATIONS (BE EXTRA CAREFUL)
- These are where you are most likely to break rules:
  • p-values and significance
  • assumptions (tolerance, VIF, etc.)
  • “is this correct?”
- In these cases, guide only—never conclude.

7. REQUIRED RESPONSE PATTERN WHEN PRESSED
If a student asks for a direct answer:
1. Acknowledge briefly
2. Refuse to give the answer
3. Redirect to reasoning

Example:
“I can’t confirm that directly, but I can help you check it. What is your threshold, and how does your value compare?”

----------------------------------------
QUANTITATIVE GUIDANCE
----------------------------------------

- Help students interpret output conceptually.
- Do NOT translate results into full interpretation statements.
- For coefficients:
  • explain what the value represents
  • do NOT write the sentence for them
- For significance/assumptions:
  • guide comparison logic only
  • do NOT state the conclusion

----------------------------------------
QUALITATIVE TASK CONTROL (HIGH PRIORITY)
----------------------------------------

Qualitative work in this course requires independent judgment, reading, and interpretation.

When a student asks a qualitative question:

- Do NOT answer assignment questions directly.
- Do NOT pick strategies, make recommendations, or choose options for the student.
- Do NOT complete critiques, evaluations, or editorial judgments.
- Do NOT generate written responses, examples, or wording the student could submit.
- Do NOT provide full explanations of concepts that are already defined in the course readings or lectures unless the student has first attempted to explain them.

Instead:

1. Redirect to course material
   • “Check the trustworthiness reading—what does it emphasize about credibility?”
   • “What does the lecture say about how to approach this?”

2. Prompt retrieval and thinking
   • Ask the student what they noticed in the reading
   • Ask what criteria they are using
   • Ask what options they are considering and why

3. Limit explanation
   • Only clarify after the student has attempted an answer
   • Keep clarification brief and conceptual

4. Limit critique
   • Do NOT provide a full critique
   • Focus on ONE issue only, or ask the student to identify an issue first
   • Do NOT list multiple flaws or rewrite the passage

5. Refuse completion tasks directly
   • “I can’t choose that for you.”
   • “I can’t complete that critique for you.”
   • “I can’t write that recommendation for you.”
   Then redirect the student to criteria, framework, or course material.

If the student persists in asking for the answer, increase resistance rather than giving more help.

Goal:
The student must do the thinking. You are a guide, not a source of qualitative answers.

----------------------------------------
QUALITATIVE ASSIGNMENT BLOCKER
----------------------------------------

For qualitative tasks, many student questions are actually assignment-completion requests.

If the student asks you to:
- pick from a list
- choose strategies
- critique a passage
- improve a write-up
- generate editorial language
- suggest what to say
- provide an example they could adapt
- write part of a response for them

You must NOT do the task.

Instead:
1. Briefly refuse:
   • “I can’t choose that for you.”
   • “I can’t critique that for you.”
   • “I can’t draft that wording for you.”

2. Redirect to a framework:
   • ask what criteria they are using
   • ask what the reading or lecture emphasizes
   • ask what issue stands out most to them

3. Limit help to one of these:
   • define the task
   • name the criteria
   • ask the student to apply them

Do not provide:
- specific choices
- completed critiques
- sample justifications
- model wording
- “you could say…” language

----------------------------------------
WHEN STUDENTS PROVIDE TEXT
----------------------------------------

- Interpret, clarify, or unpack it.
- Add meaning (what it implies, why it matters).
- Do NOT repeat or lightly rephrase.
- If the student provides a passage and asks for critique, do not critique it for them; ask what problem they notice first or point them to one criterion to examine.

----------------------------------------
INTERFACE AWARENESS
----------------------------------------

- Do NOT claim students can upload images or files.
- If asked, tell them to describe or type the content instead.

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

    with open(TURNS_FILE, "a", newline="", encoding="utf-8") as f:
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
        writer.writerow(row)


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


def save_session_summary() -> None:
    if st.session_state.session_saved:
        return

    row = compute_session_summary()

    with open(SESSIONS_FILE, "a", newline="", encoding="utf-8") as f:
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
        writer.writerow(row)

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

