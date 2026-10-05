from __future__ import annotations

import streamlit as st

import html
import uuid

from companion.service import CompanionService
from companion.settings import ROOT_DIR, CompanionSettings
from companion.voice import SentenceStreamer, split_sentences, strip_tags


st.set_page_config(page_title="Companion", page_icon="✦", layout="centered")
st.markdown(
    """<style>
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
/* Voice plays through hidden audio elements: speech is heard, never shown. */
audio { display: none !important; }
.stApp {
    background:
        radial-gradient(900px 500px at 15% -5%, rgba(124, 58, 237, 0.14), transparent 60%),
        radial-gradient(800px 460px at 90% 10%, rgba(34, 211, 238, 0.10), transparent 60%),
        radial-gradient(700px 700px at 50% 110%, rgba(236, 72, 153, 0.08), transparent 60%);
}
.companion-hero { text-align: center; padding: 1.6rem 1rem 0.4rem; }
.companion-title {
    font-size: 2.6rem; font-weight: 800; letter-spacing: -0.02em; margin: 0;
    background: linear-gradient(92deg, #c4b5fd, #67e8f9 55%, #f9a8d4);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.companion-sub { opacity: 0.72; margin-top: 0.35rem; font-size: 1.02rem; }
.pills { display: flex; gap: 0.45rem; justify-content: center; flex-wrap: wrap; margin-top: 0.9rem; }
.pill {
    font-size: 0.78rem; padding: 0.28rem 0.7rem; border-radius: 999px;
    border: 1px solid rgba(148, 163, 184, 0.25); background: rgba(148, 163, 184, 0.10);
}
.pill.ok { border-color: rgba(52, 211, 153, 0.4); background: rgba(52, 211, 153, 0.10); }
.pill.warn { border-color: rgba(251, 191, 36, 0.45); background: rgba(251, 191, 36, 0.10); }
.sugg-title { text-align: center; opacity: 0.65; margin: 1.6rem 0 0.6rem; font-size: 0.9rem; }
[data-testid="stSidebar"] { border-right: 1px solid rgba(148, 163, 184, 0.14); }
[data-testid="stChatMessage"] {
    border-radius: 1.1rem; border: 1px solid rgba(148, 163, 184, 0.14);
    padding: 0.9rem 1.05rem;
}
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"] { border-radius: 0.8rem; }
[data-testid="stChatInput"] { border-radius: 1.2rem; }
.stButton > button { border-radius: 0.8rem; }
[data-testid="stExpander"] { border-radius: 0.9rem; }
@media (prefers-color-scheme: light) {
    .stApp {
        background:
            radial-gradient(900px 500px at 15% -5%, rgba(124, 58, 237, 0.08), transparent 60%),
            radial-gradient(800px 460px at 90% 10%, rgba(8, 145, 178, 0.08), transparent 60%);
    }
}
</style>""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_service(model_name: str, user_id: str) -> CompanionService:
    return CompanionService(CompanionSettings(model_name=model_name, user_id=user_id))


if "voice_on" not in st.session_state:
    st.session_state.voice_on = True
if "turn" not in st.session_state:
    st.session_state.turn = 0
if "run_id" not in st.session_state:
    st.session_state.run_id = uuid.uuid4().hex
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

with st.sidebar:
    st.header("Companion Settings")
    model_name = st.text_input("Ollama model", value="companion-gemma")
    user_id = st.text_input("Memory profile", value="default_user")
    service = get_service(model_name, user_id)
    healthy, status = service.health()
    if healthy:
        st.success(status)
    else:
        st.warning(status)
    st.checkbox("Speak replies aloud", key="voice_on")
    voice_healthy, voice_status = service.voice.health()
    if st.session_state.voice_on and not voice_healthy:
        _, voice_status = service.voice.ensure_server(ROOT_DIR)
    if st.session_state.voice_on:
        if voice_healthy:
            st.success("Voice ready.")
        else:
            st.warning(voice_status)
    if st.button("Clear chat display"):
        st.session_state.messages = []
        st.rerun()
    with st.expander("About this companion"):
        st.markdown(
            "- Private by design: Gemma, memory, and voice all run on your machine.\n"
            "- Remembers your tastes across restarts (Mem0 + graph memory).\n"
            "- Speaks replies aloud with Fish Audio S2 when voice is on."
        )
    st.caption("Private • Local • Yours")


def _pill(label: str, tone: str = "") -> str:
    return f'<span class="pill {tone}">{html.escape(label)}</span>'


_voice_label = "Voice on" if st.session_state.voice_on and voice_healthy else (
    "Voice starting…" if st.session_state.voice_on else "Voice off"
)
_voice_tone = "ok" if st.session_state.voice_on and voice_healthy else (
    "warn" if st.session_state.voice_on else ""
)
st.markdown(
    '<div class="companion-hero">'
    '<div class="companion-title">✦ Companion</div>'
    '<div class="companion-sub">Your private local AI — remembers you, speaks to you.</div>'
    '<div class="pills">'
    + _pill(f"Model · {model_name}")
    + _pill(f"Profile · {user_id}")
    + _pill(_voice_label, _voice_tone)
    + _pill("Memory ready" if healthy else "Memory offline", "ok" if healthy else "warn")
    + "</div></div>",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for item in st.session_state.messages:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])
        if "retrieved_context" in item:
            with st.expander("Retrieved memory", expanded=False):
                st.write("Vector / hybrid memories")
                st.write(item["retrieved_context"]["memories"] or ["No matching memories."])
                st.write("Graph memories")
                st.write(item["retrieved_context"]["graph_facts"] or ["No related graph facts."])
        if item.get("role") == "assistant" and item.get("voice") and st.session_state.voice_on:
            # One hidden player per reply. Args never change after mount, so
            # reruns preserve playback: no restarts, no overlaps.
            st.audio(item["voice"], format="audio/wav", autoplay=True)


@st.fragment(run_every=2)
def show_background_updates() -> None:
    received_update = False
    for update in service.pop_supplements():
        received_update = True
        if "error" in update:
            st.session_state.messages.append({"role": "assistant", "content": f"Memory lookup failed: {update['error']}"})
        else:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": update["content"],
                    "retrieved_context": update["context"],
                }
            )
    for track in service.voice.pop_ready():
        for item in reversed(st.session_state.messages):
            if (
                item.get("role") == "assistant"
                and item.get("run") == track["run"]
                and item.get("turn") == track["turn"]
            ):
                item["voice"] = track["audio"]
                received_update = True
                break
    if received_update:
        # A completed worker has updated session state; redraw the full conversation immediately.
        st.rerun(scope="app")


show_background_updates()

if not st.session_state.messages:
    st.markdown('<div class="sugg-title">Try asking…</div>', unsafe_allow_html=True)
    _sugg_cols = st.columns(3)
    for _i, (_col, _text) in enumerate(
        zip(
            _sugg_cols,
            [
                "What do you remember about me?",
                "Recommend something for tonight",
                "Plan my perfect weekend",
            ],
        )
    ):
        if _col.button(_text, key=f"sugg_{_i}", use_container_width=True):
            st.session_state.pending_prompt = _text

typed_prompt = st.chat_input("Talk to your companion")
prompt = st.session_state.pending_prompt or typed_prompt
st.session_state.pending_prompt = None
if prompt:
    if not healthy:
        st.error("Start Ollama and create the model shown in the README before chatting.")
        st.stop()

    history_before = list(st.session_state.messages)
    turn = st.session_state.turn
    st.session_state.turn += 1
    voice_active = st.session_state.voice_on and voice_healthy
    if voice_active:
        service.voice.new_turn(st.session_state.run_id, turn)
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if service.needs_memory(prompt):
        # Lookup-first path: a human-style thinking-aloud line appears instantly
        # while memory is checked in the background, then the grounded answer
        # streams in underneath it. No guessing first, and never a blank reply.
        filler = service.pick_filler(prompt)
        with st.chat_message("assistant"):
            st.markdown(filler)
        st.session_state.messages.append(
            {"role": "assistant", "content": filler, "turn": turn, "run": st.session_state.run_id}
        )

        context: dict[str, list[str]] = {"memories": [], "graph_facts": []}
        full = ""
        error_note = ""
        with st.chat_message("assistant"):
            answer_placeholder = st.empty()
            streamer = SentenceStreamer()
            voice_idx = 0
            try:
                context = service.retrieve(prompt)
                for piece in service.stream_reply_with_context(prompt, history_before, context):
                    full += piece
                    answer_placeholder.markdown(strip_tags(full) + "▌")
                    if voice_active:
                        for sentence in streamer.feed(piece):
                            service.voice.submit(voice_idx, sentence)
                            voice_idx += 1
                full = full.strip()
                if voice_active:
                    for sentence in streamer.flush():
                        service.voice.submit(voice_idx, sentence)
                        voice_idx += 1
                if not strip_tags(full):
                    full = service.memory_fallback_answer(context)
                    error_note = "Polished answer came back empty, so I answered from saved facts instead."
                    if voice_active:
                        for sentence in split_sentences(full):
                            service.voice.submit(voice_idx, sentence)
                            voice_idx += 1
            except Exception as error:
                full = service.memory_fallback_answer(context)
                error_note = f"Live generation failed ({error}); answered from saved facts."
                if voice_active:
                    for sentence in split_sentences(full):
                        service.voice.submit(voice_idx, sentence)
                        voice_idx += 1
            clean = strip_tags(full)
            answer_placeholder.markdown(clean)
            if error_note:
                st.caption(error_note)
            with st.expander("Retrieved memory", expanded=False):
                st.write("Vector / hybrid memories")
                st.write(context["memories"] or ["No matching memories."])
                st.write("Graph memories")
                st.write(context["graph_facts"] or ["No related graph facts."])

        if voice_active:
            service.voice.finish_turn()
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": clean,
                "retrieved_context": context,
                "turn": turn,
                "run": st.session_state.run_id,
            }
        )
        service.queue_ingestion(prompt, clean)
    else:
        # Streaming path: words appear as generated and each finished sentence
        # is spoken while the next one generates. No 45s wall of silence.
        reply = ""
        with st.chat_message("assistant"):
            text_placeholder = st.empty()
            streamer = SentenceStreamer()
            voice_idx = 0
            try:
                for piece in service.stream_reply(prompt, history_before):
                    reply += piece
                    text_placeholder.markdown(strip_tags(reply) + "▌")
                    if voice_active:
                        for sentence in streamer.feed(piece):
                            service.voice.submit(voice_idx, sentence)
                            voice_idx += 1
                reply = reply.strip()
                if not strip_tags(reply):
                    # Stream came back empty; fall back to one-shot generation.
                    reply = service.reply(prompt, history_before)
                if voice_active:
                    pending = streamer.flush() if strip_tags(reply) else split_sentences(reply)
                    for sentence in pending:
                        service.voice.submit(voice_idx, sentence)
                        voice_idx += 1
            except Exception as error:
                st.error(f"Companion error: {error}")
                st.stop()
            clean_reply = strip_tags(reply) or "Hmm, I lost my train of thought there — could you say that again?"
            text_placeholder.markdown(clean_reply)

        if voice_active:
            service.voice.finish_turn()
        # The user message was already appended above; only the reply is new here.
        st.session_state.messages.append(
            {"role": "assistant", "content": clean_reply, "turn": turn, "run": st.session_state.run_id}
        )
        service.queue_ingestion(prompt, clean_reply)
        service.queue_lookup(prompt, st.session_state.messages, clean_reply)
