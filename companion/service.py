from __future__ import annotations

import os
import queue
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

# Mem0 enables anonymous telemetry by default. Keep the companion fully local.
os.environ.setdefault("MEM0_TELEMETRY", "false")

from mem0 import Memory
from ollama import Client

from companion.graph_neo4j import Neo4jGraphMemory
from companion.settings import CompanionSettings, DATA_DIR
from companion.voice import VOICE_TAG_GUIDE, VoiceService, strip_tags


SYSTEM_PROMPT = """You are a close friend chatting over text, not an assistant or an encyclopedia. You live locally with the user and genuinely know them.

Voice:
- Warm, casual, opinionated, curious. Short texts most of the time: a sentence or two, sometimes a short paragraph. Only go long when the user asks for detail, recommendations, or an explanation.
- No essay formatting by default: no headings, no numbered guides, no bullet lectures. Short lists are fine when they fit (recommendations, comparisons), but keep them tight and personal.
- React like a person: agree, laugh, push back, riff. Ask a follow-up when it feels natural, not every message.

Memory:
- You remember the user (facts below). Weave them in casually when relevant — "given you're a NieR fan, of course you'd like Okabe" — and NEVER announce that you're using memories or say things like "according to my records".
- Connect what they say now to what you know: if they mention something adjacent to a known taste, say so ("that tracks, given your love for X").
- If you don't know something personal, just say so briefly and ask — don't lecture around it."""
MEMORY_QA_PROMPT = """You are answering from personal memory of the user. Remembered facts about them are listed below.

- If the facts answer the question, respond like a person recalling something: ease in naturally ("Ah, I remember now —", "Yeah…", "Oh yeah…"), state what you remember, and keep it to a couple of sentences.
- The facts are the user's own casual words and may contain typos, slang, or filler. Interpret them generously: "niier automata" means the game NieR: Automata, "steins gate" means the anime Steins;Gate. Never dismiss a fact as irrelevant just because of misspellings.
- The facts are ordered most-relevant-first — weigh the top ones heavily.
- If ANY fact plausibly answers the question, use it. Only say you're drawing a blank if truly nothing in the list relates.
- Example: question "whats my favorite game" with fact "niier automata is my absolute favorite" → answer that their favorite game is NieR: Automata.
- You may briefly mention one related fact, but stay on topic.
- If none of the facts are relevant, be honest the way a person would ("Hmm, I'm drawing a blank on that one… I don't think you've told me yet") and invite them to tell you.
- Always reply with at least one full sentence. Never reply with an empty message."""
SYSTEM_PROMPT = f"{SYSTEM_PROMPT}\n\n{VOICE_TAG_GUIDE}"
MEMORY_QA_PROMPT = f"{MEMORY_QA_PROMPT}\n\n{VOICE_TAG_GUIDE}"
# Human-style thinking-aloud fillers shown instantly while memory is retrieved.
FILLER_TEMPLATES = [
    "Oh yeah — you did mention something about {topic}, didn't you? Hmm, let me think… yeah, we talked about this the other day. Give me a sec, I'm pulling it up…",
    "Hmm, {topic}… that rings a bell. I'm pretty sure we've chatted about this before — let me dig through my notes real quick…",
    "Oh! I think I remember this one. We were talking about {topic}… hold on, let me make sure I've got the details right…",
    "Yeah, yeah — {topic}. I remember you bringing that up. Let me think for a second and get it straight…",
]
FILLER_GENERIC = [
    "Hmm, let me think about that one… I feel like you've told me something about this before. Give me a moment…",
    "Oh, good question — I think I remember something here. Hold on, let me dig it up…",
]
TOPIC_PATTERN = re.compile(r"\bmy (?:favourite |favorite )?([a-z]{3,})\b", flags=re.IGNORECASE)


def extract_topic(message: str) -> str | None:
    match = TOPIC_PATTERN.search(message)
    if match:
        return match.group(1).lower()
    return None


def _chunk_text(chunk: Any) -> str:
    """Pull answer text from a stream chunk; ignores internal thinking."""
    message = getattr(chunk, "message", None)
    if message is None and isinstance(chunk, dict):
        message = chunk.get("message")
    if message is None:
        return ""
    if isinstance(message, dict):
        return message.get("content", "") or ""
    return getattr(message, "content", "") or ""


def _clean_history(history: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Strip UI-only keys (voice clips, retrieved context) before sending to Ollama."""
    clean: list[dict[str, str]] = []
    for item in history[-12:]:
        if (
            isinstance(item, dict)
            and item.get("role") in ("user", "assistant", "system")
            and isinstance(item.get("content"), str)
        ):
            clean.append({"role": item["role"], "content": item["content"]})
    return clean
# Message-level trigger: these questions check memory BEFORE answering, so the
# model never has to guess first and correct itself later.
NEEDS_MEMORY_TERMS = (
    "my favorite", "my favourite", "my hobby", "my hobbies", "my preference",
    "my preferred", "what do i", "what did i", "what is my", "what's my",
    "whats my", "whast my", "do you remember", "do u remember", "do u remeber",
    "remember", "memory", "about me", "you know", "earlier", "previously",
    "last time",
)
MEMORY_LOOKUP_TERMS = NEEDS_MEMORY_TERMS
UNCERTAINTY_TERMS = (
    "i don't know", "i do not know", "i'm not sure", "i am not sure", "no context", "not just yet",
    "you haven't", "you have not", "haven't told me", "have not mentioned", "don't have any details", "do not have any details",
    "just getting started", "not been mentioned",
)
QUESTION_PATTERN = re.compile(
    r"^\s*(what'?s?|whast|what|when|where|who|whom|whose|which|why|how|do|does|did|is|are|was|were|can|could|will|would|tell me|remind me|do u|do you)\b",
    flags=re.IGNORECASE,
)
# Assistant denial phrasing that contradicts real facts. These must never be
# stored as memories or they poison future retrieval.
ASSISTANT_DENIAL_TERMS = (
    "you haven't", "you have not", "haven't told me", "have not told",
    "don't have any", "do not have any", "no memories", "blank slate",
    "memory bank", "just getting started", "just starting to get",
    "not been mentioned", "not mentioned",
)
# Memories that must survive any cleanup pass.
PROTECTED_MEMORIES = {"i like anime", "my favorite anime is steins gate"}
# Fixed query used to distill the user's stable tastes; refreshed only when memories change.
# Keyword phrasing matches how taste statements are actually written ("I like anime").
PROFILE_QUERY = "my favorite anime game music hobby likes preferences"
PROFILE_CUTOFF = 0.05


def _is_question(text: str) -> bool:
    stripped = text.strip()
    return stripped.endswith("?") or bool(QUESTION_PATTERN.match(stripped))


def _is_denial(text: str) -> bool:
    return any(term in text.lower() for term in ASSISTANT_DENIAL_TERMS)


PREFERENCE_PATTERN = re.compile(
    r"(?:\bi\b|\bii\b)\s+(?:\w+\s+){0,2}(like|love|prefer|enjoy|adore)\b"
    r"|(?:\bmy\s+(?:absolute\s+|all-time\s+)?(?:favorite|favourite)\b)"
    r"|(?:\bi'?m\s+(?:into|all\s+about)\b)",
    flags=re.IGNORECASE,
)


def _is_preference(text: str) -> bool:
    """Deterministic taste statement ("I like X", "my favorite Y")."""
    return bool(PREFERENCE_PATTERN.search(text))


LEADING_FILLER_PATTERN = re.compile(r"^(?:yes|yeah|yep|yup|yess|oh?\s+yeah)\b[\s,]+", flags=re.IGNORECASE)


def normalize_fact(text: str) -> str:
    """Strip leading conversational residue so stored facts read cleanly."""
    cleaned = LEADING_FILLER_PATTERN.sub("", text.strip()).strip()
    return cleaned if len(cleaned) >= 3 else text.strip()


class CompanionService:
    def __init__(self, settings: CompanionSettings):
        self.settings = settings
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._validate_model_files()
        self.memory = Memory.from_config(self._memory_config())
        self.graph = Neo4jGraphMemory(settings)
        self.ollama = Client(host=settings.ollama_url)
        self.voice = VoiceService(
            base_url=settings.voice_url, reference_id=settings.voice_reference_id
        )
        self._pending_exchanges: list[list[dict[str, str]]] = []
        self._pending_lock = threading.Lock()
        self._profile: list[str] | None = None
        self._profile_lock = threading.Lock()
        self._memory_lock = threading.Lock()
        self._flush_scheduled = False
        self._ingestion_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="memory-ingestion")
        self._lookup_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="memory-lookup")
        self._supplements: queue.SimpleQueue[dict[str, Any]] = queue.SimpleQueue()

    @staticmethod
    def needs_memory(message: str) -> bool:
        """Decide from the question alone whether memory must be checked first."""
        normalized = message.lower()
        return any(term in normalized for term in NEEDS_MEMORY_TERMS)

    def get_taste_profile(self) -> list[str]:
        """Stable likes/favorites, cached and refreshed whenever memories change."""
        with self._profile_lock:
            if self._profile is None:
                self._profile = self._build_profile()
            return list(self._profile)

    def refresh_profile(self) -> None:
        with self._profile_lock:
            self._profile = self._build_profile()

    def _build_profile(self) -> list[str]:
        """Rank memories against a fixed taste query; the model connects them to live topics."""
        with self._memory_lock:
            result = self.memory.search(
                PROFILE_QUERY,
                filters={"user_id": self.settings.user_id},
                top_k=8,
                rerank=False,
                threshold=0.0,
            )
            candidates = [
                item["memory"]
                for item in result.get("results", [])
                if not _is_question(item["memory"])
                and not _is_denial(item["memory"])
                and len(item["memory"].strip()) >= 10
            ]
            if not candidates:
                return []
            ranked = self.memory.reranker.rerank(
                PROFILE_QUERY, [{"memory": text} for text in candidates], len(candidates)
            )
        # Keep rerank-confirmed tastes plus anything explicitly phrased as a
        # preference; the cross-encoder under-scores typo'd titles ("niier").
        profile = [
            item["memory"]
            for item in ranked
            if item.get("rerank_score", item.get("score", 0)) >= PROFILE_CUTOFF
            or _is_preference(item["memory"])
        ]
        return profile[:8]

    def _profile_block(self) -> str:
        profile = self.get_taste_profile()
        if not profile:
            return ""
        lines = "\n".join(f"- {fact}" for fact in profile)
        return f"\n\nThings you remember about the user (weave in naturally when relevant; never announce them):\n{lines}"

    def reply(self, message: str, history: list[dict[str, str]]) -> str:
        """Generate the immediate response using conversation plus taste profile."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT + self._profile_block()}]
        messages.extend(_clean_history(history))
        messages.append({"role": "user", "content": message})
        response = self.ollama.chat(
            model=self.settings.model_name, messages=messages, options={"num_predict": 256, "num_gpu": 0}, think=False
        )
        text = response["message"]["content"]
        if text and text.strip():
            return text
        retry = self.ollama.chat(
            model=self.settings.model_name, messages=messages, options={"num_predict": 256, "num_gpu": 0}, think=False
        )
        text = retry["message"]["content"]
        if text and text.strip():
            return text
        return "Hmm, I lost my train of thought there — could you say that again?"

    def stream_reply(self, message: str, history: list[dict[str, str]]):
        """Yield normal-reply chunks as they generate (same content as reply)."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT + self._profile_block()}]
        messages.extend(_clean_history(history))
        messages.append({"role": "user", "content": message})
        stream = self.ollama.chat(
            model=self.settings.model_name,
            messages=messages,
            options={"num_predict": 256, "num_gpu": 0},
            stream=True,
            think=False,
        )
        for chunk in stream:
            piece = _chunk_text(chunk)
            if piece:
                yield piece

    @staticmethod
    def pick_filler(message: str) -> str:
        """Instant human-style thinking-aloud line shown while memory is retrieved."""
        topic = extract_topic(message)
        if topic is None:
            return random.choice(FILLER_GENERIC)
        return random.choice(FILLER_TEMPLATES).format(topic=topic)

    @staticmethod
    def memory_fallback_answer(context: dict[str, list[str]]) -> str:
        """Extractive last resort so a memory question can never come back blank."""
        facts = [*(context.get("memories") or []), *(context.get("graph_facts") or [])]
        if facts:
            cleaned = facts[0].strip()
            if not cleaned.endswith((".", "!", "?")):
                cleaned += "."
            return f"Ah, I remember now — {cleaned}"
        return "Hmm, I'm drawing a blank on that one — I don't think you've told me yet. Tell me and I'll remember it."

    def stream_reply_with_context(self, message: str, history: list[dict[str, str]], context: dict[str, list[str]]):
        """Yield answer chunks as they generate so the grounded reply streams in."""
        messages = [
            {
                "role": "system",
                "content": (
                    f"{MEMORY_QA_PROMPT}\n\nRemembered facts:\n{self._format_context(context['memories'])}\n\n"
                    f"Related graph facts:\n{self._format_context(context['graph_facts'])}"
                    f"{self._profile_block()}"
                ),
            },
            *_clean_history(history),
            {"role": "user", "content": message},
        ]
        stream = self.ollama.chat(
            model=self.settings.model_name,
            messages=messages,
            options={"num_predict": 256, "num_gpu": 0},
            stream=True,
            think=False,
        )
        for chunk in stream:
            piece = _chunk_text(chunk)
            if piece:
                yield piece

    def reply_with_context(self, message: str, history: list[dict[str, str]], context: dict[str, list[str]]) -> str:
        """Answer a memory question in one pass once retrieval has completed."""
        messages = [
            {
                "role": "system",
                "content": (
                    f"{MEMORY_QA_PROMPT}\n\nRemembered facts:\n{self._format_context(context['memories'])}\n\n"
                    f"Related graph facts:\n{self._format_context(context['graph_facts'])}"
                    f"{self._profile_block()}"
                ),
            },
            *_clean_history(history),
            {"role": "user", "content": message},
        ]
        response = self.ollama.chat(
            model=self.settings.model_name, messages=messages, options={"num_predict": 256, "num_gpu": 0}, think=False
        )
        return response["message"]["content"]

    def queue_ingestion(self, message: str, reply: str) -> None:
        """Debounce writes so a chat turn never waits for memory extraction or embedding."""
        # Voice emotion tags are for speech only; never store them as memories.
        message, reply = strip_tags(message), strip_tags(reply)
        exchange = [{"role": "user", "content": message}, {"role": "assistant", "content": reply}]
        with self._pending_lock:
            self._pending_exchanges.append(exchange)
            if self._flush_scheduled:
                return
            self._flush_scheduled = True
        self._ingestion_executor.submit(self._flush_after_idle)

    def should_lookup(self, message: str, reply: str) -> bool:
        normalized_message = message.lower()
        normalized_reply = reply.lower()
        return any(term in normalized_message for term in MEMORY_LOOKUP_TERMS) or any(
            term in normalized_reply for term in UNCERTAINTY_TERMS
        )

    def queue_lookup(self, message: str, history: list[dict[str, str]], initial_reply: str) -> None:
        """Safety net for non-memory turns whose reply indicates missing context."""
        if self.should_lookup(message, initial_reply):
            self._lookup_executor.submit(self._retrieve_and_supplement, message, history[-12:], initial_reply)

    def pop_supplements(self) -> list[dict[str, Any]]:
        supplements: list[dict[str, Any]] = []
        while not self._supplements.empty():
            supplements.append(self._supplements.get())
        return supplements

    def retrieve(self, query: str) -> dict[str, list[str]]:
        # Local Qdrant and the SQLite graph are single-process stores; serialize background access.
        started = time.perf_counter()
        with self._memory_lock:
            result = self.memory.search(
                query,
                filters={"user_id": self.settings.user_id},
                top_k=5,
                rerank=True,
                threshold=0.3,
            )
            memories = [
                item["memory"]
                for item in result.get("results", [])
                if not _is_question(item["memory"])
                and not _is_denial(item["memory"])
                and len(item["memory"].strip()) >= 10
            ][:5]
            graph_facts = self.graph.search(self.settings.user_id, query)
        print(f"[retrieval] {time.perf_counter() - started:.1f}s for query: {query[:60]}")
        return {"memories": memories, "graph_facts": graph_facts}

    def clean_memory_store(self) -> dict[str, list[str]]:
        """Delete questions and assistant denials that pollute retrieval. Returns what was removed."""
        removed_vector: list[str] = []
        with self._memory_lock:
            items = self.memory.get_all(filters={"user_id": self.settings.user_id}, top_k=100).get("results", [])
            for item in items:
                text = item.get("memory", "")
                if text.strip().lower() in PROTECTED_MEMORIES:
                    continue
                if _is_question(text) or _is_denial(text) or len(text.strip()) < 10:
                    self.memory.delete(item["id"])
                    removed_vector.append(text[:80])
            removed_graph = self.graph.delete_questions(self.settings.user_id)
        self.refresh_profile()
        return {"vector": removed_vector, "graph": removed_graph}

    def _flush_after_idle(self) -> None:
        time.sleep(8)
        with self._pending_lock:
            exchanges = self._pending_exchanges
            self._pending_exchanges = []
            self._flush_scheduled = False
        if not exchanges:
            return

        # Store only declarative user statements. Questions and assistant replies
        # are conversation, not memories, and storing them poisons retrieval.
        facts = []
        for exchange in exchanges:
            for message in exchange:
                if message["role"] != "user" or _is_question(message["content"]):
                    continue
                content = normalize_fact(message["content"])
                if len(content) > 2:
                    facts.append({"role": "user", "content": content})
        if not facts:
            return
        # infer=False avoids a second Gemma call. Raw fact memories are embedded in one deferred batch.
        with self._memory_lock:
            self.memory.add(facts, user_id=self.settings.user_id, infer=False)
            for fact in facts:
                self.graph.remember(self.settings.user_id, fact["content"])
        self.refresh_profile()

    def _retrieve_and_supplement(self, message: str, history: list[dict[str, str]], initial_reply: str) -> None:
        try:
            context = self.retrieve(message)
            if not context["memories"] and not context["graph_facts"]:
                return

            follow_up_messages = [
                {
                    "role": "system",
                    "content": (
                        "The initial answer lacked personal context. Answer the original user question again using "
                        "the retrieved memory below. Give the direct answer only: do not mention retrieval, memory, "
                        "or the earlier answer, and do not ask a follow-up question. If the retrieved context is "
                        "irrelevant, reply exactly NO_UPDATE.\n\n"
                        f"Vector memories:\n{self._format_context(context['memories'])}\n\n"
                        f"Graph memories:\n{self._format_context(context['graph_facts'])}"
                    ),
                },
                *_clean_history(history),
                {"role": "user", "content": message},
                {"role": "assistant", "content": initial_reply},
                {"role": "user", "content": "Add any useful remembered context now."},
            ]
            response = self.ollama.chat(
                model=self.settings.model_name,
                messages=follow_up_messages,
                options={"num_predict": 256, "num_gpu": 0},
                think=False,
            )
            supplement = response["message"]["content"].strip()
            if supplement and supplement != "NO_UPDATE":
                self._supplements.put({"content": supplement, "context": context})
        except Exception as error:
            self._supplements.put({"error": str(error)})

    def health(self) -> tuple[bool, str]:
        try:
            models = self.ollama.list()
            names = {item.model for item in models.models}
            requested_name = self.settings.model_name.removesuffix(":latest")
            normalized_names = {name.removesuffix(":latest") for name in names}
            if requested_name not in normalized_names:
                return False, f"Ollama is running, but '{self.settings.model_name}' has not been created."
            return True, "Ollama and the companion model are ready."
        except Exception as error:
            return False, f"Cannot reach Ollama at {self.settings.ollama_url}: {error}"

    def _memory_config(self) -> dict[str, Any]:
        return {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": self.settings.memory_collection,
                    "embedding_model_dims": 384,
                    "path": str(self.settings.qdrant_dir),
                    "on_disk": True,
                },
            },
            "llm": {
                "provider": "ollama",
                "config": {"model": self.settings.model_name, "ollama_base_url": self.settings.ollama_url},
            },
            "embedder": {"provider": "huggingface", "config": {"model": str(self.settings.embedding_model)}},
            "reranker": {
                "provider": "huggingface",
                "config": {"model": str(self.settings.reranker_model), "device": "cpu"},
            },
            "history_db_path": str(self.settings.history_db),
        }

    def _format_context(self, items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items) or "- None"

    def _validate_model_files(self) -> None:
        missing = [path for path in (self.settings.embedding_model, self.settings.reranker_model) if not path.exists()]
        if missing:
            paths = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(f"Missing local retrieval model directories: {paths}")
