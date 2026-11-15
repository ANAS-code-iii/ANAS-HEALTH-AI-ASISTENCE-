"""
attachments.py - NEW helper (llm.py is untouched except 2 tiny edits).

1. read_attachments(): turns uploaded images / PDF / DOCX / TXT into plain text
   that the health LLM can use.
2. route_ml(): asks the LLM "is a YES/NO ML screening useful here, and which
   disease + which values did the user actually state?"

Install:  pip install python-multipart pypdf python-docx
"""
import base64
import io
import json
import os
import re

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from llm import clean, get_llm

# Groq model that can read images (Qwen text model cannot).
VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

MAX_FILES = 4
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_CHARS_PER_FILE = 3000
MAX_CONTEXT_CHARS = 8000


def _vision_llm() -> ChatGroq:
    key = os.getenv("GROQ_API_KEY") or os.getenv("Groq_api")
    if not key:
        raise RuntimeError("Groq API key missing. Add GROQ_API_KEY=your_key to your .env file.")
    return ChatGroq(model=VISION_MODEL, api_key=key, temperature=0.1, max_tokens=1500)


def _read_image(data: bytes, mime: str) -> str:
    b64 = base64.b64encode(data).decode()
    msg = HumanMessage(content=[
        {"type": "text", "text": (
            "This image was uploaded to a health assistant. Transcribe all visible text, "
            "test names, values and units exactly. If it is a photo of a body part or a "
            "symptom, describe only what is visible. Do NOT diagnose. Plain text only."
        )},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
    ])
    return clean(_vision_llm().invoke([msg]).content)


def _read_pdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError("PDF support missing. Run: pip install pypdf")
    reader = PdfReader(io.BytesIO(data))
    text = "\n".join((p.extract_text() or "") for p in reader.pages[:15]).strip()
    if not text:
        raise ValueError("This PDF has no readable text (scanned?). Please upload it as a photo instead.")
    return text


def _read_docx(data: bytes) -> str:
    try:
        import docx
    except ImportError:
        raise RuntimeError("Word support missing. Run: pip install python-docx")
    d = docx.Document(io.BytesIO(data))
    parts = [p.text for p in d.paragraphs if p.text.strip()]
    for t in d.tables:
        for row in t.rows:
            parts.append(" | ".join(c.text.strip() for c in row.cells))
    return "\n".join(parts).strip()


def read_attachments(items: list[tuple[str, str, bytes]]) -> str:
    """items = [(filename, content_type, bytes), ...] -> one text block."""
    if len(items) > MAX_FILES:
        raise ValueError(f"Please attach at most {MAX_FILES} files.")
    blocks = []
    for name, mime, data in items:
        name = name or "file"
        mime = (mime or "").lower()
        low = name.lower()
        if len(data) > MAX_FILE_BYTES:
            raise ValueError(f"{name} is too large (max 8 MB).")
        if mime.startswith("image/"):
            text = _read_image(data, mime)
        elif mime == "application/pdf" or low.endswith(".pdf"):
            text = _read_pdf(data)
        elif low.endswith(".docx"):
            text = _read_docx(data)
        elif mime.startswith("text/") or low.endswith((".txt", ".csv", ".md")):
            text = data.decode("utf-8", errors="ignore")
        else:
            raise ValueError(f"{name}: unsupported file type. Use image, PDF, DOCX or TXT.")
        text = text.strip()[:MAX_CHARS_PER_FILE]
        if text:
            blocks.append(f"[Attached file: {name}]\n{text}")
    return "\n\n".join(blocks)[:MAX_CONTEXT_CHARS]


# --------------------------------------------------------------------------- #
# ML ROUTER  (the "YES / NO" decision)
# --------------------------------------------------------------------------- #
ROUTER_SYSTEM = """You decide whether a machine-learning disease screening should be run.
Return ONLY one JSON object, no other text:
{"disease": "<id or none>", "inputs": {"<field key>": <value>}}

Rules:
- Choose a disease id only if the person's text is clearly about that disease's risk/screening,
  or contains test values that belong to it. Otherwise use "none".
- In "inputs" include ONLY values the person explicitly stated (or that are written in the
  attached report). NEVER guess or invent a value. Leave unknown fields out.
- For fields with options, the value must be EXACTLY one of the listed options.
- For numeric fields use a plain number in the unit shown."""


def _spec_text(specs: list[dict]) -> str:
    lines = []
    for s in specs:
        lines.append(f'id="{s["id"]}" ({s["name"]})')
        for f in s["fields"]:
            if f["type"] == "select":
                lines.append(f'  {f["key"]}: one of {f["options"]}')
            else:
                lines.append(f'  {f["key"]}: number {f["min"]}-{f["max"]} {f.get("unit", "")}')
    return "\n".join(lines)


def route_ml(text: str, specs: list[dict]) -> dict:
    """Returns {"disease": id|None, "inputs": {...}}. Never raises."""
    if not specs or not text.strip():
        return {"disease": None, "inputs": {}}
    try:
        prompt = f"Available screening models:\n{_spec_text(specs)}\n\nPerson's text:\n{text[:6000]}"
        raw = clean(get_llm().invoke([("system", ROUTER_SYSTEM), ("user", prompt)]).content)
        m = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(m.group(0)) if m else {}
    except Exception:
        return {"disease": None, "inputs": {}}

    spec = next((s for s in specs if s["id"] == data.get("disease")), None)
    if spec is None:
        return {"disease": None, "inputs": {}}

    clean_inputs = {}
    raw_inputs = data.get("inputs") if isinstance(data.get("inputs"), dict) else {}
    for f in spec["fields"]:
        val = raw_inputs.get(f["key"])
        if val is None or val == "":
            continue
        if f["type"] == "select":
            if val in f["options"]:
                clean_inputs[f["key"]] = val
        else:
            try:
                clean_inputs[f["key"]] = float(val)
            except (TypeError, ValueError):
                pass
    return {"disease": spec["id"], "inputs": clean_inputs}