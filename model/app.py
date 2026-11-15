import json
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from fastmcp import Client
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, Field

from health_mcp import mcp
from llm import answer_question, detect_emergency
from attachments import read_attachments, route_ml

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="Bangladesh Health Assistant")


async def call_tool(name: str, args: dict):
    """Call an MCP tool and turn MCP errors into clean HTTP errors."""
    try:
        async with Client(mcp) as client:
            result = await client.call_tool(name, args)
    except ToolError as exc:
        # strip FastMCP's "Error calling tool 'x': " prefix for user-facing text
        raise HTTPException(status_code=400, detail=str(exc).split("': ", 1)[-1])
    except Exception as exc:  # network / Groq / etc.
        raise HTTPException(status_code=502, detail=f"AI service error: {exc}")
    data = result.data
    return data if data is not None else result.structured_content


class ChatIn(BaseModel):
    message: str = Field(..., max_length=4000)
    history: list[dict] = []


class PredictIn(BaseModel):
    disease: str
    inputs: dict


class AgentIn(BaseModel):
    history_and_symptoms: str = Field(..., max_length=4000)
    additional_information: str = Field("", max_length=4000)
    predictions: list[dict] = []


@app.get("/api/diseases")
async def diseases():
    return await call_tool("list_diseases", {})


@app.post("/api/chat")
async def chat(body: ChatIn):
    return await call_tool("ask_health_question", {"question": body.message, "history": body.history})


@app.post("/api/predict")
async def predict(body: PredictIn):
    return await call_tool("predict_disease", {"disease": body.disease, "inputs": body.inputs})


@app.post("/api/agent")
async def agent(body: AgentIn):
    return await call_tool(
        "health_agent",
        {
            "history_and_symptoms": body.history_and_symptoms,
            "additional_information": body.additional_information,
            "predictions": body.predictions,
        },
    )


@app.post("/api/ask")
async def ask(
    message: str = Form(""),
    history: str = Form("[]"),
    files: list[UploadFile] = File(default=[]),
):
    """Claude-style box: text + optional images/documents.
    The LLM answers; then it decides if the ML screening (YES/NO) should run."""
    message = message.strip()
    if not message and not files:
        raise HTTPException(status_code=400, detail="Please type a question or attach a file.")
    if len(message) > 4000:
        raise HTTPException(status_code=400, detail="Question is too long (max 4000 characters).")
    try:
        hist = json.loads(history)
        if not isinstance(hist, list):
            hist = []
    except ValueError:
        hist = []
    message = message or "Please review the attached file and explain it in simple words."

    try:
        items = [(f.filename, f.content_type, await f.read()) for f in files]
        context = await run_in_threadpool(read_attachments, items)
        answer = await run_in_threadpool(answer_question, message, hist, context)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"AI service error: {exc}")

    # ---- ML step: must never break the chat answer ----
    ml = None
    try:
        specs = [s for s in await call_tool("list_diseases", {}) if s["ready"]]
        user_text = "\n".join(str(m.get("content", "")) for m in hist[-6:] if m.get("role") == "user")
        decision = await run_in_threadpool(route_ml, f"{user_text}\n{message}\n{context}", specs)
        spec = next((s for s in specs if s["id"] == decision["disease"]), None)
        if spec:
            inputs = decision["inputs"]
            missing = [f["label"] for f in spec["fields"] if f["key"] not in inputs]
            if missing:
                ml = {"status": "need_more", "disease": spec["id"], "name": spec["name"], "missing": missing}
            else:
                try:
                    ml = {"status": "done", **await call_tool("predict_disease", {"disease": spec["id"], "inputs": inputs})}
                except HTTPException as exc:
                    ml = {"status": "need_more", "disease": spec["id"], "name": spec["name"],
                          "missing": [], "error": exc.detail}
    except Exception:
        ml = None

    return {
        "answer": answer["answer"],
        "emergency_flags": detect_emergency(message + " " + context),
        "ml": ml,
    }



app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")