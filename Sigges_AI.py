# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "ollama>=0.6",
#     "fastapi",
#     "uvicorn",
#     "mcp>=2,<3",
#     "pywebview>=5",
#     "numpy",
# ]
# ///
"""Sigge's AI Assistant: local Qwen3.5 (Ollama) + vision + open-source MCP tools (see MCP_servers.json).

The chat page is ui.html (plain HTML/CSS/JS) served by a small FastAPI server and shown in a native window."""

import asyncio
import base64
import datetime
import json
import math
import os
import re
import shutil
import socket
import sys
import threading
import time
import uuid
from contextlib import AsyncExitStack
from pathlib import Path

import numpy as np
import ollama
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

if sys.stdout is None:   # a windowed exe has no console
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

# Folder with ui.html, mcp_servers.json and memory.jsonl (next to the exe when packaged)
BASE_DIR = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent

# ---------------- Settings (override them in start.bat) ----------------
MODEL = os.environ.get("GAME_AI_MODEL", "qwen3.5:0.8b")
GUARD_MODEL = os.environ.get("GAME_AI_GUARD_MODEL", "llama-guard3:1b")   # content filter for web results
NUM_CTX = int(os.environ.get("GAME_AI_NUM_CTX", "8192"))       # bigger context = more RAM
CPU_ONLY = os.environ.get("GAME_AI_CPU_ONLY", "0") == "1"   # 1 = don't touch the GPU
USE_WINDOW = os.environ.get("GAME_AI_WINDOW", "1") == "1"   # 1 = native app window, 0 = browser
USER_NAME = os.environ.get("GAME_AI_USER_NAME", "").replace('"', "").strip() or "User"   # memory is filed under this name
APP_NAME = os.environ.get("GAME_AI_APP_NAME", "Sigge's AI Assistant")   # window title and page header
EMBED_MODEL = os.environ.get("GAME_AI_EMBED_MODEL", "nomic-embed-text")  # small embedding model, only used when pruning memory
MEMORY_BUDGET = int(os.environ.get("GAME_AI_MEMORY_BUDGET", "3000"))  # max characters of memory put into the prompt
MEMORY_PRUNE_AT = 0.9        # prune when memory is more than 90% of the budget
MEMORY_PRUNE_FRACTION = 0.3  # ...and delete this share of the entries
MCP_CONFIG = BASE_DIR / "MCP_servers.json"
MAX_TOOL_ROUNDS = 4
MAX_HISTORY_MSGS = 20
MAX_TOOL_RESULT = 4000   # long tool output overwhelms a small model's context

SYSTEM_PROMPT = """You are a gaming assistant. The user ({user_name}) is a bilingual(Swedish/English) gamer and developer who prefers short, concise answers.
Rules:
- Be short and direct: conclusion first, then only the necessary steps.
- Only use a tool when the question actually needs it. For greetings, small talk, or things you already know, answer directly without tools.
- Greetings, small talk, thanks, and general questions: answer directly. Do NOT call any tool.
- Only when the user asks about something that can change (guides, builds, patches, events, numbers, news), search before answering. Do not make things up.
- If search snippets are not enough, open the most relevant page with the tool and read it.
- When the user sends a screenshot, look at it carefully before answering.
- For exact math (damage, odds, resources, frame times) or to verify code/algorithms, write Python and run it with run_python_code. Always print the result.
- Answer in English/Swedish based on the current context.
Memory:
- You have long-term memory. When the user tells you something lasting about themselves (name, games they play,
  progress, characters/builds, preferences, habits), call add_observations with entityName "{user_name}" and one short English sentence per fact.
- If the user says a fact is wrong or asks you to forget it, call delete_observations.
- Do not store temporary or one-off questions.
What you currently remember about the user:
{memory}
Today is {date}."""
GREETING_NAME = "" if USER_NAME in ("User", "Friend") else USER_NAME             # shown in the greeting


# ---- Adult-content guard start ----
# Only web results are screened (search results and opened pages), by Llama Guard 3 running in Ollama.
#   1. One guard call on the whole result. If it is clean, nothing else happens (the common, fast case).
#   2. If the guard flags sexual content (S12) or child sexual exploitation (S4), each numbered
#      search result is checked on its own:
#        - at least half of the results are flagged -> the user is clearly after adult content:
#          the model never sees them and the fixed reply below is returned
#        - fewer than half are flagged -> only the flagged results are removed
#          (e.g. NSFW mods showing up in a normal game search)
#   3. An opened web page that is flagged is blocked as a whole.
#   4. If the guard itself fails, the results are withheld (fail closed).
ADULT_REPLY = "wait bro I'm not sure if it's ok for you to ask this :o -- Karl"
GUARD_CODES = {"s12", "s4","s3","s11"}      # Llama Guard categories: S12 sexual content, S4 child sexual exploitation
GUARD_BLOCK_RATIO = 0.5
PAGE_BLOCKED = "(This page was blocked by the content filter.)"
RESULTS_BLOCKED = "(These search results were blocked by the content filter.)"
FILTER_ERROR = "(Web results are unavailable because the content filter failed.)"
_NUMBERED = re.compile(r"\s*\d+\.\s")          # start of a numbered search result
_SPLIT_RESULTS = re.compile(r"\n(?=\d+\.\s)")  # split before each numbered search result


def guard_is_adult(text: str) -> bool:
    resp = client.chat(
        model=GUARD_MODEL,
        messages=[{"role": "user", "content": text[:MAX_TOOL_RESULT]}],
        options={"temperature": 0, "num_predict": 16, "num_ctx": 6144},
        keep_alive="5m",
    )
    out = (resp.message.content or "").strip().lower()
    if out.startswith("safe"):
        return False
    if out.startswith("unsafe"):
        codes = set(re.findall(r"s\d+", out))
        print(f"[Guard] unsafe {sorted(codes)}", flush=True)
        return not codes or bool(codes & GUARD_CODES)
    raise ValueError(f"unexpected guard output: {out!r}")


def screen_web_result(tool_name: str, result: str):
    """Return (text_for_the_model, fixed_reply_or_None)."""
    try:
        if not guard_is_adult(result):
            return result, None
        if "fetch" in tool_name:  # an opened page
            return PAGE_BLOCKED, None
        blocks = _SPLIT_RESULTS.split(result)
        is_result = [bool(_NUMBERED.match(b)) for b in blocks]
        is_bad = [numbered and guard_is_adult(b) for b, numbered in zip(blocks, is_result)]
        n_results, n_bad = sum(is_result), sum(is_bad)
        if n_results == 0:
            return RESULTS_BLOCKED, None
        if n_bad / n_results >= GUARD_BLOCK_RATIO:
            return "", ADULT_REPLY
        return "\n".join(b for b, bad in zip(blocks, is_bad) if not bad), None
    except Exception as e:  # noqa: BLE001
        print(f"[Guard] failed: {e!r}", flush=True)
        return FILTER_ERROR, None
# ---- Adult-content guard end ----


# ---------------- MCP tool manager ----------------
def _resolve_command(cmd: str) -> str:
    """Find uvx/npx etc. A freshly installed uv may not be on PATH yet."""
    found = shutil.which(cmd)
    if found:
        return found
    candidates = []
    if os.environ.get("UV"):  # set by `uv run`
        candidates.append(Path(os.environ["UV"]).parent)
    candidates.append(Path.home() / ".local" / "bin")
    if sys.platform == "win32":
        candidates.append(Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "nodejs")
    for d in candidates:
        for ext in ((".exe", ".cmd") if sys.platform == "win32" else ("",)):
            if (d / (cmd + ext)).exists():
                return str(d / (cmd + ext))
    return cmd


class MCPHub:
    """Starts every MCP server in MCP_servers.json on a background thread and routes tool calls."""

    def __init__(self, config_path: Path):
        self.tools: list[dict] = []          # tool definitions sent to Ollama
        self.route: dict[str, ClientSession] = {}
        self.tool_server: dict[str, str] = {}     # tool name -> server name
        self.failed: list[str] = []          # servers that could not start
        self._ready = threading.Event()
        self._loop = asyncio.new_event_loop()
        cfg = {}
        if config_path.exists():
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
        else:
            print(f"[MCP] {config_path.name} not found, running without tools.")
        threading.Thread(target=self._loop.run_forever, daemon=True).start()
        self._stop = None
        self._main_future = asyncio.run_coroutine_threadsafe(self._main(cfg), self._loop)
        self._main_future.add_done_callback(self._on_main_done)
        # No waiting here: the window opens at once, and the tools appear as they finish starting
        # (the first launch downloads the tool packages, which can take a few minutes).

    async def _main(self, cfg):
        self._stop = asyncio.Event()
        async with AsyncExitStack() as stack:
            for name, spec in cfg.get("mcpServers", {}).items():
                if spec.get("disabled"):
                    continue
                print(f"[MCP] Starting {name} ...", flush=True)
                try:
                    command = _resolve_command(spec["command"])
                    app_dir = str(BASE_DIR)
                    env = {**os.environ,
                           **{k: str(v).replace("${APP_DIR}", app_dir)
                              for k, v in spec.get("env", {}).items()}}
                    # node / deno installed by winget may not be on PATH yet; add common locations
                    extra = [str(Path(command).parent)]
                    if sys.platform == "win32":
                        extra += [
                            str(Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "nodejs"),
                            str(Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links"),
                            str(Path.home() / ".deno" / "bin"),
                        ]
                    env["PATH"] = os.pathsep.join([*extra, env.get("PATH", "")])
                    params = StdioServerParameters(
                        command=command, args=spec.get("args", []), env=env,
                    )
                    read, write = await stack.enter_async_context(stdio_client(params))
                    session = await stack.enter_async_context(ClientSession(read, write))
                    await asyncio.wait_for(session.initialize(), timeout=300)
                    listed = await session.list_tools()
                except Exception as e:  # noqa: BLE001
                    print(f"[MCP] WARNING: {name} failed to start, skipped: {e!r}", flush=True)
                    self.failed.append(name)
                    continue
                allow = spec.get("tools")  # only these are shown to the model; the app can still call the rest
                for t in listed.tools:
                    if t.name in self.route:
                        print(f"[MCP] Duplicate tool name, skipping {name}.{t.name}")
                        continue
                    self.route[t.name] = session
                    self.tool_server[t.name] = name
                    if allow is not None and t.name not in allow:
                        continue
                    self.tools.append({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description or "",
                            "parameters": t.input_schema or {"type": "object", "properties": {}},
                        },
                    })
                print(f"[MCP] OK {name}: {', '.join(t.name for t in listed.tools)}", flush=True)
            self._ready.set()
            await self._stop.wait()

    def _on_main_done(self, fut):
        """Without this, a crash in _main is silent and the tools never become ready."""
        if not fut.cancelled() and fut.exception() is not None:
            print(f"[MCP] ERROR: tool manager crashed: {fut.exception()!r}", flush=True)
            self._ready.set()  # don't leave _startup_memory waiting forever

    async def _call(self, name: str, args: dict) -> str:
        session = self.route.get(name)
        if session is None:
            return f"Unknown tool: {name}"
        resp = await asyncio.wait_for(session.call_tool(name, args), timeout=60)
        parts = [getattr(c, "text", None) or f"[{c.type} content]" for c in resp.content]
        text = "\n".join(parts) or "(the tool returned nothing)"
        return ("Tool error: " if getattr(resp, "is_error", False) else "") + text

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    def close(self):
        """Stop all MCP tool servers (called when the app window is closed)."""
        if self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
            try:
                self._main_future.result(timeout=8)
            except Exception:  # noqa: BLE001
                pass

    def call(self, name: str, args: dict) -> str:
        fut = asyncio.run_coroutine_threadsafe(self._call(name, args), self._loop)
        return fut.result(timeout=90)


hub = MCPHub(MCP_CONFIG)
client = ollama.Client()


# ---------------- Memory ----------------
def _init_memory():
    if "create_entities" in hub.route:
        try:  # ignored if it already exists
            hub.call("create_entities", {"entities": [
                {"name": USER_NAME, "entityType": "person", "observations": []}]})
        except Exception as e:  # noqa: BLE001
            print(f"[Memory] Init failed: {e!r}")


def load_memory() -> str:
    """Read all memory before each turn and put it in the system prompt, so a small model doesn't have to look it up."""
    if "read_graph" not in hub.route:
        return "(memory is not available)"
    try:
        graph = json.loads(hub.call("read_graph", {}))
    except Exception:  # noqa: BLE001
        return "(could not read memory)"
    lines = []
    for ent in graph.get("entities", []):
        for obs in ent.get("observations", []):
            prefix = "" if ent.get("name") == USER_NAME else f"{ent.get('name')}: "
            lines.append(f"- {prefix}{obs}")
    for rel in graph.get("relations", []):
        lines.append(f"- {rel.get('from')} {rel.get('relationType')} {rel.get('to')}")
    text = "\n".join(lines) or "(nothing remembered yet)"
    return text[-int(MEMORY_BUDGET * 0.95):]  # safety net: stay a little under the budget


# ---- Memory pruning ----
# Memory is a list of short sentences. When it uses more than MEMORY_PRUNE_AT of MEMORY_BUDGET,
# the most redundant MEMORY_PRUNE_FRACTION of the sentences are deleted: each sentence is embedded,
# the pair with the highest cosine similarity is found, and the OLDER sentence of that pair is deleted;
# repeat until enough are gone. If the embedding model is unavailable, the oldest sentences go instead.
_prune_lock = threading.Lock()


def _memory_items() -> list[tuple[str, str]]:
    graph = json.loads(hub.call("read_graph", {}))
    return [(e.get("name", ""), o) for e in graph.get("entities", []) for o in e.get("observations", [])]


def _item_size(entity: str, obs: str) -> int:
    return len(obs) + (0 if entity == USER_NAME else len(entity) + 2) + 3   # "- " prefix and newline


def pick_redundant(texts: list[str], k: int) -> list[int]:
    """Indices of the k most redundant entries (higher cosine similarity goes first, older goes first)."""
    resp = client.embed(model=EMBED_MODEL, input=texts, keep_alive=0)
    emb = np.array(resp.embeddings, dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    sim = emb @ emb.T
    np.fill_diagonal(sim, -1.0)
    dropped = []
    for _ in range(k):
        i, j = np.unravel_index(np.argmax(sim), sim.shape)
        if sim[i, j] <= -1.0:
            break
        old = int(min(i, j))          # keep the newer sentence: it is more likely to be correct
        dropped.append(old)
        sim[old, :] = -1.0
        sim[:, old] = -1.0
    return dropped


def prune_memory():
    if "read_graph" not in hub.route or "delete_observations" not in hub.route:
        return
    with _prune_lock:
        try:
            items = _memory_items()
            size = sum(_item_size(e, o) for e, o in items)
            if size <= MEMORY_BUDGET * MEMORY_PRUNE_AT:
                return
            k = max(1, math.ceil(len(items) * MEMORY_PRUNE_FRACTION))
            try:
                drop = pick_redundant([o for _, o in items], k)
                how = "most similar"
            except Exception as e:  # noqa: BLE001
                print(f"[Memory] Embedding failed ({e!r}); deleting the oldest entries instead.", flush=True)
                drop = list(range(k))
                how = "oldest"
            by_entity: dict[str, list[str]] = {}
            for idx in drop:
                by_entity.setdefault(items[idx][0], []).append(items[idx][1])
            hub.call("delete_observations", {"deletions": [
                {"entityName": ent, "observations": obs} for ent, obs in by_entity.items()]})
            print(f"[Memory] {size} chars was near the {MEMORY_BUDGET} limit: deleted "
                  f"{len(drop)} of {len(items)} entries ({how}).", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[Memory] Pruning failed: {e!r}", flush=True)



def _startup_memory():
    hub._ready.wait()
    _init_memory()
    prune_memory()


threading.Thread(target=_startup_memory, daemon=True).start()


# ---------------- Chat logic ----------------
DEFAULT_PROMPT = "What is going on in this image? What should I do?"
MAX_IMAGES = 4
MAX_IMAGE_BYTES = 8_000_000


def _decode_images(images) -> list[bytes]:
    out = []
    for item in (images or [])[:MAX_IMAGES]:
        try:
            raw = base64.b64decode(str(item).split(",", 1)[-1], validate=False)
        except Exception:  # noqa: BLE001
            continue
        if 0 < len(raw) <= MAX_IMAGE_BYTES:
            out.append(raw)
    return out


def _short(value, n=90) -> str:
    text = str(value)
    return text if len(text) <= n else text[: n - 1] + "..."


def chat_stream(text: str, images, history):
    """Run one turn and yield events for the page (one JSON object each)."""
    text = (text or "").strip() or DEFAULT_PROMPT
    new_user = {"role": "user", "content": text}
    pics = _decode_images(images)
    if pics:
        new_user["images"] = pics
    # Old screenshots and tool results are not resent: saves RAM and context
    past = [{"role": m["role"], "content": m["content"]}
            for m in (history or [])[-MAX_HISTORY_MSGS:]
            if isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)]
    prune_memory()
    system = {"role": "system",
              "content": SYSTEM_PROMPT.format(date=datetime.date.today().isoformat(),
                                              memory=load_memory(), user_name=USER_NAME)}
    msgs = [system, *past, new_user]

    options = {"num_ctx": NUM_CTX}
    if CPU_ONLY:
        options["num_gpu"] = 0

    answer = ""
    try:
        for rnd in range(MAX_TOOL_ROUNDS + 1):
            use_tools = bool(hub.tools) and rnd < MAX_TOOL_ROUNDS
            content, tool_calls = "", []
            stream = client.chat(
                model=MODEL,
                messages=msgs,
                tools=hub.tools if use_tools else None,
                stream=True,
                options=options,
                keep_alive="30m",
            )
            for chunk in stream:
                piece = chunk.message.content or ""
                if piece:
                    content += piece
                    yield {"type": "token", "text": piece}
                if chunk.message.tool_calls:
                    tool_calls.extend(chunk.message.tool_calls)

            msgs.append(ollama.Message(role="assistant", content=content,
                                       tool_calls=tool_calls or None))
            if not tool_calls:
                answer = content
                break

            for tc in tool_calls:
                name = tc.function.name
                args = dict(tc.function.arguments or {})
                tid = uuid.uuid4().hex[:8]
                first = next(iter(args.values()), "")
                start = {"type": "tool_start", "id": tid, "name": name, "arg": _short(first)}
                if name == "run_python_code":
                    start["code"] = str(args.get("python_code") or args.get("code") or first)[:3000]
                yield start
                try:
                    result = hub.call(name, args)
                except Exception as e:  # noqa: BLE001
                    result = f"Tool failed: {e!r}"
                if hub.tool_server.get(name) == "search":  # web results: screen before the model sees them
                    result, fixed = screen_web_result(name, result)
                    if fixed:
                        yield {"type": "tool_result", "id": tid, "text": "(results hidden by the content filter)"}
                        yield {"type": "token", "text": fixed}
                        yield {"type": "done", "answer": fixed}
                        return
                result = result[:MAX_TOOL_RESULT]
                yield {"type": "tool_result", "id": tid,
                       "text": result[:1500] + ("..." if len(result) > 1500 else "")}
                msgs.append({"role": "tool", "content": result, "tool_name": name})
    except ollama.ResponseError as e:
        err = str(e.error)
        if "not found" in err.lower():
            err = f"The model {MODEL} is not installed yet. Close the app and run start.bat again to download it."
        yield {"type": "error", "text": err}
        return
    except Exception as e:  # noqa: BLE001
        print(f"[Chat] {e!r}", flush=True)
        yield {"type": "error", "text": "Can't reach the AI model. Make sure Ollama is running, then try again."}
        return
    yield {"type": "done", "answer": answer}


def _ndjson(events):
    for ev in events:
        yield json.dumps(ev, ensure_ascii=False) + "\n"


# ---------------- Web server ----------------
api = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
NO_CACHE = {"Cache-Control": "no-store"}


@api.get("/")
def page():
    ui = BASE_DIR / "ui.html"
    if not ui.exists():
        return PlainTextResponse("ui.html is missing. Put it in the same folder as app.py.", status_code=500)
    return FileResponse(ui, headers=NO_CACHE)


@api.get("/icon.jpg")
def icon():
    path = BASE_DIR / "icon.jpg"
    if not path.exists():
        return PlainTextResponse("", status_code=404)
    return FileResponse(path)


@api.get("/api/info")
def api_info():
    return JSONResponse({
        "app": APP_NAME, "model": MODEL, "name": GREETING_NAME, "ready": hub.ready, "failed": hub.failed,
        "tools": [t["function"]["name"] for t in hub.tools],
    }, headers=NO_CACHE)


@api.post("/api/chat")
async def api_chat(request: Request):
    body = await request.json()
    events = chat_stream(body.get("text"), body.get("images"), body.get("history"))
    return StreamingResponse(_ndjson(events), media_type="application/x-ndjson",
                             headers={**NO_CACHE, "X-Accel-Buffering": "no"})


@api.get("/api/memory")
def api_memory():
    if "read_graph" not in hub.route:
        return JSONResponse({"available": False, "items": []}, headers=NO_CACHE)
    try:
        items = [{"entity": e, "text": o} for e, o in _memory_items()]
    except Exception:  # noqa: BLE001
        return JSONResponse({"available": False, "items": []}, headers=NO_CACHE)
    return JSONResponse({"available": True, "items": items}, headers=NO_CACHE)


@api.post("/api/memory/delete")
async def api_memory_delete(request: Request):
    body = await request.json()
    entity, text = str(body.get("entity") or ""), str(body.get("text") or "")
    if not entity or not text or "delete_observations" not in hub.route:
        return JSONResponse({"ok": False}, status_code=400)
    hub.call("delete_observations", {"deletions": [{"entityName": entity, "observations": [text]}]})
    return JSONResponse({"ok": True})


@api.post("/api/open")
async def api_open(request: Request):
    """Open a link from an answer in the default browser (the app window itself never navigates away)."""
    url = str((await request.json()).get("url") or "")
    if url.startswith(("http://", "https://")):
        import webbrowser
        webbrowser.open(url)
    return JSONResponse({"ok": True})


def pick_port(preferred: int = 7860) -> int:
    for port in (preferred, 0):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    return preferred


def start_server():
    port = pick_port()
    server = uvicorn.Server(uvicorn.Config(api, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.05)
    return server, f"http://127.0.0.1:{port}/"


if __name__ == "__main__":
    server, url = start_server()
    print(f"[App] Running at {url}", flush=True)
    if USE_WINDOW:
        try:  # app mode: a native window around the local page
            import webview
            webview.create_window(APP_NAME, url, width=1100, height=860, min_size=(520, 480))
            webview.start()          # blocks until the window is closed
            hub.close()
            os._exit(0)
        except Exception as e:  # noqa: BLE001
            print(f"[Window] Native window unavailable ({e!r}); using the browser instead.", flush=True)
    import webbrowser
    webbrowser.open(url)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        hub.close()
