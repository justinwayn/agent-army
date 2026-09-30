import os
import json
import time
import urllib.parse
from pathlib import Path
import requests
import redis
import chromadb
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Any
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage

# --- 1. Memory Layers ---

# Redis: Fast, volatile memory for active tasks
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# ChromaDB: Persistent, vector memory for learning
chroma_client = chromadb.PersistentClient(path="./chroma_hive_memory")
antipattern_collection = chroma_client.get_or_create_collection(name="antipatterns")
success_collection = chroma_client.get_or_create_collection(name="success_patterns")

# --- 2. State Definition (LangGraph) ---

class HiveState(TypedDict):
    # The current active task
    current_step: str

    # What genre/niche this cycle should focus on (asked for at start-up,
    # blank means "general side-hustle/KDP signal, no filter")
    niche: str

    # Data flowing between agents
    raw_data: List[dict]
    blueprint: dict
    product_id: str

    # Human Approval Gate
    pending_approval: bool
    approval_status: str # "approved", "rejected", None

    # Memory injection
    anti_patterns: List[str]

# --- 3. The Agents ---

# Subreddits picked to line up with Justin's actual niches (KDP, POD, side-hustle
# content ideas) rather than anything generic. Add/remove freely.
MICHELANGELO_SUBREDDITS = [
    "sidehustle",
    "KindleDirectPublishing",
    "printondemand",
    "Etsy",
    "smallbusiness",
]

# Reddit's public .json endpoints work without login for light, occasional use
# like this. They are NOT the registered OAuth API — if this ever runs on a
# schedule (not just manual triggers) or needs to scale past a handful of
# calls a day, register a free Reddit API app instead and swap this out.
MICHELANGELO_HEADERS = {
    "User-Agent": "hive-michelangelo/0.2 (personal side-project, by u/justinwayn)"
}


def michelangelo_node(state: HiveState) -> dict:
    """
    Michelangelo (renamed from "Scavenger" — same job, better name) pulls
    real, live signal from Reddit: post titles as topic candidates,
    upvotes/comment count as a rough popularity proxy, and post body text
    (when there is any) as a pain-point signal. Pushes it to Redis for the
    Alchemist to work from.

    If a niche/genre was given at start-up, this runs a real Reddit search
    for it FIRST, then adds the baseline subreddit sweep — so picking a
    niche actually changes what gets scraped, not just what gets picked
    afterward.

    Upvotes are a popularity proxy, not real keyword-search volume — a
    first real data source, not the final one.
    """
    niche = (state.get("niche") or "").strip()
    print(f"🎨 Michelangelo: Scanning for inspiration{f' — niche: {niche}' if niche else ''}...")

    raw_data: List[dict] = []

    if niche:
        q = urllib.parse.quote(niche)
        url = f"https://www.reddit.com/search.json?q={q}&sort=top&t=month&limit=15"
        try:
            resp = requests.get(url, headers=MICHELANGELO_HEADERS, timeout=10)
            resp.raise_for_status()
            posts = resp.json().get("data", {}).get("children", [])
            hits = 0
            for post in posts:
                d = post.get("data", {})
                if d.get("stickied"):
                    continue
                raw_data.append({
                    "topic": (d.get("title") or "").strip(),
                    "source": f"reddit-search:{niche}",
                    "volume": d.get("score", 0),
                    "comments": d.get("num_comments", 0),
                    "pain_point": (d.get("selftext") or "")[:280].strip()
                        or "(title-only post, no body text)",
                    "url": f"https://reddit.com{d.get('permalink', '')}",
                })
                hits += 1
            print(f"  -> niche search '{niche}': {hits} posts")
        except Exception as e:
            print(f"  -> niche search '{niche}' failed ({e}), falling back to baseline subreddits")
        time.sleep(2)

    for sub in MICHELANGELO_SUBREDDITS:
        url = f"https://www.reddit.com/r/{sub}/top.json?limit=10&t=week"
        try:
            resp = requests.get(url, headers=MICHELANGELO_HEADERS, timeout=10)
            resp.raise_for_status()
            posts = resp.json().get("data", {}).get("children", [])
        except Exception as e:
            print(f"  -> r/{sub}: fetch failed ({e}), skipping")
            continue

        hits = 0
        for post in posts:
            d = post.get("data", {})
            if d.get("stickied"):
                continue
            raw_data.append({
                "topic": (d.get("title") or "").strip(),
                "source": f"r/{sub}",
                "volume": d.get("score", 0),            # upvotes = popularity proxy
                "comments": d.get("num_comments", 0),
                "pain_point": (d.get("selftext") or "")[:280].strip()
                    or "(title-only post, no body text)",
                "url": f"https://reddit.com{d.get('permalink', '')}",
            })
            hits += 1
        print(f"  -> r/{sub}: {hits} posts")

        time.sleep(2)  # basic politeness between calls

    if not raw_data:
        print("  -> WARNING: zero posts collected this run — check network / Reddit availability")

    # Strongest signal first
    raw_data.sort(key=lambda x: x["volume"], reverse=True)

    # Push to Redis
    redis_client.lpush("raw_data_queue", json.dumps(raw_data))
    return {"current_step": "scraping_complete", "raw_data": raw_data}

# Local Ollama config — override with env vars if your model/port differs.
# Check what you actually have pulled with `ollama list` in your terminal.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5")  # confirmed via `ollama run qwen3.5` in your terminal


def call_ollama(prompt: str, system: str = "") -> str:
    """
    Calls a locally-running Ollama model. No API key, no per-call cost —
    just needs `ollama serve` (or the Ollama app) running on this machine.
    """
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "system": system,
                "stream": False,
                "format": "json",
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Can't reach Ollama at {OLLAMA_URL} — is `ollama serve` / the Ollama app running?"
        )
    except Exception as e:
        raise RuntimeError(f"Ollama call failed: {e}")


def alchemist_node(state: HiveState) -> dict:
    """
    The Alchemist reads Michelangelo's real Reddit data, asks a locally
    running Ollama model to turn the strongest signal into a product
    blueprint, and checks it against Anti-Patterns already learned.
    """
    print("🔮 Alchemist: Synthesizing ideas...")

    # 1. Fetch Anti-Patterns from ChromaDB
    #    (bug fix: anti-pattern rules live in `metadatas`, not `documents` —
    #    the old code read `item['metadata']` off a list of plain document
    #    strings, which would have thrown the first time this collection
    #    actually had anything in it)
    anti_patterns_result = antipattern_collection.get()
    metadatas = anti_patterns_result.get('metadatas') or []
    anti_patterns = [m.get('rule', '') for m in metadatas if m and m.get('rule')]

    print(f"  -> Loaded {len(anti_patterns)} Active Anti-Patterns")

    # 2. Generate Blueprint — real call to local Ollama, grounded in what
    #    Michelangelo actually found this run. Falls back to a clearly-
    #    labeled placeholder (never a silent fake) if Ollama isn't reachable
    #    or the model doesn't return usable JSON, so the graph doesn't crash.
    niche = (state.get("niche") or "").strip()
    raw_data = state.get("raw_data") or []
    top_signals = raw_data[:5]
    blueprint = {"title": None, "keywords": [], "format": None}
    raw_response = ""

    if not top_signals:
        print("  -> No raw_data from Michelangelo this run — skipping generation")
        blueprint["note"] = "no input signal — Michelangelo returned nothing this run"
    else:
        niche_clause = (
            f"The user specifically wants ideas in this niche/genre: '{niche}'. "
            f"Prefer signals that genuinely fit it. If none of today's signals "
            f"actually fit, say so honestly in the 'reasoning' field instead of "
            f"forcing an unrelated topic to fit. "
            if niche else ""
        )
        system_prompt = (
            "You are the Alchemist agent in a content/product idea pipeline. "
            "Given real trending topics scraped from Reddit, pick the single "
            "strongest opportunity and turn it into a concrete product blueprint. "
            + niche_clause +
            "Respect the anti-pattern rules — these are past mistakes the system "
            "already learned to avoid. Respond with ONLY a JSON object with keys: "
            "title (string), keywords (array of strings), format (string, e.g. "
            "'KDP Paperback', 'Etsy digital download', 'POD shirt design'), and "
            "reasoning (one sentence on why this signal, in plain terms)."
        )
        user_prompt = (
            f"Anti-patterns to avoid: {json.dumps(anti_patterns) if anti_patterns else '(none yet)'}\n\n"
            f"Top signals this run:\n{json.dumps(top_signals, indent=2)}"
        )
        try:
            raw_response = call_ollama(user_prompt, system=system_prompt)
            blueprint = json.loads(raw_response)
        except RuntimeError as e:
            print(f"  -> {e}")
            blueprint["note"] = f"Ollama unavailable this run: {e}"
        except json.JSONDecodeError:
            print("  -> Ollama didn't return valid JSON, keeping raw text for review")
            blueprint["note"] = "model returned non-JSON response"
            blueprint["raw_response"] = raw_response

    # 3. Store in Long-Term Memory — only when we actually generated something
    #    real, so failed/empty runs don't pollute the success-pattern library
    if blueprint.get("title"):
        success_collection.add(
            documents=[json.dumps(blueprint)],
            ids=[str(len(success_collection.get()['ids']) + 1)]
        )

    return {"current_step": "blueprint_generated", "blueprint": blueprint, "anti_patterns": anti_patterns}

# Where real build output lands — override with an env var if you want it
# somewhere other than a `builds/` folder next to this script.
BUILD_OUTPUT_DIR = os.environ.get("BUILD_OUTPUT_DIR", "./builds")


def generate_page_image(prompt: str, out_path: Path, width: int = 1024, height: int = 1024) -> bool:
    """
    Generates one real image via Pollinations.ai's free, no-key image API
    and saves it to out_path. Returns True/False, never raises — one bad
    page shouldn't kill the whole build.

    Honest caveat: this is a free public service, not a model fine-tuned for
    coloring books. It's what's actually usable start-to-finish without a
    paid API key (OpenAI/Stability) or a local GPU diffusion setup — neither
    of which is confirmed on this machine. Quality/consistency won't match
    your manual DALL-E/ChatGPT workflow. If quality isn't good enough once
    you see real output, swap this one function for a paid API — everything
    else (the manifest, the per-page prompts) stays the same.
    """
    full_prompt = (
        f"{prompt}, coloring book page, black and white line art, no shading, "
        f"no color, no grayscale, clean bold outlines, white background"
    )
    url = f"https://image.pollinations.ai/prompt/{urllib.parse.quote(full_prompt)}?width={width}&height={height}&nologo=true"
    try:
        resp = requests.get(url, timeout=90)
        resp.raise_for_status()
        if not resp.content or len(resp.content) < 1000:
            print("    -> image response too small/empty, treating as failure")
            return False
        with open(out_path, "wb") as f:
            f.write(resp.content)
        return True
    except Exception as e:
        print(f"    -> image generation failed: {e}")
        return False


def builder_node(state: HiveState) -> dict:
    """
    The Builder turns the Alchemist's blueprint into a real, KDP-ready book
    package, start to finish:
      1. Asks Ollama for a page-by-page manifest (subject + art prompt per
         page) and KDP metadata — real call, written to a real JSON file.
      2. Actually generates a real PNG for every page via a free image API
         (see generate_page_image) and saves them to disk, updating the
         manifest with each page's image path/status.
    Real Ollama call, real files, real images — nothing mocked. Per-page
    image failures are logged and skipped, not silently faked — a failed
    page keeps its art_prompt in the manifest so it can still be pasted into
    DALL-E/ChatGPT by hand if the free service dropped it.
    """
    print("🛠️ Builder: Generating book package...")

    blueprint = state.get("blueprint") or {}
    if not blueprint.get("title"):
        print("  -> No usable blueprint this run (Alchemist had nothing to build from) — skipping")
        return {"current_step": "build_skipped", "product_id": ""}

    system_prompt = (
        "You are the Builder agent. Given a coloring-book blueprint, produce "
        "a page-by-page manifest for a real KDP coloring book of 20-30 pages. "
        "Respond with ONLY a JSON object with this shape: "
        "{\"page_count\": <int>, \"pages\": [{\"page_number\": <int>, "
        "\"subject\": \"<short subject>\", \"art_prompt\": \"<a detailed "
        "black-and-white line-art prompt: bold clean outlines, no shading, "
        "no color, coloring-book style, ready to paste into an image "
        "generator>\"}], \"kdp_metadata\": {\"title\": \"...\", "
        "\"subtitle\": \"...\", \"description\": \"...\", \"categories\": "
        "[\"...\"], \"trim_size\": \"8.5x11\"}}"
    )
    user_prompt = f"Blueprint: {json.dumps(blueprint)}"

    try:
        raw_response = call_ollama(user_prompt, system=system_prompt)
        package = json.loads(raw_response)
    except RuntimeError as e:
        print(f"  -> {e}")
        return {"current_step": "build_failed", "product_id": ""}
    except json.JSONDecodeError:
        print("  -> Builder's model didn't return valid JSON — build failed, not faked")
        return {"current_step": "build_failed", "product_id": ""}

    # Real deliverable #1: the manifest file.
    product_id = f"prod_{int(time.time())}"
    out_dir = Path(BUILD_OUTPUT_DIR) / product_id
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "book_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(package, f, indent=2)

    pages = package.get("pages", [])
    page_count = package.get("page_count", len(pages))
    print(f"  -> Wrote {manifest_path} — {page_count} pages planned")

    # Real deliverable #2: the actual page images.
    images_dir = out_dir / "images"
    images_dir.mkdir(exist_ok=True)
    succeeded = 0
    for i, page in enumerate(pages):
        try:
            page_num = int(page.get("page_number", i + 1))
        except (TypeError, ValueError):
            page_num = i + 1
        art_prompt = page.get("art_prompt", "")
        if not art_prompt:
            page["image_status"] = "skipped — no art_prompt from Builder"
            continue

        img_path = images_dir / f"page_{page_num:02d}.png"
        print(f"  -> Generating image {i + 1}/{len(pages)} (page {page_num})...")
        ok = generate_page_image(art_prompt, img_path)
        if ok:
            page["image_path"] = str(img_path)
            page["image_status"] = "generated"
            succeeded += 1
        else:
            page["image_path"] = None
            page["image_status"] = "failed — art_prompt above still usable, paste into DALL-E/ChatGPT by hand"
        time.sleep(3)  # politeness between image requests

    # Re-write the manifest now that image paths/status are attached
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(package, f, indent=2)

    print(f"  -> Images: {succeeded}/{len(pages)} generated successfully")
    if succeeded < len(pages):
        print(f"  -> {len(pages) - succeeded} page(s) failed — see {manifest_path.name} "
              f"for which ones, each still has a usable art_prompt")

    return {
        "current_step": "building_complete",
        "product_id": product_id,
        "blueprint": {
            **blueprint,
            "manifest_path": str(manifest_path),
            "images_dir": str(images_dir),
            "page_count": page_count,
            "images_generated": succeeded,
        },
    }

def governor_node(state: HiveState) -> dict:
    """
    The Governor decides whether there's anything worth putting in front of
    a human. No point pausing for approval on a cycle that produced nothing
    (Michelangelo found no signal, Alchemist had no blueprint, Builder
    failed) — that just gets skipped straight to the end, and says why.
    """
    product_id = state.get("product_id") or ""
    blueprint = state.get("blueprint") or {}

    if not product_id or not blueprint.get("manifest_path"):
        print("👮 Governor: Nothing built this cycle — nothing to approve, ending here.")
        return {"current_step": "nothing_to_approve", "pending_approval": False}

    print("👮 Governor: Build complete — routing to human approval before this counts as done.")
    return {"current_step": "awaiting_human_approval", "pending_approval": True}

def human_approval_node(state: HiveState) -> dict:
    """
    A REAL approval gate: pauses this script and waits for you to actually
    type y/n at the terminal. Nothing gets called "done" without you looking
    at it.

    This is NOT the Telegram/Discord "approve from your phone" version those
    other chats described — that needs a bot token and a running webhook
    server, neither of which exists, and I'm not going to fake having built
    it. This is the honest, immediately-real version: it genuinely blocks
    and genuinely waits for you, right here. Worth upgrading to a real
    webhook later if you want to approve from your phone instead of being
    at the machine — that's a real, separate next step, not a one-liner.
    """
    if not state.get("pending_approval"):
        return {"current_step": "done", "approval_status": "n/a"}

    blueprint = state.get("blueprint") or {}
    print("\n" + "=" * 60)
    print("👤 HUMAN APPROVAL REQUIRED")
    print(f"  Title:     {blueprint.get('title')}")
    print(f"  Pages:     {blueprint.get('page_count', '?')} planned, "
          f"{blueprint.get('images_generated', 0)} images generated")
    print(f"  Manifest:  {blueprint.get('manifest_path', '(none)')}")
    print(f"  Images:    {blueprint.get('images_dir', '(none)')}")
    print("=" * 60)

    answer = input("Approve this build? [y/N]: ").strip().lower()

    if answer == "y":
        print("✅ Approved.")
        return {"current_step": "approved", "approval_status": "approved"}
    print("❌ Rejected — nothing further happens to this build automatically "
          "(no silent auto-retry, that's exactly the kind of runaway loop to avoid).")
    return {"current_step": "rejected", "approval_status": "rejected"}

# --- 4. The State Machine ---

def route_after_michelangelo(state: HiveState) -> str:
    return "alchemist"

def route_after_alchemist(state: HiveState) -> str:
    return "builder"

def route_after_builder(state: HiveState) -> str:
    return "governor"

def route_after_governor(state: HiveState) -> str:
    if state.get("pending_approval"):
        return "human_approval"
    return END  # nothing built, nothing to approve

def route_after_human(state: HiveState) -> str:
    # Approved or rejected, this cycle's work ends here either way. No
    # auto-retry loop back into Builder — a rejection should be a deliberate
    # decision by you on the next run (maybe a different niche), not the
    # system silently burning more Ollama/image-gen calls on its own.
    return END

# Build the graph
graph_builder = StateGraph(HiveState)

# Add nodes
graph_builder.add_node("michelangelo", michelangelo_node)
graph_builder.add_node("alchemist", alchemist_node)
graph_builder.add_node("builder", builder_node)
graph_builder.add_node("governor", governor_node)
graph_builder.add_node("human_approval", human_approval_node)

# Define edges
graph_builder.set_entry_point("michelangelo")

graph_builder.add_conditional_edges(
    "michelangelo",
    route_after_michelangelo,
    {"alchemist": "alchemist"}
)

graph_builder.add_conditional_edges(
    "alchemist",
    route_after_alchemist,
    {"builder": "builder"}
)

graph_builder.add_conditional_edges(
    "builder",
    route_after_builder,
    {"governor": "governor"}
)

graph_builder.add_conditional_edges(
    "governor",
    route_after_governor,
    {"human_approval": "human_approval", END: END}
)

graph_builder.add_conditional_edges(
    "human_approval",
    route_after_human,
    {END: END}
)

# Compile
hive_app = graph_builder.compile()

# --- 5. Run the Hive ---

if __name__ == "__main__":
    # Initialize memory
    print("🚀 Initializing Hive Memory...")
    redis_client.ping()
    print("✅ Redis Connected")

    niche = input(
        "🎨 What genre or niche should this cycle focus on? "
        "(leave blank for general side-hustle/KDP signal): "
    ).strip()

    # Run one cycle
    initial_state = {
        "current_step": "idle",
        "niche": niche,
        "raw_data": [],
        "blueprint": {},
        "product_id": "",
        "pending_approval": False,
        "approval_status": None,
        "anti_patterns": []
    }

    print("🏁 Starting Hive Cycle...")
    try:
        final_state = hive_app.invoke(initial_state)
        print("✅ Cycle Complete!")
        print(f"Final State: {final_state}")
    except Exception as e:
        print(f"❌ Error: {e}")
