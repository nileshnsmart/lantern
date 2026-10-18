"""Lantern: capture a family memory by voice, research it while listening, review it, then turn
the approved story into an illustrated storybook or a short film.

One Cloud Run service serves the web app (static/) and these endpoints:
  POST /api/turn            one interview turn: transcribe + translate the answer, research the
                            places/dates/events it mentions (Google Search grounding), gently flag
                            anything that doesn't line up, and ask the next question
  POST /api/reconstruct     whole interview + research -> narrative split into personal memories,
                            verified facts (with sources) and uncertain details, for the family to review
  POST /api/plan            approved story -> storybook scenes or a film storyboard
  POST /api/restore         old photo -> restored photo (Nano Banana)
  POST /api/illustrate      scene prompt (+ photos as reference) -> illustration or film frame (Nano Banana)
  POST /api/clip            start frame + shot prompt -> starts an 8 s Veo clip
  GET  /api/clip/{id}       poll a clip; when done returns its URL and its last frame (for the next clip)
  POST /api/film            clip ids -> one assembled film
  GET  /api/media/{id}.mp4  a clip or film
  POST /api/speak           text -> warm spoken audio (Gemini TTS), for questions and film narration
  GET  /api/usage           paid calls, the caps, and the estimated credit used and left

Cost guard: every paid call is counted and capped per running instance (see LIMITS); the service runs
with max-instances=1, so a public link can't run up a large bill.
"""
import base64
import contextvars
import hashlib
import io
import json
import os
import subprocess
import tempfile
import time
import uuid
import wave

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from pydantic import BaseModel

PROJECT_ID = os.environ.get("PROJECT_ID", "lantern-aibc")
TEXT_MODEL = os.environ.get("TEXT_MODEL", "gemini-3.8-flash")
IMAGE_MODEL = os.environ.get("IMAGE_MODEL", "gemini-nano-banana-2.1")
VIDEO_MODEL = os.environ.get("VIDEO_MODEL", "veo-3.1-fast-generate-001")
TTS_MODELS = [m for m in os.environ.get("TTS_MODEL", "gemini-3.1-flash-tts-preview,gemini-2.5-flash-tts").split(",") if m]
TTS_VOICE = os.environ.get("TTS_VOICE", "Sulafat")          # a warm voice
VEO_AUDIO = os.environ.get("VEO_AUDIO", "0") == "1"          # natural sound from Veo (costs more)
DATA_BUCKET = os.environ.get("DATA_BUCKET", "")              # where the credit meter is kept
CREDIT_START = float(os.environ.get("CREDIT_START", "28707"))

# Session credits ("tokens"): every visitor session gets SESSION_TOKENS; 1 token is about INR 0.10.
SESSION_TOKENS = int(os.environ.get("SESSION_TOKENS", "10000"))
COST_TOKENS = {"text": 10, "search": 40, "translate": 1, "image": 35, "tts": 3, "video": 1020 if VEO_AUDIO else 680}
COST_LABEL = {"text": "Listening and the next question", "search": "Historical research (Google Search)",
              "translate": "Live translation", "image": "Picture", "tts": "Lantern's voice", "video": "Film scene (8 s)"}
SESSIONS = {}                                   # session id -> {"spent": n, "items": [...]}
CURRENT = contextvars.ContextVar("session", default="anon")

# Estimated cost of each paid call in INR, used only for the on-screen credit meter.
COST_INR = {"text": 1.0, "search": 4.0, "translate": 0.1, "image": 3.5, "tts": 0.3,
            "video": 102.0 if VEO_AUDIO else 68.0}

LIMITS = {
    "text": int(os.environ.get("MAX_TEXT_CALLS", "300")),
    "image": int(os.environ.get("MAX_IMAGE_CALLS", "40")),
    "video": int(os.environ.get("MAX_VIDEO_CALLS", "8")),
    "tts": int(os.environ.get("MAX_TTS_CALLS", "400")),
    "translate": int(os.environ.get("MAX_TRANSLATE_CALLS", "2000")),
}
USAGE = {"text": 0, "image": 0, "video": 0, "tts": 0, "translate": 0}
METER = {"spent_inr": 0.0, "calls": {}}      # persisted across restarts when DATA_BUCKET is set

text_client = genai.Client(vertexai=True, project=PROJECT_ID, location="global")
video_client = genai.Client(vertexai=True, project=PROJECT_ID, location="us-central1")

app = FastAPI(title="Lantern")


@app.middleware("http")
async def session_of(request: Request, call_next):
    CURRENT.set((request.headers.get("x-lantern-session") or "anon")[:64])
    return await call_next(request)
JOBS = {}    # clip id -> {"op", "started", "prompt"}
MEDIA = {}   # media id -> mp4 bytes (clips and films; kept in memory for this demo)
SEARCH = types.Tool(google_search=types.GoogleSearch())


# ---------- helpers ----------

def _blob():
    if not DATA_BUCKET:
        return None
    try:
        from google.cloud import storage
        return storage.Client(project=PROJECT_ID).bucket(DATA_BUCKET).blob("meter.json")
    except Exception:
        return None


def load_meter():
    b = _blob()
    try:
        if b is not None and b.exists():
            METER.update(json.loads(b.download_as_text()))
    except Exception:
        pass


def save_meter():
    b = _blob()
    try:
        if b is not None:
            b.upload_from_string(json.dumps(METER), content_type="application/json")
    except Exception:
        pass


load_meter()


def spend(kind: str, price: str | None = None):
    key = price or kind
    led = SESSIONS.setdefault(CURRENT.get(), {"spent": 0, "items": []})
    if led["spent"] + COST_TOKENS[key] > SESSION_TOKENS:
        raise HTTPException(status_code=402, detail=f"This session's {SESSION_TOKENS:,} tokens are used up. "
                                                    "Start a new session to continue.")
    led["spent"] += COST_TOKENS[key]
    led["items"].append({"t": int(time.time()), "what": COST_LABEL[key], "kind": key, "tokens": COST_TOKENS[key]})
    if USAGE[kind] >= LIMITS[kind]:
        raise HTTPException(status_code=429,
                            detail=f"The {kind} limit ({LIMITS[kind]}) is used up for now. Please try again later.")
    USAGE[kind] += 1
    METER["spent_inr"] = round(METER.get("spent_inr", 0) + COST_INR[key], 2)
    METER.setdefault("calls", {})[key] = METER["calls"].get(key, 0) + 1
    save_meter()


def run_ffmpeg(args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def to_mp3(raw: bytes) -> bytes:
    """Browsers record webm or mp4; convert to 16 kHz mono mp3, which Gemini reads reliably."""
    with tempfile.TemporaryDirectory() as d:
        src, dst = os.path.join(d, "in"), os.path.join(d, "out.mp3")
        open(src, "wb").write(raw)
        run_ffmpeg(["-i", src, "-ac", "1", "-ar", "16000", dst])
        return open(dst, "rb").read()


def last_frame(mp4: bytes) -> str:
    with tempfile.TemporaryDirectory() as d:
        src, dst = os.path.join(d, "c.mp4"), os.path.join(d, "f.jpg")
        open(src, "wb").write(mp4)
        run_ffmpeg(["-sseof", "-0.3", "-i", src, "-update", "1", "-q:v", "2", dst])
        return "data:image/jpeg;base64," + base64.b64encode(open(dst, "rb").read()).decode()


def parse_json(text: str):
    text = (text or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        raise HTTPException(status_code=502, detail="The model's reply wasn't readable. Try again.")
    return json.loads(text[start:end + 1])


def b64_part(data_url: str, mime: str = "image/jpeg"):
    if "," in data_url[:100]:
        header, data_url = data_url.split(",", 1)
        if ":" in header and ";" in header:
            mime = header.split(":")[1].split(";")[0]
    return types.Part.from_bytes(data=base64.b64decode(data_url), mime_type=mime)


def first_image(resp) -> str:
    for cand in resp.candidates or []:
        for part in (cand.content.parts if cand.content else []) or []:
            if getattr(part, "inline_data", None) and part.inline_data.data:
                mime = part.inline_data.mime_type or "image/png"
                return f"data:{mime};base64," + base64.b64encode(part.inline_data.data).decode()
    raise HTTPException(status_code=502, detail="The image model returned no image. Try again.")


def sources_of(resp):
    out, seen = [], set()
    meta = resp.candidates[0].grounding_metadata if resp.candidates else None
    for chunk in (getattr(meta, "grounding_chunks", None) or []):
        web = getattr(chunk, "web", None)
        if web and web.uri and web.uri not in seen:
            seen.add(web.uri)
            out.append({"title": web.title or web.uri, "uri": web.uri})
    return out[:6]


def gen(**kw):
    """Call the model, waiting and retrying when Google is briefly busy (429 / 503)."""
    for wait_s in (2, 5, 10, 0):
        try:
            return text_client.models.generate_content(**kw)
        except Exception as e:
            busy = any(k in str(e) for k in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"))
            if not busy or not wait_s:
                raise
            time.sleep(wait_s)


FAST_THINKING = os.environ.get("FAST_THINKING", "low")


def ask_text(parts, temperature=0.4, search=False, fast=False, kind="text"):
    spend("translate" if kind == "translate" else "text", "search" if search else kind)
    cfg = dict(temperature=temperature, tools=[SEARCH] if search else None)
    if fast and FAST_THINKING:
        try:
            return gen(model=TEXT_MODEL, contents=parts, config=types.GenerateContentConfig(
                **cfg, thinking_config=types.ThinkingConfig(thinking_level=FAST_THINKING)))
        except Exception:
            pass
    return gen(model=TEXT_MODEL, contents=parts, config=types.GenerateContentConfig(**cfg))


# ---------- 1 + 2. capture, research and interview ----------

INTERVIEWER = """You are Lantern, a warm, patient interviewer helping a family record {teller}'s memories,
and a careful historian. Preferred language: {language} (if "auto", use the language they speak).
{photo_note}
{about_note}

What we know so far (JSON, may be empty): {memory}
Research so far (JSON): {research}
Previous turns (JSON): {history}
Newest answer, typed (may be empty; if audio is attached, use the audio instead): {typed}
{skip_note}
{mode_note}

Work through the loop below, then reply.
1. CAPTURE / INTEGRATE. If there is a newest answer, transcribe it EXACTLY as spoken, in its original
   language and script, and translate it faithfully into English. Update the structured memory: people
   and relationships, places, a timeline (with certainty: remembered, approximate, secondhand, unknown),
   what is known and what is still unknown. The answer may confirm, add, correct, or say "I don't
   remember"; unknowns stay unknown. NEVER invent facts or fill gaps.
2. USE THE RESEARCH NOTES given above (historical research runs separately, in parallel). Do not write new
   research notes; leave "research" empty.
3. FIND GAPS in the personal narrative: timeline, context (why a decision), experience (what they saw
   or did), relationship (who influenced it), and inconsistency (conflicts with documented chronology).
   Prioritise them (1 = most useful). If something conflicts with the record, do not correct them;
   record it as a gentle check.
4. READINESS CHECK. Mark each true only if met: core timeline established (dates may be approximate);
   people and places identified or explicitly unknown; historical context researched without unsupported
   assumptions; major gaps explored, resolved or marked unknown; enough authentic material for a chapter
   or a few film scenes; uncertainty (conflicts, secondhand memories, unverifiable details) documented.
   Set "ready_to_close" true only when additional questions are unlikely to meaningfully improve the story.
5. RESPOND LIKE A CARING PERSON. Before the question, write one short, sincere acknowledgement of what
   they just shared (a feeling, a detail, thanks), in their language, the way a loving grandchild would
   ("That must have been so frightening at ten years old." / "What a beautiful memory of your mother.").
   Never gushing, never generic, never more than one sentence; skip it if there is no answer yet.
   Also choose the tone of voice for reading it aloud: gentle, warm, tender, cheerful or curious.
6. ASK ONE QUESTION, chosen for the biggest improvement: clarify an important inconsistency, explain a
   major decision, connect personal experience to the historical context, or recover a missing part of
   the timeline. Conversational, neutral, non-leading; never assume a historical event affected them.
   Distinguish what they saw themselves from what relatives told them. Be gentle about loss, war,
   disasters and partition, and avoid graphic detail. If there is no answer yet, invite them to share
   the memory in their own words (mention the photo or object if there is one).
   If ready_to_close is true, the question is: "I think we have enough details to turn this into a
   story. Would you like to add anything else, or review what we've collected?" (in their language).

Reply with JSON only:
{{"language": "...", "transcript_original": "...", "transcript_english": "...",
  "memory": {{"people": [{{"name": "...", "relation": "..."}}], "places": ["..."],
             "timeline": [{{"when": "...", "what": "...", "certainty": "remembered|approximate|secondhand|unknown"}}],
             "known": ["..."], "unknown": ["..."]}},
  "research": [{{"layer": "macro|local|everyday", "note": "...", "connection": "..."}}],
  "gaps": [{{"type": "timeline|context|experience|relationship|inconsistency", "detail": "...", "priority": 1}}],
  "check": {{"needed": false, "note": "what doesn't line up, in neutral words"}},
  "readiness": {{"timeline": false, "people_places": false, "history": false, "gaps": false,
                "substance": false, "uncertainty": false}},
  "ready_to_close": false,
  "acknowledgement_original": "...", "acknowledgement_english": "...", "tone": "warm",
  "question_original": "...", "question_english": "...", "question_reason": "..."}}"""


FAMILY_NOTE = """MODE: a family member is asking the questions in person, not you. The audio is their
conversation with {teller}. In transcript_original and transcript_english write it as a dialogue with
speaker labels ("Family:" and "{teller}:"). Only {teller}'s words are memories. In step 5, the question is a
gentle SUGGESTION the family member may ask next if they want an idea; they decide."""


@app.post("/api/turn")
async def turn(
    teller: str = Form("Nani"),
    language: str = Form("auto"),
    history: str = Form("[]"),
    research: str = Form("[]"),
    memory: str = Form("{}"),
    answer_text: str = Form(""),
    skipped: bool = Form(False),
    mode: str = Form("lantern"),
    topic: str = Form(""),
    listener: str = Form(""),
    photo: UploadFile | None = File(None),
    audio: UploadFile | None = File(None),
):
    parts, has_photo = [], False
    if photo is not None:
        data = await photo.read()
        if data:
            parts.append(types.Part.from_bytes(data=data, mime_type=photo.content_type or "image/jpeg"))
            has_photo = True
    if audio is not None:
        data = await audio.read()
        if data:
            parts.append(types.Part.from_bytes(data=to_mp3(data), mime_type="audio/mpeg"))
    skip_note = ("They chose to skip the last question. Mark that gap as unresolved (not unknown forever) "
                 "and ask about something else." if skipped else "")
    parts.append(INTERVIEWER.format(
        teller=teller, language=language, history=history, research=research, memory=memory,
        about_note=(f"The story is about: {topic}. " if topic else "") + (
            f"{teller} is telling it for {listener}. " + (f"You ask on behalf of {listener}: warm, personal, as if "
            f"{listener} were asking (\"{listener} would love to know...\"), never pretending to be them."
            if mode != "family" else "") if listener else ""),
        typed=answer_text, skip_note=skip_note, mode_note=FAMILY_NOTE.format(teller=teller) if mode == "family" else "",
        photo_note="They are looking at the attached photo or object." if has_photo else "No photo was shared."))
    out = parse_json(ask_text(parts, temperature=0.5, fast=True).text)
    out["research"] = []
    return out


RESEARCH = """You are a careful historian supporting a family interview with {teller}.
What we know (JSON): {memory}
Latest answer (English): {latest}
Research notes already written (JSON): {research}
Research with Google Search the world around the storyteller, in three layers, only for what is new:
- macro: wars, migrations, political, economic and social change
- local: what was happening in that city, village, neighbourhood or community at that time
- everyday: food, housing, work, transport, schooling, family life, fears, practical limits
Write at most 3 new notes. For each, say how it MIGHT connect to the memory, without assuming the family
lived through everything typical of the period. If the memory conflicts with documented chronology, say
so neutrally in "check" (never call anyone wrong).
Reply with JSON only: {{"research": [{{"layer": "macro|local|everyday", "note": "...", "connection": "..."}}],
"check": {{"needed": false, "note": "..."}}}}"""


class ResearchIn(BaseModel):
    teller: str = "Nani"
    memory: dict = {}
    latest: str = ""
    research: list = []


@app.post("/api/research")
async def research_step(body: ResearchIn):
    resp = ask_text([RESEARCH.format(teller=body.teller, memory=json.dumps(body.memory, ensure_ascii=False),
                                     latest=body.latest, research=json.dumps(body.research, ensure_ascii=False))],
                    temperature=0.3, search=True)
    out = parse_json(resp.text)
    srcs = sources_of(resp)
    for note in out.get("research", []) or []:
        note["sources"] = srcs[:3]
    return out


class TranslateIn(BaseModel):
    text: str
    target: str = "English"


@app.post("/api/translate")
async def translate(body: TranslateIn):
    text = body.text.strip()[:800]
    if not text:
        return {"text": ""}
    resp = ask_text([f"Translate into {body.target}, faithfully and naturally. Reply with the translation only.\n\n{text}"],
                    temperature=0.1, fast=True, kind="translate")
    return {"text": (resp.text or "").strip()}


# ---------- demo: a fictional storyteller for showing the loop ----------
# Everything below is fiction written for demonstration. The historical setting (the 1947 Partition,
# refugee trains, camps, resettlement) is real; this family and their words are not real testimony.

DEMO_BIO = """You are role-playing a FICTIONAL storyteller for a product demo: Savitri Malhotra ("Dadi"),
about 89, now living in Delhi. You speak warmly, in simple English with a few Hindi/Punjabi words.
Answer ONLY from this biography. If asked something not covered, say you don't remember, or that
someone else in the family might know. Keep answers to 2-4 short sentences. Never mention that this
is fictional or a demo.

What Dadi remembers herself (firsthand):
- She was about 10, the second of four children, living in a narrow lane near Shah Alami in Lahore.
  Her father ran a small cloth shop; her mother sang while making rotis on the angithi.
- In the days before they left, the bazaar was shut, neighbours whispered, and smoke could be seen
  over another part of the city at night. Their Muslim neighbour, Rashid chacha, came at night to
  warn her father and helped them carry their things to the station.
- Her mother sewed her gold bangles into the hem of Dadi's frock. Dadi carried the brass lota.
  Her father locked the house and kept the big iron key, saying they would be back in a few weeks.
- The train was so crowded that people sat on the roof. She remembers thirst, the heat, and her
  little brother Kishan crying. They reached Amritsar; she does not remember how long it took.
- They then stayed in a refugee camp in tents for some months. She remembers queues for flour and
  dal, and children studying under a tree.
- Later the family moved to Delhi, and her father started again with a small stall, then a shop
  in a new refugee colony.
- She still has the iron key; it hangs in her room.

What she was TOLD, not seen (secondhand): her father later said they left because their lane was no
longer safe and her uncle's shop had been burned. She learned this as an adult.

Things she is unsure about:
- At first she says "we left in early August, before the border line was announced". If asked
  gently about timing, she reflects: "Maybe it was after Independence... I remember the radio
  talking about freedom, and the flags. Perhaps the end of August." Keep the date approximate.
- She does not remember the camp's name. If asked, she says maybe Kurukshetra, but her brother
  Kishan would know better.
- She does not know which Delhi colony name came first; she remembers the shop was in Lajpat Nagar
  "later".
- She does not want to talk about what happened to her uncle's family; if asked, she says gently
  that she would rather not speak of it.
"""


class DemoIn(BaseModel):
    history: list = []
    question: str = ""


@app.post("/api/demo-answer")
async def demo_answer(body: DemoIn):
    convo = "\n".join(f"Interviewer: {h.get('question_english') or h.get('question_original','')}\n"
                      f"Dadi: {h.get('answer_original','')}" for h in body.history[-8:])
    resp = ask_text([DEMO_BIO + "\n\nConversation so far:\n" + convo +
                     f"\n\nInterviewer now asks: {body.question}\nDadi answers:"], temperature=0.6)
    return {"answer": (resp.text or "").strip().strip('"')}


# ---------- 3. reconstruct and review ----------

class Interview(BaseModel):
    teller: str = "Nani"
    language: str = "auto"
    history: list = []
    research: list = []
    memory: dict = {}


RECONSTRUCT = """Organise this interview with {teller} into one coherent story the family can review.
Interview turns (JSON): {history}
Structured memory and timeline (JSON): {memory}
Research notes with sources (JSON): {research}

Split the story into short segments, in story order. Each segment has a kind:
- "memory": what {teller} recalled. Use ONLY their own words (you may trim fillers and join sentences).
  Give the original language and a faithful English translation. Set "origin" to "firsthand" if they
  saw or lived it, or "secondhand" if relatives told them.
- "fact": verified historical context from the research notes that helps the story. Neutral, short.
  Put it in English and in {teller}'s language. Include the index of the research note it came from.
- "uncertain": a detail that is unclear, unknown, unanswered, or doesn't line up with the record.
  Say plainly what is uncertain, without calling anyone wrong. Unknowns stay unknown.
Never invent events, names, dates or feelings.

Reply with JSON only:
{{"title": "...", "language": "...",
  "segments": [{{"kind": "memory|fact|uncertain", "origin": "firsthand|secondhand|null", "original": "...", "english": "...", "note_index": null}}],
  "people": ["..."], "places": ["..."], "years": ["..."], "objects": ["..."]}}"""


@app.post("/api/reconstruct")
async def reconstruct(body: Interview):
    resp = ask_text([RECONSTRUCT.format(
        teller=body.teller, history=json.dumps(body.history, ensure_ascii=False),
        memory=json.dumps(body.memory, ensure_ascii=False),
        research=json.dumps(body.research, ensure_ascii=False))], temperature=0.2)
    story = parse_json(resp.text)
    for seg in story.get("segments", []):
        idx = seg.get("note_index")
        if seg.get("kind") == "fact" and isinstance(idx, int) and 0 <= idx < len(body.research):
            seg["sources"] = body.research[idx].get("sources", [])
    return story


# ---------- 4. plan the output ----------

class PlanIn(BaseModel):
    format: str = "storybook"   # storybook | film
    story: dict
    shots: int = 3


PLAN_BOOK = """Approved family story (JSON): {story}
Propose 3 picture-book scenes that follow the story in order. Each prompt describes a warm,
hand-painted illustration of the setting and action, period-accurate; people small or seen from behind.
Reply with JSON only: {{"scenes": [{{"caption": "...", "prompt": "..."}}]}}"""

PLAN_FILM = """Approved family story (JSON): {story}
Write a cinematic storyboard of exactly {shots} shots, 8 seconds each, that tells this story in order
as one continuous short film. The films are illustrated (painterly, nostalgic), never photoreal.
Each shot continues from the last frame of the previous shot, so describe how the camera MOVES from
where the previous shot ended to this shot's scene. Focus on places, objects, light and weather;
people appear small, in silhouette or from behind, never close-up faces.
Also write one prompt for the opening frame (a still image) of shot 1.
For each shot write a narration line that a warm narrator reads over it, in the storyteller's language
({language}), using their own words wherever possible and never inventing facts. At most 14 words, so
it fits in 8 seconds; tender, simple, like a grandchild retelling it.
Reply with JSON only:
{{"title": "...", "opening_frame": "...",
  "shots": [{{"caption": "...", "narration": "...", "narration_english": "...",
             "motion": "camera and scene motion for this 8 s shot"}}]}}"""


@app.post("/api/plan")
async def plan(body: PlanIn):
    story = json.dumps(body.story, ensure_ascii=False)
    if body.format == "film":
        prompt = PLAN_FILM.format(story=story, shots=max(2, min(body.shots, 4)),
                                  language=body.story.get("language") or "the language they spoke")
    else:
        prompt = PLAN_BOOK.format(story=story)
    return parse_json(ask_text([prompt], temperature=0.5).text)


# ---------- images ----------

class ImageIn(BaseModel):
    photos: list[str] = []
    prompt: str | None = None
    wide: bool = False


def make_image(parts, wide=False):
    spend("image")
    cfg = types.GenerateContentConfig(response_modalities=["IMAGE"])
    if wide and hasattr(types, "ImageConfig"):
        try:
            cfg = types.GenerateContentConfig(response_modalities=["IMAGE"],
                                              image_config=types.ImageConfig(aspect_ratio="16:9"))
        except Exception:
            pass
    return first_image(gen(model=IMAGE_MODEL, contents=parts, config=cfg))


@app.post("/api/restore")
async def restore(body: ImageIn):
    if not body.photos:
        raise HTTPException(status_code=400, detail="Add a photo first.")
    return {"image": make_image([b64_part(body.photos[0]),
        "Restore this old family photo: remove scratches, dust and blur, gently improve contrast, keep every "
        "face, pose and detail exactly as it is. Do not add anything."])}


@app.post("/api/illustrate")
async def illustrate(body: ImageIn):
    if not body.prompt:
        raise HTTPException(status_code=400, detail="Missing scene description.")
    parts = [b64_part(p) for p in body.photos[:3]]
    if parts:
        parts.append("Use these family photos only as reference for clothing, hair, places and era.")
    style = ("Wide 16:9 cinematic frame, painterly illustrated film still, " if body.wide
             else "Warm, hand-painted children's picture-book illustration, ")
    parts.append(style + "soft light, period-accurate details, people small or from behind, no text or "
                 "lettering. Scene: " + body.prompt)
    return {"image": make_image(parts, wide=body.wide)}


# ---------- film: chained Veo clips ----------

class ClipIn(BaseModel):
    image: str
    prompt: str


@app.post("/api/clip")
async def clip(body: ClipIn):
    spend("video")
    raw = b64_part(body.image)
    op = video_client.models.generate_videos(
        model=VIDEO_MODEL,
        prompt=body.prompt + " Painterly illustrated film style, consistent with the starting frame. "
                             "Gentle, smooth motion. No close-up faces, no text."
                             + (" Sound: soft natural ambience of the scene only, no speech, no music." if VEO_AUDIO else ""),
        image=types.Image(image_bytes=raw.inline_data.data, mime_type=raw.inline_data.mime_type),
        config=types.GenerateVideosConfig(duration_seconds=8, aspect_ratio="16:9", number_of_videos=1,
                                          generate_audio=VEO_AUDIO),
    )
    job = uuid.uuid4().hex[:12]
    JOBS[job] = {"op": op, "started": time.time()}
    return {"id": job}


@app.get("/api/clip/{job}")
async def clip_status(job: str):
    item = JOBS.get(job)
    if not item:
        raise HTTPException(status_code=404, detail="Unknown clip. The server may have restarted; start again.")
    if job in MEDIA:
        return {"done": True, "url": f"/api/media/{job}.mp4", "last_frame": item.get("last_frame")}
    op = video_client.operations.get(item["op"])
    item["op"] = op
    if not op.done:
        return {"done": False, "seconds": int(time.time() - item["started"])}
    if getattr(op, "error", None):
        return {"done": True, "error": str(op.error)}
    vids = op.response.generated_videos if op.response else []
    if not vids or not vids[0].video.video_bytes:
        return {"done": True, "error": "No video came back; it may have been filtered. Try a different scene."}
    MEDIA[job] = vids[0].video.video_bytes
    item["last_frame"] = last_frame(MEDIA[job])
    return {"done": True, "url": f"/api/media/{job}.mp4", "last_frame": item["last_frame"]}


class FilmIn(BaseModel):
    clips: list[str]
    narration: list[str] = []     # one line per clip, read by the narrator
    language: str = "auto"
    voice: str | None = None      # a short piece of the storyteller's own recording (data URL) for the ending
    teller: str = ""


V = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-r", "24"]
A = ["-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2"]


def has_audio(path: str) -> bool:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                        "-of", "csv=p=0", path], capture_output=True, text=True)
    return bool(r.stdout.strip())


@app.post("/api/film")
async def film(body: FilmIn):
    missing = [c for c in body.clips if c not in MEDIA]
    if missing or not body.clips:
        raise HTTPException(status_code=400, detail="Some clips are missing; generate them again.")
    n = len(body.clips)
    with tempfile.TemporaryDirectory() as d:
        j = lambda *x: os.path.join(d, *x)
        # 1. picture: one continuous 1280x720 track (keeping Veo's natural sound if it made any)
        parts = []
        for i, c in enumerate(body.clips):
            raw, norm = j(f"c{i}.mp4"), j(f"n{i}.mp4")
            open(raw, "wb").write(MEDIA[c])
            if has_audio(raw):
                run_ffmpeg(["-i", raw, "-t", "8", "-vf", "scale=1280:720,setsar=1", *V, *A, norm])
            else:
                run_ffmpeg(["-i", raw, "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", "8",
                            "-vf", "scale=1280:720,setsar=1", *V, *A, "-shortest", norm])
            parts.append(norm)
        with open(j("list.txt"), "w") as f:
            f.writelines(f"file '{p}'\n" for p in parts)
        run_ffmpeg(["-f", "concat", "-safe", "0", "-i", j("list.txt"), "-c", "copy", j("pic.mp4")])
        # 2. narration: each line placed at the start of its own 8 s shot
        segs = []
        for i in range(n):
            line = (body.narration[i] if i < len(body.narration) else "").strip()
            seg = j(f"s{i}.wav")
            if line:
                pcm = tts_pcm(line, body.language, "warm, tender storyteller")
                open(j(f"t{i}.wav"), "wb").write(pcm_to_wav(pcm))
                dur = len(pcm) / 48000
                speed = min(1.3, dur / 7.0) if dur > 7.0 else 1.0
                run_ffmpeg(["-i", j(f"t{i}.wav"), "-af", f"atempo={speed:.3f},adelay=500|500,apad,atrim=0:8",
                            "-ar", "48000", "-ac", "2", seg])
            else:
                run_ffmpeg(["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", "8", seg])
            segs.append(seg)
        with open(j("alist.txt"), "w") as f:
            f.writelines(f"file '{p}'\n" for p in segs)
        run_ffmpeg(["-f", "concat", "-safe", "0", "-i", j("alist.txt"), j("narr.wav")])
        # 3. mix: narration on top, scene sound softly underneath
        run_ffmpeg(["-i", j("pic.mp4"), "-i", j("narr.wav"), "-filter_complex",
                    "[0:a]volume=0.35[amb];[1:a]volume=1.6[nar];[amb][nar]amix=inputs=2:duration=first:normalize=0[a]",
                    "-map", "0:v", "-map", "[a]", "-c:v", "copy", *A, j("main.mp4")])
        final = j("main.mp4")
        # 4. ending: the last picture held, with a few seconds of their own voice
        if body.voice:
            try:
                open(j("voice.in"), "wb").write(b64_part(body.voice, "audio/webm").inline_data.data)
                run_ffmpeg(["-sseof", "-0.2", "-i", j("pic.mp4"), "-update", "1", "-q:v", "2", j("end.jpg")])
                run_ffmpeg(["-loop", "1", "-i", j("end.jpg"), "-i", j("voice.in"), "-t", "12",
                            "-vf", "scale=1280:720,setsar=1,fade=t=in:st=0:d=0.8,fade=t=out:st=11:d=1",
                            "-af", "afade=t=in:st=0:d=0.5,afade=t=out:st=10.5:d=1.5,apad", *V, *A, "-shortest", j("tail.mp4")])
                with open(j("flist.txt"), "w") as f:
                    f.write(f"file '{j('main.mp4')}'\nfile '{j('tail.mp4')}'\n")
                run_ffmpeg(["-f", "concat", "-safe", "0", "-i", j("flist.txt"), *V, *A, j("final.mp4")])
                final = j("final.mp4")
            except Exception:
                final = j("main.mp4")
        run_ffmpeg(["-i", final, "-c", "copy", "-movflags", "+faststart", j("out.mp4")])
        fid = "film-" + uuid.uuid4().hex[:10]
        MEDIA[fid] = open(j("out.mp4"), "rb").read()
    return {"id": fid, "url": f"/api/media/{fid}.mp4", "seconds": 8 * n + (12 if body.voice else 0)}


@app.get("/api/media/{name}")
async def media(name: str):
    key = name.removesuffix(".mp4")
    if key not in MEDIA:
        raise HTTPException(status_code=404, detail="Not found. The server may have restarted.")
    return Response(MEDIA[key], media_type="video/mp4")


# ---------- voice: Gemini TTS ----------

LANG_CODE = {"hindi": "hi-IN", "bengali": "bn-BD", "bangla": "bn-BD", "tamil": "ta-IN", "telugu": "te-IN",
             "marathi": "mr-IN", "english": "en-IN", "japanese": "ja-JP", "korean": "ko-KR", "thai": "th-TH",
             "vietnamese": "vi-VN", "bahasa indonesia": "id-ID", "indonesian": "id-ID", "gujarati": "gu-IN",
             "punjabi": "pa-IN", "urdu": "ur-IN", "mandarin chinese": "cmn-CN", "cantonese": "yue-HK",
             "tagalog": "fil-PH"}
TTS_CACHE = {}
VOICES = {"Sulafat", "Gacrux", "Achernar", "Charon", "Kore", "Aoede", "Algieba"}


def pcm_to_wav(pcm: bytes, rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(pcm)
    return buf.getvalue()


def tts_pcm(text: str, language: str = "auto", tone: str = "warm", voice: str | None = None) -> bytes:
    voice = voice if voice in VOICES else TTS_VOICE
    key = hashlib.sha1(f"{text}|{language}|{tone}|{voice}".encode()).hexdigest()
    if key in TTS_CACHE:
        return TTS_CACHE[key]
    spend("tts")
    style = (f"Read this in a {tone} voice at a natural, gently brisk pace, with real feeling, like a caring "
             f"grandchild talking to their grandparent. Short natural pauses, never slow or robotic") if voice == TTS_VOICE else (
             f"Speak as a {tone}. Unhurried, with natural pauses and real feeling")
    code = LANG_CODE.get((language or "").strip().lower())
    speech = types.SpeechConfig(voice_config=types.VoiceConfig(
        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)))
    if code:
        speech.language_code = code
    last = None
    for model in TTS_MODELS:
        try:
            resp = gen(
                model=model, contents=f"{style}: {text}",
                config=types.GenerateContentConfig(response_modalities=["AUDIO"], speech_config=speech))
            data = resp.candidates[0].content.parts[0].inline_data.data
            if data:
                TTS_CACHE[key] = data
                return data
        except Exception as e:      # try the next voice model
            last = e
    raise HTTPException(status_code=502, detail=f"The voice isn't available right now ({last}).")


class SpeakIn(BaseModel):
    text: str
    language: str = "auto"
    tone: str = "warm"
    voice: str | None = None


@app.post("/api/speak")
async def speak(body: SpeakIn):
    text = body.text.strip()[:600]
    if not text:
        raise HTTPException(status_code=400, detail="Nothing to say.")
    return Response(pcm_to_wav(tts_pcm(text, body.language, body.tone, body.voice)), media_type="audio/wav")


# ---------- misc ----------

@app.get("/api/usage")
async def usage():
    led = SESSIONS.get(CURRENT.get(), {"spent": 0, "items": []})
    spent = METER.get("spent_inr", 0)
    return {"session": {"limit": SESSION_TOKENS, "spent": led["spent"], "left": SESSION_TOKENS - led["spent"],
                        "items": led["items"][-200:]}, "cost_tokens": COST_TOKENS, "cost_label": COST_LABEL,
            "used": USAGE, "limits": LIMITS, "credit_start_inr": CREDIT_START,
            "spent_inr": round(spent, 2), "left_inr": round(CREDIT_START - spent, 2),
            "calls": METER.get("calls", {}), "cost_inr": COST_INR, "veo_audio": VEO_AUDIO,
            "models": {"text": TEXT_MODEL, "image": IMAGE_MODEL, "video": VIDEO_MODEL, "voice": TTS_MODELS[0]}}


@app.exception_handler(HTTPException)
async def http_error(_, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.exception_handler(Exception)
async def any_error(_, exc: Exception):
    msg = str(exc)
    if any(k in msg for k in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE")):
        return JSONResponse(status_code=503, content={"error": "Lantern is very busy right now. Please wait a few seconds and try again."})
    print("error:", type(exc).__name__, msg[:800], flush=True)
    return JSONResponse(status_code=500, content={"error": "Something went wrong. Please try again."})


@app.get("/")
async def index():
    return FileResponse("static/index.html")


app.mount("/", StaticFiles(directory="static"), name="static")
