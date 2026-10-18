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
  GET  /api/usage           paid calls made by this instance, and the caps

Cost guard: every paid call is counted and capped per running instance (see LIMITS); the service runs
with max-instances=1, so a public link can't run up a large bill.
"""
import base64
import json
import os
import subprocess
import tempfile
import time
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from pydantic import BaseModel

PROJECT_ID = os.environ.get("PROJECT_ID", "lantern-aibc")
TEXT_MODEL = os.environ.get("TEXT_MODEL", "gemini-3.8-flash")
IMAGE_MODEL = os.environ.get("IMAGE_MODEL", "gemini-nano-banana-2.1")
VIDEO_MODEL = os.environ.get("VIDEO_MODEL", "veo-3.1-fast-generate-001")

LIMITS = {
    "text": int(os.environ.get("MAX_TEXT_CALLS", "300")),
    "image": int(os.environ.get("MAX_IMAGE_CALLS", "40")),
    "video": int(os.environ.get("MAX_VIDEO_CALLS", "8")),
}
USAGE = {"text": 0, "image": 0, "video": 0}

text_client = genai.Client(vertexai=True, project=PROJECT_ID, location="global")
video_client = genai.Client(vertexai=True, project=PROJECT_ID, location="us-central1")

app = FastAPI(title="Lantern")
JOBS = {}    # clip id -> {"op", "started", "prompt"}
MEDIA = {}   # media id -> mp4 bytes (clips and films; kept in memory for this demo)
SEARCH = types.Tool(google_search=types.GoogleSearch())


# ---------- helpers ----------

def spend(kind: str):
    if USAGE[kind] >= LIMITS[kind]:
        raise HTTPException(status_code=429,
                            detail=f"The demo's {kind} limit ({LIMITS[kind]}) is used up. Ask the owner to raise it.")
    USAGE[kind] += 1


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


def ask_text(parts, temperature=0.4, search=False):
    spend("text")
    cfg = types.GenerateContentConfig(temperature=temperature, tools=[SEARCH] if search else None)
    return text_client.models.generate_content(model=TEXT_MODEL, contents=parts, config=cfg)


# ---------- 1 + 2. capture, research and interview ----------

INTERVIEWER = """You are Lantern, a warm, patient interviewer helping a family record {teller}'s memories,
and a careful historian. Preferred language: {language} (if "auto", use the language they speak).
{photo_note}

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
2. RESEARCH (Google Search) the world around the storyteller, in three layers, only for what is not
   already in the research notes:
   - macro: wars, migrations, political, economic and social change
   - local: what was happening in that city, village, neighbourhood or community at that time
   - everyday: food, housing, work, transport, schooling, family life, fears, practical limits
   Write at most 3 new notes. For each, say how it MIGHT connect to the memory, without assuming the
   family lived through everything typical of the period.
3. FIND GAPS in the personal narrative: timeline, context (why a decision), experience (what they saw
   or did), relationship (who influenced it), and inconsistency (conflicts with documented chronology).
   Prioritise them (1 = most useful). If something conflicts with the record, do not correct them;
   record it as a gentle check.
4. READINESS CHECK. Mark each true only if met: core timeline established (dates may be approximate);
   people and places identified or explicitly unknown; historical context researched without unsupported
   assumptions; major gaps explored, resolved or marked unknown; enough authentic material for a chapter
   or a few film scenes; uncertainty (conflicts, secondhand memories, unverifiable details) documented.
   Set "ready_to_close" true only when additional questions are unlikely to meaningfully improve the story.
5. ASK ONE QUESTION, chosen for the biggest improvement: clarify an important inconsistency, explain a
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
        typed=answer_text, skip_note=skip_note, mode_note=FAMILY_NOTE.format(teller=teller) if mode == "family" else "",
        photo_note="They are looking at the attached photo or object." if has_photo else "No photo was shared."))
    resp = ask_text(parts, temperature=0.5, search=True)
    out = parse_json(resp.text)
    srcs = sources_of(resp)
    for note in out.get("research", []) or []:
        note["sources"] = srcs[:3]
    return out


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
Reply with JSON only:
{{"title": "...", "opening_frame": "...",
  "shots": [{{"caption": "...", "motion": "camera and scene motion for this 8 s shot"}}]}}"""


@app.post("/api/plan")
async def plan(body: PlanIn):
    story = json.dumps(body.story, ensure_ascii=False)
    if body.format == "film":
        prompt = PLAN_FILM.format(story=story, shots=max(2, min(body.shots, 4)))
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
    return first_image(text_client.models.generate_content(model=IMAGE_MODEL, contents=parts, config=cfg))


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
                             "Gentle, smooth motion. No close-up faces, no text.",
        image=types.Image(image_bytes=raw.inline_data.data, mime_type=raw.inline_data.mime_type),
        config=types.GenerateVideosConfig(duration_seconds=8, aspect_ratio="16:9", number_of_videos=1,
                                          generate_audio=False),
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


@app.post("/api/film")
async def film(body: FilmIn):
    missing = [c for c in body.clips if c not in MEDIA]
    if missing or not body.clips:
        raise HTTPException(status_code=400, detail="Some clips are missing; generate them again.")
    with tempfile.TemporaryDirectory() as d:
        listing = os.path.join(d, "list.txt")
        with open(listing, "w") as f:
            for i, c in enumerate(body.clips):
                path = os.path.join(d, f"{i}.mp4")
                open(path, "wb").write(MEDIA[c])
                f.write(f"file '{path}'\n")
        out = os.path.join(d, "film.mp4")
        run_ffmpeg(["-f", "concat", "-safe", "0", "-i", listing, "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", out])
        fid = "film-" + uuid.uuid4().hex[:10]
        MEDIA[fid] = open(out, "rb").read()
    return {"id": fid, "url": f"/api/media/{fid}.mp4", "seconds": 8 * len(body.clips)}


@app.get("/api/media/{name}")
async def media(name: str):
    key = name.removesuffix(".mp4")
    if key not in MEDIA:
        raise HTTPException(status_code=404, detail="Not found. The server may have restarted.")
    return Response(MEDIA[key], media_type="video/mp4")


# ---------- misc ----------

@app.get("/api/usage")
async def usage():
    return {"used": USAGE, "limits": LIMITS,
            "models": {"text": TEXT_MODEL, "image": IMAGE_MODEL, "video": VIDEO_MODEL}}


@app.exception_handler(HTTPException)
async def http_error(_, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.exception_handler(Exception)
async def any_error(_, exc: Exception):
    return JSONResponse(status_code=500, content={"error": f"{type(exc).__name__}: {exc}"[:500]})


@app.get("/")
async def index():
    return FileResponse("static/index.html")


app.mount("/", StaticFiles(directory="static"), name="static")
