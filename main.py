"""Lantern: turn a grandparent's spoken stories into a bilingual, history-grounded storybook.

One Cloud Run service serves the web page (static/) and these JSON endpoints:
  POST /api/turn        one interview turn: photo + recorded answer -> transcript, translation, next question
  POST /api/story       whole interview -> chapter in the teller's words + English, entities, scenes, era facts with sources
  POST /api/restore     old photo -> restored cover image (Nano Banana)
  POST /api/illustrate  scene prompt (+ photo as character reference) -> illustrated page (Nano Banana)
  POST /api/animate     illustrated scene -> starts an 8 s Veo clip (setting only)
  GET  /api/animate/{id} poll the clip
  GET  /api/usage       how many paid calls this instance has made, and the caps

Cost guard: every paid call is counted and capped per running instance (see LIMITS),
so a public link can never run up a large bill.
"""
import base64
import json
import os
import subprocess
import tempfile
import time
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from pydantic import BaseModel

PROJECT_ID = os.environ.get("PROJECT_ID", "lantern-aibc")
TEXT_MODEL = os.environ.get("TEXT_MODEL", "gemini-3.8-flash")
IMAGE_MODEL = os.environ.get("IMAGE_MODEL", "gemini-nano-banana-2.1")
VIDEO_MODEL = os.environ.get("VIDEO_MODEL", "veo-3.1-fast-generate-001")

# Paid-call caps per running instance. Cloud Run is deployed with max-instances=1.
LIMITS = {
    "text": int(os.environ.get("MAX_TEXT_CALLS", "300")),
    "image": int(os.environ.get("MAX_IMAGE_CALLS", "40")),
    "video": int(os.environ.get("MAX_VIDEO_CALLS", "3")),
}
USAGE = {"text": 0, "image": 0, "video": 0}

text_client = genai.Client(vertexai=True, project=PROJECT_ID, location="global")
video_client = genai.Client(vertexai=True, project=PROJECT_ID, location="us-central1")

app = FastAPI(title="Lantern")
VIDEO_JOBS = {}


def spend(kind: str):
    if USAGE[kind] >= LIMITS[kind]:
        raise HTTPException(
            status_code=429,
            detail=f"The demo's {kind} limit ({LIMITS[kind]}) is used up. Ask the owner to raise it.",
        )
    USAGE[kind] += 1


def to_mp3(raw: bytes) -> bytes:
    """Browsers record webm or mp4; convert to 16 kHz mono mp3, which Gemini reads reliably."""
    with tempfile.TemporaryDirectory() as d:
        src, dst = os.path.join(d, "in"), os.path.join(d, "out.mp3")
        with open(src, "wb") as f:
            f.write(raw)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-ac", "1", "-ar", "16000", dst],
            check=True,
        )
        with open(dst, "rb") as f:
            return f.read()


def parse_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    return json.loads(text[text.find("{"): text.rfind("}") + 1])


def b64_part(b64: str, mime: str = "image/jpeg"):
    if "," in b64[:80]:
        header, b64 = b64.split(",", 1)
        if ":" in header and ";" in header:
            mime = header.split(":")[1].split(";")[0]
    return types.Part.from_bytes(data=base64.b64decode(b64), mime_type=mime)


def first_image(resp) -> str:
    for cand in resp.candidates or []:
        for part in cand.content.parts or []:
            if getattr(part, "inline_data", None) and part.inline_data.data:
                mime = part.inline_data.mime_type or "image/png"
                return f"data:{mime};base64," + base64.b64encode(part.inline_data.data).decode()
    raise HTTPException(status_code=502, detail="The image model returned no image. Try again.")


INTERVIEWER = """You are Lantern, a warm, patient interviewer. A grandchild is recording their
grandparent ({teller}) telling family stories. The grandparent is looking at the attached photo or
heirloom. The grandparent's language: {language} (if "auto", detect it from the audio).

Rules:
- If an audio answer is attached, transcribe it EXACTLY as spoken, in its original language and script
  (Hindi in Devanagari, Chinese in Chinese characters, and so on). Do not add or change facts.
- Translate that transcript faithfully into English.
- Then ask ONE short, kind follow-up question in the grandparent's language, plus its English translation.
  Ask about concrete things: who is in the photo, where, which year, smells, sounds, objects, feelings.
  Build on what they just said. Never suggest answers or put words in their mouth.
- If a typed answer is given instead of audio, treat it as the transcript (keep its language).
- If there is no answer yet, open the interview with one gentle question about the photo.
- After about 6 answers, set "done" to true and ask a warm closing question.

Previous turns (JSON): {history}
Typed answer (may be empty): {typed}

Reply with JSON only:
{{"language": "...", "transcript_original": "...", "transcript_english": "...",
  "question_original": "...", "question_english": "...", "done": false}}"""


@app.post("/api/turn")
async def turn(
    teller: str = Form("Nani"),
    language: str = Form("auto"),
    history: str = Form("[]"),
    photo: UploadFile | None = File(None),
    audio: UploadFile | None = File(None),
    answer_text: str = Form(""),
):
    parts = []
    if photo is not None:
        data = await photo.read()
        if data:
            parts.append(types.Part.from_bytes(data=data, mime_type=photo.content_type or "image/jpeg"))
    if audio is not None:
        data = await audio.read()
        if data:
            parts.append(types.Part.from_bytes(data=to_mp3(data), mime_type="audio/mpeg"))
    parts.append(INTERVIEWER.format(teller=teller, language=language, history=history, typed=answer_text))
    spend("text")
    resp = text_client.models.generate_content(
        model=TEXT_MODEL,
        contents=parts,
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.6),
    )
    return parse_json(resp.text)


class StoryIn(BaseModel):
    teller: str = "Nani"
    history: list = []
    photo: str | None = None


STORY_PROMPT = """You turn an interview with a grandparent ({teller}) into one storybook chapter.
Interview turns (JSON, each has the original-language answer and its English translation):
{history}

Rules:
- Use ONLY the grandparent's own words. You may remove fillers and repetitions and join sentences,
  but never invent events, names, dates or feelings.
- Keep each paragraph in the original language and give a faithful English translation.
- Pull out people, places, years and objects that were actually mentioned.
- Propose 3 picture-book scenes from the story. Each prompt describes a warm illustration of the
  setting and action; people are small or seen from behind.

Reply with JSON only:
{{"title": "...", "language": "...",
  "paragraphs": [{{"original": "...", "english": "..."}}],
  "people": ["..."], "places": ["..."], "years": ["..."], "objects": ["..."],
  "scenes": [{{"caption": "...", "prompt": "..."}}]}}"""


ERA_PROMPT = """A grandparent told a family story set in {places} around {years}. Story summary: {summary}
Using Google Search, write exactly 3 short, neutral, factual sentences about what that place and
time were like (daily life, transport, notable events) that a child reading the story would find
helpful. One sentence per line. No speculation about this family. Avoid graphic detail."""


@app.post("/api/story")
async def story(body: StoryIn):
    parts = []
    if body.photo:
        parts.append(b64_part(body.photo))
    parts.append(STORY_PROMPT.format(teller=body.teller, history=json.dumps(body.history, ensure_ascii=False)))
    spend("text")
    resp = text_client.models.generate_content(
        model=TEXT_MODEL,
        contents=parts,
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.3),
    )
    chapter = parse_json(resp.text)

    summary = " ".join(p.get("english", "") for p in chapter.get("paragraphs", []))[:1500]
    spend("text")
    era = text_client.models.generate_content(
        model=TEXT_MODEL,
        contents=ERA_PROMPT.format(
            places=", ".join(chapter.get("places", [])) or "the places in the story",
            years=", ".join(chapter.get("years", [])) or "the time of the story",
            summary=summary,
        ),
        config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())], temperature=0.2),
    )
    facts = [ln.strip(" -•*0123456789.") for ln in (era.text or "").splitlines() if len(ln.strip()) > 20][:3]
    sources, seen = [], set()
    meta = era.candidates[0].grounding_metadata if era.candidates else None
    for chunk in (getattr(meta, "grounding_chunks", None) or []):
        web = getattr(chunk, "web", None)
        if web and web.uri and web.uri not in seen:
            seen.add(web.uri)
            sources.append({"title": web.title or web.uri, "uri": web.uri})
    chapter["era"] = {"facts": facts, "sources": sources[:5]}
    return chapter


class ImageIn(BaseModel):
    photo: str | None = None
    prompt: str | None = None


@app.post("/api/restore")
async def restore(body: ImageIn):
    if not body.photo:
        raise HTTPException(status_code=400, detail="Add a photo first.")
    spend("image")
    resp = text_client.models.generate_content(
        model=IMAGE_MODEL,
        contents=[b64_part(body.photo), "Restore this old family photo: remove scratches, dust and blur, "
                  "gently improve contrast, keep every face, pose and detail exactly as it is. Do not add anything."],
        config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
    )
    return {"image": first_image(resp)}


@app.post("/api/illustrate")
async def illustrate(body: ImageIn):
    if not body.prompt:
        raise HTTPException(status_code=400, detail="Missing scene description.")
    parts = []
    if body.photo:
        parts.append(b64_part(body.photo))
        parts.append("Use the people in this photo only as character reference (clothing, hair, era).")
    parts.append("Warm, hand-painted children's picture-book illustration, soft light, period-accurate "
                 "details, no text or lettering. Scene: " + body.prompt)
    spend("image")
    resp = text_client.models.generate_content(
        model=IMAGE_MODEL,
        contents=parts,
        config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
    )
    return {"image": first_image(resp)}


class AnimateIn(BaseModel):
    image: str
    prompt: str = "Gentle camera push-in, soft natural motion in the scene, calm and nostalgic."


@app.post("/api/animate")
async def animate(body: AnimateIn):
    spend("video")
    raw = b64_part(body.image)
    op = video_client.models.generate_videos(
        model=VIDEO_MODEL,
        prompt=body.prompt + " Illustrated picture-book style. Focus on the place and objects; no close-up faces.",
        image=types.Image(image_bytes=raw.inline_data.data, mime_type=raw.inline_data.mime_type),
        config=types.GenerateVideosConfig(duration_seconds=8, aspect_ratio="16:9", number_of_videos=1,
                                          generate_audio=False),
    )
    job = uuid.uuid4().hex[:12]
    VIDEO_JOBS[job] = {"op": op, "started": time.time()}
    return {"id": job}


@app.get("/api/animate/{job}")
async def animate_status(job: str):
    item = VIDEO_JOBS.get(job)
    if not item:
        raise HTTPException(status_code=404, detail="Unknown clip.")
    op = video_client.operations.get(item["op"])
    item["op"] = op
    if not op.done:
        return {"done": False, "seconds": int(time.time() - item["started"])}
    if getattr(op, "error", None):
        return {"done": True, "error": str(op.error)}
    vids = op.response.generated_videos if op.response else []
    if not vids or not vids[0].video.video_bytes:
        return {"done": True, "error": "No video returned (it may have been filtered)."}
    return {"done": True, "video": "data:video/mp4;base64," + base64.b64encode(vids[0].video.video_bytes).decode()}


@app.get("/api/usage")
async def usage():
    return {"used": USAGE, "limits": LIMITS, "models": {"text": TEXT_MODEL, "image": IMAGE_MODEL, "video": VIDEO_MODEL}}


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
