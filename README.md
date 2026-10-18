# Lantern

**Tell me again.** Lantern turns the stories a grandparent tells into a family storybook page: in their own words and language, with an English translation, picture pages, and the history around the story, then brings it back to the family on the days that matter.

Built for the Google Cloud AI Builder Cup 2026, theme **Media, Content & Digital Experiences**.

## How it works

1. **Capture a memory.** The storyteller talks in their own language. An old photo or meaningful object is optional and helps memories come back. Every answer is transcribed in the original script and translated.
2. **Research & interview loop.** After each answer Lantern updates a structured memory (people, places, a timeline with certainty, known and unknown), researches three layers of history with Google Search (macro events, local history, everyday life), finds gaps (timeline, context, experience, relationship, inconsistency), and asks one neutral, non-leading question with a tracked reason. Conflicts with the historical record become gentle clarifying questions, never corrections. A six-point **Story Readiness Check** decides when to propose closing; the storyteller can skip any question or stop at any time.
3. **Reconstruct & review.** The interview becomes a narrative split into personal memories (marked lived or heard from relatives), verified facts with sources, and uncertain details. The family edits and approves it before anything is generated.
4. **Choose an output.** *Storybook*: original words with translation, restored photo, illustrations, and history with citations. *Short film*: a storyboard of 8-second Veo clips where the last frame of each clip starts the next, joined into one film. Both use the same approved story, so the other format can be made later without repeating the interview.
5. **Preserve & share.** Original recordings, the approved narrative, and the storybook or film are saved to the family collection and can be downloaded or shared.

**Principles:** restore and retell, never impersonate. Unknowns stay unknown, no invented words, no cloned voice, no animated real faces.

## Google Cloud stack

| Step | Service / model |
| --- | --- |
| Interview loop: transcription, translation, memory, gaps, readiness | Gemini on Vertex AI: `gemini-3.8-flash` |
| Historical research with citations | `gemini-3.8-flash` + Grounding with Google Search |
| Story reconstruction, storyboard | `gemini-3.8-flash` |
| Photo restoration, illustrations, film opening frame | `gemini-nano-banana-2.1` |
| Film clips chained by last frame | Veo: `veo-3.1-fast-generate-001`, joined with ffmpeg |
| Warm spoken questions, film narration | Gemini TTS: `gemini-3.1-flash-tts-preview` (fallback `gemini-2.5-flash-tts`) |
| Hosting | Cloud Run (one service serves the web app and the API) |

## Cost guard

Every paid model call is counted and capped per instance (defaults: 300 text, 40 image, 8 video, 400 voice calls), and an estimated credit meter (kept in a Cloud Storage bucket, `DATA_BUCKET`) is shown at the top of the app. The service is deployed with `--max-instances 1`, so a public link can't run up a large bill. Current usage: `GET /api/usage`.

## Run it on Google Cloud

```bash
gcloud config set project YOUR_PROJECT_ID
gcloud services enable aiplatform.googleapis.com run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
gcloud run deploy lantern --source . --region us-central1 --allow-unauthenticated \
  --max-instances 1 --memory 1Gi --timeout 300 --set-env-vars PROJECT_ID=YOUR_PROJECT_ID
```

## Project layout

```
main.py            FastAPI app: API endpoints and cost guard
static/index.html  Phone-first web app
Dockerfile         Python 3.12 + ffmpeg (converts browser audio for Gemini)
```
