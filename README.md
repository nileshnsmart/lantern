# Lantern

**Tell me again.** Lantern turns the stories a grandparent tells into a family storybook page: in their own words and language, with an English translation, picture pages, and the history around the story, then brings it back to the family on the days that matter.

Built for the Google Cloud AI Builder Cup 2026, theme **Media, Content & Digital Experiences**.

## How it works

1. **Capture.** Photograph an old photo or heirloom. Lantern looks at it and asks the first question in the grandparent's language. They answer by voice (or the family types the answer). Each answer is transcribed, translated, and followed by one warm follow-up question.
2. **Enrich.** Lantern pulls out the people, places, years and objects that were mentioned, then uses Grounding with Google Search to add three cited facts about that place and time, kept separate from the teller's words.
3. **Render.** A chapter in the teller's own words (cleaned, never invented) beside an English translation, a restored cover photo, three illustrated picture pages, and an optional 8-second clip of the setting.
4. **Resurface.** The family chooses days for the story to return: birthdays, festivals of remembrance (Pitru Paksha, Qingming, Obon, Chuseok, Undas), gatherings.

**Principles:** restore and retell, never impersonate. No invented words, no cloned voice, no animated real faces.

## Google Cloud stack

| Step | Service / model |
| --- | --- |
| Interview turns, transcription, translation | Gemini on Vertex AI: `gemini-3.8-flash` |
| History context with citations | `gemini-3.8-flash` + Grounding with Google Search |
| Photo restoration, picture pages | `gemini-nano-banana-2.1` |
| Setting clip | Veo: `veo-3.1-fast-generate-001` |
| Hosting | Cloud Run (one service serves the web app and the API) |

## Cost guard

Every paid model call is counted and capped per instance (defaults: 300 text, 40 image, 3 video calls). The service is deployed with `--max-instances 1`, so a public link can't run up a large bill. Current usage: `GET /api/usage`.

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
