# Lantern

**Record a story, for your loved ones.** Lantern listens to a grandparent tell a story in their own voice and language, asks gentle follow-up questions out loud, researches the history around it, and keeps it as an illustrated storybook (PDF) or a short narrated film (MP4) that ends in their own voice.

Built for the Google Cloud AI Builder Cup 2026, theme **Media, Content & Digital Experiences**.

**Live app:** https://lantern-539412975581.us-central1.run.app · guided demo: add `?demo=1`

## How it works

1. **Before we begin.** Three optional boxes: who is telling the story, who it is for, what it is about. Choose **Lantern asks** (Lantern asks questions out loud on behalf of the listener) or **I will ask** (a family member asks and Lantern simply records). An old photo and a language are optional. Nothing needs to be filled in to start.
2. **Tell your story.** One big microphone. The speaker's words appear live on screen with a live translation into another language. After each answer Lantern replies with a short, caring acknowledgement and one gentle follow-up question in a warm voice. Meanwhile it updates a structured memory (people, places, a timeline with certainty), researches the history with Google Search, finds gaps, and tracks a six-point Story Readiness Check. The storyteller can skip any question, type instead, or stop at any time.
3. **Review together.** The conversation becomes a story split into *In their words* (lived or heard from family), *History* (with sources) and *Not sure yet*. The family can listen to it, edit, relabel, remove or add lines, and approves it before anything is generated.
4. **Create a keepsake.** *Storybook*: their original words with translation, illustrations, restored photo and history with citations, saved as a PDF. *Short film*: 8-second Veo scenes chained by their last frame, narrated in the storyteller's language, ending with a moment of their own recorded voice, saved as an MP4. The other format can be made later from the same approved story.
5. **Share.** Save or share the PDF or MP4 straight from the phone. The approved words are kept in Cloud Firestore, so *My stories* still has them after a page refresh.

**Principles:** restore and retell, never impersonate. Unknowns stay unknown, no invented words, no cloned voice, no animated real faces.

## Google Cloud stack

| Step | Service / model |
| --- | --- |
| Interview loop: transcription, translation, memory, gaps, readiness, next question | Gemini on Vertex AI: `gemini-3.8-flash` |
| Historical research with citations | `gemini-3.8-flash` + Grounding with Google Search |
| Live translation of the transcript | `gemini-3.8-flash` |
| Story reconstruction, storybook and film storyboard | `gemini-3.8-flash` |
| Photo restoration, illustrations, film opening frame | `gemini-nano-banana-2.1` |
| Film scenes chained by last frame | Veo: `veo-3.1-fast-generate-001`, joined with ffmpeg |
| Warm spoken questions, story read-aloud, film narration | Gemini TTS: `gemini-3.1-flash-tts-preview` (fallback `gemini-2.5-flash-tts`) |
| Saved stories and each session's token ledger | Cloud Firestore (Firebase), written in the background so nobody waits |
| Hosting, credit meter | Cloud Run (one service for app and API), Cloud Storage |

The live on-screen transcript while someone is speaking uses the browser's speech recognition; the saved transcript and translation come from Gemini.

## Tokens and cost guard

Every visitor session gets **10,000 tokens** (1 token ≈ ₹0.10), shown at the top of the app. Tapping it shows what this session used and what each action costs (question 10, research 40, translation 1, picture 35, voice 3, film scene 680). Each server instance also has hard caps on paid calls, and the service runs with `--max-instances 1`, so a public link can't run up a large bill. If Google's models are briefly busy, Lantern waits and retries before showing a friendly message.

## Run it on Google Cloud

```bash
gcloud config set project YOUR_PROJECT_ID
gcloud services enable aiplatform.googleapis.com run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com firestore.googleapis.com
gcloud firestore databases create --location=nam5   # once
gcloud run deploy lantern --source . --region us-central1 --allow-unauthenticated \
  --max-instances 1 --memory 1Gi --timeout 600 --set-env-vars PROJECT_ID=YOUR_PROJECT_ID
```

## Project layout

```
main.py             FastAPI app: interview, research, story, images, film, voice, tokens, Firestore
static/index.html   Landing page
static/studio.html  The storytelling studio (phone-first)
static/style.css    Design system
static/assets/      Images and logo
Dockerfile          Python 3.12 + ffmpeg (audio conversion, film assembly)
```
