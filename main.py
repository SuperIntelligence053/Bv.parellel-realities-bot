import os
import json
import time
import random
import subprocess
from pathlib import Path

import requests

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

import edge_tts

ROOT = Path(__file__).parent
OUTDIR = ROOT / "outputs"
OUTDIR.mkdir(exist_ok=True)

STATE_PATH = ROOT / "state.json"


# ----------------------------
# 0) State (stop after 30)
# ----------------------------
def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"day": 0, "uploads": []}


def save_state(state):
    STATE_PATH.write_text(
        json.dumps(state, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ----------------------------
# 1) Gemini (text) generation
# ----------------------------
def gemini_generate_story(gemini_key: str) -> dict:
    """
    Calls Gemini and returns a dict with:
      title, hook, script, on_screen_captions, description, tags, image_prompts, voice
    Robustly extracts JSON even if Gemini adds extra text.
    """

    tones = ["cinematic", "intimate", "tense", "mysterious", "urgent", "reflective"]
    settings = [
        "near-future Europe",
        "desert megacity",
        "floating archipelago",
        "underground metro-world",
        "rainy neon port",
        "quiet rural town with a hidden anomaly",
    ]
    twist_types = [
        "moral dilemma",
        "unreliable narrator",
        "time-loop reveal",
        "cost-of-convenience",
        "sacrifice",
        "unexpected benefactor",
    ]

    tone = random.choice(tones)
    setting = random.choice(settings)
    twist = random.choice(twist_types)

    prompt = f"""
You are writing an original YouTube Shorts story (35–45 seconds).
Language: Dutch.
Genre: speculative "What if...?" micro-fiction.
Make it feel human and varied (avoid repeated catchphrases).
Setting hint: {setting}
Tone hint: {tone}
Twist flavor: {twist}

Return ONLY valid JSON with this schema:
{{
  "title": "max 60 chars, Dutch",
  "hook": "1 sentence",
  "script": "Full VO script, 90–120 words, Dutch",
  "on_screen_captions": ["8-12 short caption lines, Dutch"],
  "description": "2-3 sentences, Dutch",
  "tags": ["10-15 tags, Dutch/English mix ok"],
  "image_prompts": ["7 prompts, each describing a vertical cinematic scene, no text in image"],
  "voice": {{
    "edge_tts_voice": "nl-NL-MaartenNeural",
    "speed": "+0%",
    "pitch": "+0Hz"
  }}
}}

Constraints:
- No real brands, no real people, no copyrighted characters.
- Each image prompt: cinematic, dramatic lighting, realistic, vertical 9:16 composition.
- Script ends with a question to the viewer.
"""

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    headers = {"Content-Type": "application/json"}
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.9, "maxOutputTokens": 800},
    }

    r = requests.post(f"{url}?key={gemini_key}", headers=headers, json=body, timeout=90)
    r.raise_for_status()
    data = r.json()

    # Safely extract model text
    try:
        text = data["candidates"][0]["content"]["parts"][0].get("text", "")
    except Exception:
        text = ""

    text = (text or "").strip()

    if not text:
        # Print full response for debugging
        print("Gemini returned empty text. Full response JSON:")
        print(json.dumps(data, indent=2))
        raise ValueError("Gemini returned empty response text")

    # If Gemini returns fenced code blocks, strip them
    if text.startswith("```"):
        parts = text.split("```")
        # try to take the inside block
        if len(parts) >= 2:
            text = parts[1].strip()
        else:
            text = text.strip("`").strip()

    # Extract the first JSON object from the response
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text_json = text[start : end + 1]
    else:
        text_json = text

    # Parse JSON with debug on failure
    try:
        story = json.loads(text_json)
    except Exception as e:
        print("Gemini raw response (post-processed):")
        print(text_json)
        raise e

    # Minimal sanity checks
    required = ["title", "script", "description", "tags", "image_prompts", "voice"]
    missing = [k for k in required if k not in story]
    if missing:
        print("Gemini JSON missing keys:", missing)
        print("Gemini JSON was:")
        print(json.dumps(story, indent=2, ensure_ascii=False))
        raise ValueError(f"Gemini JSON missing keys: {missing}")

    # Ensure 7 prompts
    if not isinstance(story.get("image_prompts"), list) or len(story["image_prompts"]) < 7:
        raise ValueError("Gemini image_prompts must be a list with at least 7 prompts")

    return story


# ----------------------------
# 2) Hugging Face image generation
# ----------------------------
def hf_generate_image(hf_token: str, prompt: str, out_path: Path):
    """
    Uses HF serverless inference. Retries on 503 (model loading).
    """
    model = "stabilityai/stable-diffusion-xl-base-1.0"
    api_url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {hf_token}"}

    payload = {
        "inputs": prompt,
        "parameters": {
            "width": 1024,
            "height": 1792,
            "guidance_scale": 7.0,
            "num_inference_steps": 30,
        },
    }

    for attempt in range(8):
        resp = requests.post(api_url, headers=headers, json=payload, timeout=180)

        # HF returns 503 while loading
        if resp.status_code == 503:
            wait = 8 + attempt * 4
            print(f"HF 503 (loading). Waiting {wait}s...")
            time.sleep(wait)
            continue

        # HF returns JSON errors sometimes
        if "application/json" in resp.headers.get("content-type", ""):
            try:
                err = resp.json()
            except Exception:
                err = {"raw": resp.text}
            raise RuntimeError(f"HF error: {resp.status_code} {err}")

        resp.raise_for_status()
        out_path.write_bytes(resp.content)
        return

    raise RuntimeError("HF image generation failed after retries.")


# ----------------------------
# 3) Voiceover via edge-tts
# ----------------------------
async def make_voice(script: str, voice: str, out_mp3: Path, rate: str = "+0%", pitch: str = "+0Hz"):
    communicate = edge_tts.Communicate(text=script, voice=voice, rate=rate, pitch=pitch)
    await communicate.save(str(out_mp3))


# ----------------------------
# 4) FFmpeg video build
# ----------------------------
def run(cmd):
    subprocess.run(cmd, check=True)


def build_video(images, audio_mp3: Path, captions_lines, out_mp4: Path):
    # 7 images -> 35 seconds
    durations = [5, 5, 5, 5, 5, 5, 5]

    concat_txt = OUTDIR / "images.txt"
    with concat_txt.open("w", encoding="utf-8") as f:
        for img, dur in zip(images, durations):
            f.write(f"file '{img.as_posix()}'\n")
            f.write(f"duration {dur}\n")
        # concat demuxer requires last file repeated
        f.write(f"file '{images[-1].as_posix()}'\n")

    # Captions: show line-by-line
    cap = captions_lines[:12] if captions_lines else ["Wat als alles anders liep?"]
    seg = max(2.5, 35.0 / len(cap))

    draw_filters = []
    t0 = 0.0
    for line in cap:
        t1 = t0 + seg
        safe = line.replace(":", r"\:").replace("'", r"\'")
        draw_filters.append(
            "drawtext="
            f"text='{safe}':"
            "x=(w-text_w)/2:"
            "y=h*0.78:"
            "fontsize=56:"
            "fontcolor=white:"
            "borderw=3:"
            f"enable='between(t,{t0:.2f},{t1:.2f})'"
        )
        t0 = t1

    vf = ",".join(
        [
            "scale=1080:1920:force_original_aspect_ratio=increase",
            "crop=1080:1920",
            "fps=30",
            *draw_filters,
        ]
    )

    tmp_video = OUTDIR / "tmp_video.mp4"

    run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_txt),
            "-vf",
            vf,
            "-pix_fmt",
            "yuv420p",
            "-r",
            "30",
            str(tmp_video),
        ]
    )

    # Merge audio
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(tmp_video),
            "-i",
            str(audio_mp3),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(out_mp4),
        ]
    )


# ----------------------------
# 5) YouTube upload
# ----------------------------
def youtube_service(client_id, client_secret, refresh_token):
    creds = Credentials(
        None,
        refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=client_id,
        client_secret=client_secret,
        scopes=["https://www.googleapis.com/auth/youtube.upload"],
    )
    return build("youtube", "v3", credentials=creds)


def upload_video(youtube, mp4_path: Path, title: str, description: str, tags):
    body = {
        "snippet": {
            "title": title[:95],
            "description": description,
            "tags": tags[:25] if isinstance(tags, list) else [],
            "categoryId": "24",
        },
        "status": {"privacyStatus": "public", "selfDeclaredMadeForKids": False},
    }

    media = MediaFileUpload(str(mp4_path), mimetype="video/mp4", resumable=True)
    req = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    resp = None
    while resp is None:
        status, resp = req.next_chunk()
        if status:
            print(f"Upload progress: {int(status.progress() * 100)}%")
    return resp


# ----------------------------
# Main
# ----------------------------
def main():
    state = load_state()
    day = int(state.get("day", 0))

    if day >= 30:
        print("✅ 30-day run complete. No further uploads.")
        return

    gemini_key = os.environ["GEMINI_API_KEY"]
    hf_token = os.environ["HF_TOKEN"]
    yt_client_id = os.environ["YT_CLIENT_ID"]
    yt_client_secret = os.environ["YT_CLIENT_SECRET"]
    yt_refresh = os.environ["YT_REFRESH_TOKEN"]

    story = gemini_generate_story(gemini_key)

    run_dir = OUTDIR / f"day_{day+1:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "story.json").write_text(json.dumps(story, indent=2, ensure_ascii=False), encoding="utf-8")

    # Images
    images = []
    for i, p in enumerate(story["image_prompts"][:7], start=1):
        out_img = run_dir / f"img_{i}.png"
        hf_generate_image(hf_token, p, out_img)
        images.append(out_img)

    # Voice
    audio_mp3 = run_dir / "voice.mp3"
    import asyncio

    asyncio.run(
        make_voice(
            story["script"],
            story["voice"].get("edge_tts_voice", "nl-NL-MaartenNeural"),
            audio_mp3,
            rate=story["voice"].get("speed", "+0%"),
            pitch=story["voice"].get("pitch", "+0Hz"),
        )
    )

    # Video
    out_mp4 = run_dir / "final.mp4"
    build_video(images, audio_mp3, story.get("on_screen_captions", []), out_mp4)

    # Upload
    yt = youtube_service(yt_client_id, yt_client_secret, yt_refresh)
    resp = upload_video(
        yt,
        out_mp4,
        story.get("title", f"Parallel Reality #{day+1}"),
        story.get("description", ""),
        story.get("tags", []),
    )
    video_id = resp.get("id")
    print("✅ Uploaded:", video_id)

    # Update state
    state["day"] = day + 1
    state.setdefault("uploads", []).append({"day": day + 1, "video_id": video_id, "title": story.get("title", "")})
    save_state(state)


if __name__ == "__main__":
    main()