from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import io
import numpy as np
import librosa
import os
import shutil
import cv2
import json
from dotenv import load_dotenv
from openai import OpenAI
import base64
import re
from fastapi import Body

# =========================
# LOAD ENV + GROQ
# =========================
load_dotenv()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
nim_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

# =========================
# CREATE APP
# =========================
app = FastAPI(title="Fake Content Detection API (Groq Powered)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# GLOBAL SCAN COUNTER
# =========================
scan_count = 0

# =========================
# REQUEST MODEL
# =========================
class TextRequest(BaseModel):
    text: str

# =========================
# HOME + STATUS
# =========================
@app.get("/")
def home():
    return {"message": "Groq Fake Detection API Running"}

@app.get("/status")
def status_check():
    return {"status": "ok"}

# -------------------------
# TEXT DETECTION (GROQ)
# -------------------------
@app.post("/detect-text")
def detect_text(request: TextRequest):
    global scan_count
    scan_count += 1
    try:
        prompt = f"""
Detect if this text is REAL or FAKE or SCAM.
Return ONLY JSON:
{{"result":"FAKE","confidence":"90%"}}

Text:
{request.text}
"""

        completion = nim_client.chat.completions.create(
            model="meta/llama-3.1-8b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        raw = completion.choices[0].message.content

        import re, json
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())

        return {"result": "ERROR", "confidence": "0%"}

    except Exception as e:
        print("NVIDIA TEXT ERROR:", e)
        return {"result": "ERROR", "confidence": "0%"}


# -------------------------
# URL DETECTION (NVIDIA)
# -------------------------
@app.post("/detect-url")
def detect_url(data: dict = Body(...)):
    global scan_count
    scan_count += 1
    try:
        url = data.get("url")

        prompt = f"""
You are a cybersecurity system.

Analyze this URL and classify it as:
SAFE / PHISHING / SCAM / SUSPICIOUS

Check for:
- fake login pages
- bank phishing
- lottery or prize scams
- shortened or masked links
- suspicious domains
- misleading URLs

Respond ONLY in JSON:
{{"result":"SAFE","confidence":"90%"}}

URL:
{url}
"""

        completion = nim_client.chat.completions.create(
            model="meta/llama-3.1-8b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200
        )

        raw = completion.choices[0].message.content
        print("NVIDIA URL RAW:", raw)

        import re, json

        raw = raw.replace("```json", "").replace("```", "").strip()

        match = re.search(r"\{.*\}", raw, re.DOTALL)

        if match:
            try:
                parsed = json.loads(match.group())

                # 🔥 IMPORTANT: send only these 2 fields to Flutter
                return {
                    "result": parsed.get("result", "SAFE"),
                    "confidence": parsed.get("confidence", "75%")
                }

            except Exception as e:
                print("URL JSON PARSE ERROR:", e)

        # fallback
        lower = raw.lower()

        if any(w in lower for w in ["phishing", "scam", "malicious", "fake"]):
            return {"result": "PHISHING", "confidence": "85%"}

        return {"result": "SAFE", "confidence": "75%"}


    except Exception as e:
        print("NVIDIA URL ERROR:", e)
    return {"result": "ERROR", "confidence": "0%"}

# -------------------------
# IMAGE DETECTION (NVIDIA)
# -------------------------
@app.post("/detect-image")
async def detect_image(file: UploadFile = File(...)):
    global scan_count
    scan_count += 1
    try:
        content = await file.read()
        image_base64 = base64.b64encode(content).decode("utf-8")

        completion = nim_client.chat.completions.create(
            model="meta/llama-3.2-11b-vision-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
"type": "text",
"text": """
You are an AI image forensic detector.

Your job is to detect AI‑generated images, deepfakes, and manipulated visuals.

Mark image as FAKE if ANY of these are present:
- AI generated artwork or portraits
- Unreal skin texture or symmetry
- Distorted fingers, eyes, teeth
- Over‑smooth lighting
- Text artifacts
- Inconsistent shadows
- Background blending errors
- Synthetic or diffusion‑style visuals

Mark REAL only if it is a natural camera photograph.

Respond ONLY in JSON:
{"result":"FAKE","confidence":"0-100%"}
"""
},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=200,
            temperature=0
        )

        raw = completion.choices[0].message.content
        print("NVIDIA IMAGE RAW:", raw)

        # 🔧 Fix invalid % formatting automatically
        raw = re.sub(r'confidence":\s*(\d+)%', r'confidence":"\1%"', raw)

        # Extract JSON safely
        match = re.search(r"\{.*\}", raw, re.DOTALL)

        if match:
            try:
                return json.loads(match.group())
            except Exception as e:
                print("JSON PARSE ERROR:", e)

                # 🔥 ADD THIS HERE (AI‑image detection fallback)
                lower_raw = raw.lower()

                if any(word in lower_raw for word in [
                    "ai generated",
                    "synthetic",
                    "diffusion",
                    "unreal",
                    "render",
                    "not a real photo",
                    "computer generated"
                ]):
                    return {"result": "FAKE", "confidence": "90%"}

                # Final fallback
            return {"result": "REAL", "confidence": "75%"}
        # 🧠 Fallback logic if JSON still invalid
        lower_raw = raw.lower()

        if any(word in lower_raw for word in ["scam", "fake", "manipulated", "deepfake", "ai generated"]):
            return {"result": "FAKE", "confidence": "85%"}

        return {"result": "REAL", "confidence": "75%"}

    except Exception as e:
        print("NVIDIA IMAGE ERROR:", e)
        return {"result": "ERROR", "confidence": "0%"}




# -------------------------
# AUDIO DETECTION (NVIDIA)
# -------------------------
@app.post("/detect-audio")
async def detect_audio(file: UploadFile = File(...)):
    global scan_count
    scan_count += 1 
    try:
        audio_bytes = await file.read()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        completion = nim_client.chat.completions.create(
            model="meta/llama-3.1-8b-instruct",
            messages=[
                {
                    "role": "user",
                    "content": f"""
You are an audio forensic AI.

If audio is:
- AI generated
- voice cloned
- synthetic
- scam call
- impersonation

Result must be FAKE.
If natural human speech → REAL.

Reply ONLY JSON:
{{"result":"FAKE or REAL","confidence":"0-100%"}}
Do not explain.
"""

                },
            ],
            temperature=0,
            max_tokens=200
        )

        raw = completion.choices[0].message.content
        print("AUDIO RAW:", raw)

        import re, json
        match = re.search(r"\{.*\}", raw, re.DOTALL)

        if match:
            return json.loads(match.group())

        lower_raw = raw.lower()

        ai_audio_keywords = [
            "ai generated",
            "synthetic",
            "text to speech",
            "tts",
            "computer generated",
            "voice cloning",
            "deepfake voice",
            "digitally altered",
            "unnatural pitch",
            "robotic",
            "uniform tone",
            "lack of breathing",
            "processed audio"
        ]

        if any(word in lower_raw for word in ai_audio_keywords):
            return {"result": "FAKE", "confidence": "90%"}
        
        scam_keywords = [
            "otp",
            "bank account",
            "urgent payment",
            "verify account",
            "credit card",
            "lottery",
            "prize",
            "click link",
            "send money"
        ]

        # AI voice signals
        if any(word in lower_raw for word in ai_audio_keywords):
            return {"result": "FAKE", "confidence": "90%"}

        # scam content signals
        if any(word in lower_raw for word in scam_keywords):
            return {"result": "FAKE", "confidence": "88%"}

        # if model unsure
        return {"result": "UNCERTAIN", "confidence": "60%"}


    except Exception as e:
        print("AUDIO ERROR:", e)
        return {"result": "ERROR", "confidence": "0%"}

# -------------------------
# VIDEO DETECTION (NVIDIA)
# -------------------------
@app.post("/detect-video")
async def detect_video(file: UploadFile = File(...)):
    global scan_count
    scan_count += 1
    try:
        # save temp video
        video_bytes = await file.read()
        with open("temp_video.mp4", "wb") as f:
            f.write(video_bytes)

        cap = cv2.VideoCapture("temp_video.mp4")

        frames_checked = 0
        fake_hits = 0

        while cap.isOpened() and frames_checked < 8:
            ret, frame = cap.read()
            if not ret:
                break

            _, buffer = cv2.imencode(".jpg", frame)
            image_base64 = base64.b64encode(buffer).decode("utf-8")

            completion = nim_client.chat.completions.create(
                model="meta/llama-3.2-11b-vision-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """
You are a digital forensic AI.

If the video frame is:
- AI generated
- deepfake
- face swapped
- synthetic
- edited
- manipulated

Then result MUST be FAKE.

If it is natural camera footage, result must be REAL.

Reply ONLY JSON:
{"result":"FAKE or REAL","confidence":"0-100%"}
Do not explain.
"""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0,
                max_tokens=150
            )

            raw = completion.choices[0].message.content.lower()

            # AI / synthetic keywords
            ai_keywords = [
                "ai generated",
                "synthetic",
                "deepfake",
                "computer generated",
                "rendered",
                "diffusion",
                "not a real video",
                "unnatural",
                "inconsistent lighting",
                "face manipulation",
                "fake"
            ]

            if any(word in raw for word in ai_keywords):
                fake_hits += 1


            frames_checked += 1

        cap.release()

        if fake_hits >= 2:
            return {"result": "FAKE", "confidence": "88%"}
        else:
            return {"result": "REAL", "confidence": "78%"}

    except Exception as e:
        print("VIDEO ERROR:", e)
        return {"result": "ERROR", "confidence": "0%"}

@app.get("/scan-count")
def get_scan_count():
    global scan_count
    return {"scans": scan_count}


