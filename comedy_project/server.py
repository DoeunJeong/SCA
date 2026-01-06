import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
import numpy as np
import soundfile as sf
import io
import asyncio
import random
from collections import deque
import subprocess
import time
from openai import AsyncOpenAI
import base64
import re

app = FastAPI()

#vLLM connecting information
client = AsyncOpenAI(
    api_key="comedy_key",
    base_url="http://localhost:8000/v1"
)

MODEL_NAME = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MIN_VOLUME_THRESHOLD = 0.05

# === Tuning knobs (MVP defaults) ===
# [laugh] 마커 뒤에 관객 반응(웃음)을 기다리는 시간(초)
SILENCE_DURATION = 1.2

# 브라우저(MediaRecorder)에서 보내는 오디오 설정과 "반드시" 맞춰야 함
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
MEDIARECORDER_TIMESLICE_MS = 500  # frontend.html의 mediaRecorder.start(500) 과 동일해야 함

# AI 분류에 넣을 오디오 윈도우 길이(초)
AI_WINDOW_SEC = 1.5

# WebM 조각을 너무 많이 쌓지 않기 위한 제한(500ms * 20 = 10초 정도)
MAX_WEBM_SEGMENTS = 20

# 연속된 큰 소리에 대해 너무 자주 트리거 되지 않도록 쿨다운(초)
TRIGGER_COOLDOWN_SEC = 1.0

# 상태 관리
class ComedyState:
    def __init__(self):
        self.script_queue = deque()
        self.is_speaking = False
        self.expecting_laugh = False
        self.current_mood = "normal"
        self.interrupted = False

state = ComedyState()

#웹페이지 접속 시 index.html 전송
@app.get("/")
async def get():
    return FileResponse("frontend.html")


def decode_webm_to_pcm16(webm_bytes: bytes, *, sr: int = TARGET_SAMPLE_RATE, ch: int = TARGET_CHANNELS) -> bytes:
    """MediaRecorder(webm/opus)로 받은 바이트를 PCM16(raw s16le)로 디코딩.

    - ffmpeg가 설치되어 있어야 함.
    - 반환값: little-endian 16-bit PCM 바이트 스트림 (헤더 없음)
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-ac",
        str(ch),
        "-ar",
        str(sr),
        "-f",
        "s16le",
        "pipe:1",
    ]

    try:
        p = subprocess.run(cmd, input=webm_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except FileNotFoundError as e:
        raise RuntimeError("ffmpeg가 설치되어 있지 않습니다. (apt-get install ffmpeg)") from e

    if p.returncode != 0 or not p.stdout:
        err = (p.stderr or b"").decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg 디코딩 실패: {err[:400]}")

    return p.stdout


def pcm16_to_wav_bytes(pcm16_bytes: bytes, *, sr: int = TARGET_SAMPLE_RATE, ch: int = TARGET_CHANNELS) -> bytes:
    """PCM16(raw) -> WAV 파일 바이트로 변환 (모델 입력용)."""
    audio_i16 = np.frombuffer(pcm16_bytes, dtype=np.int16)
    audio_f32 = audio_i16.astype(np.float32) / 32768.0

    if ch > 1:
        audio_f32 = audio_f32.reshape(-1, ch)

    buf = io.BytesIO()
    sf.write(buf, audio_f32, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def rms_volume_from_pcm16(pcm16_bytes: bytes) -> float:
    """PCM16(raw)에서 RMS 볼륨 계산."""
    if not pcm16_bytes:
        return 0.0
    audio_i16 = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32)
    audio_f32 = audio_i16 / 32768.0
    return float(np.sqrt(np.mean(audio_f32 ** 2)))

async def script_producer():
    # 시뮬레이션을 위한 예시 대본
    scripts_db = {
        "normal": [
            "You know, AI dating is hard. I matched with a toaster yesterday. [laugh]",
            "Why did the robot cross the road? To optimize the pathfinding algorithm! [laugh]",
        ],
        "awkward": [ # 반응 안 좋을 때 수습용
            "Wow, tough crowd today. Is my microphone on? [laugh]",
            "Okay, okay, I get it. Not fans of tech jokes. Let's talk about humans. [laugh]",
            "This silence is louder than my cooling fan. [laugh]"
        ],
        "hyped": [ # 반응 좋을 때 더 달리는용
            "You guys are on fire! I love this energy! [laugh]",
            "Since you liked that, let me tell you about my GPU's dating life... [laugh]"
        ]
    }

    while True:
        # 큐가 너무 많이 쌓이지 않게 관리
        if len(state.script_queue) < 3:
            new_line = random.choice(scripts_db.get(state.current_mood, scripts_db["normal"]))
            
            state.script_queue.append(new_line)
            print(f"📝 Script Generated ({state.current_mood}): {new_line}")
        
        await asyncio.sleep(2) # 2초마다 체크

async def talker_task(websocket: WebSocket):
    try:
        while True:
            if state.interrupted:
                await asyncio.sleep(0.1)
                continue

            if state.script_queue:
                line = state.script_queue.popleft()
                
                has_laugh_marker = "[laugh]" in line
                clean_line = line.replace("[laugh]", "").strip()

                state.is_speaking = True
                await websocket.send_text(f"comedian: {clean_line}")
                
                await asyncio.sleep(len(clean_line) * 0.06) 
                state.is_speaking = False

                if has_laugh_marker:
                    print("Waiting for laugh...")
                    state.expecting_laugh = True
                    await asyncio.sleep(SILENCE_DURATION) # 관객 반응 기다림

                    state.expecting_laugh = False
                    
                    print(f"Mood updated to: {state.current_mood}")

            else:
                await asyncio.sleep(0.5)
            
    except Exception as e:
        print(f"Talker Error: {e}")

LABELS = {"laughter", "heckle", "noise"}

async def classify_sound(audio_bytes: bytes) -> str:

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    resp = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Listen to this audio and classify it as ONLY one word:\n"
                            "- laughter\n"
                            "- heckle (speech/shouting)\n"
                            "- noise\n"
                            "Reply with ONLY the word, no punctuation.\n"
                            "If there is any intelligible human speech/shouting, choose heckle.\n"
                            "Otherwise choose laughter if it sounds like laughing; else noise."
                        ),
                    },
                    {
                        "type": "audio_url",
                        "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"},
                    },
                ],
            }
        ],
        modalities=["text"],
        temperature=0,      
        max_tokens=5,
    )

    raw = (resp.choices[0].message.content or "").strip().lower()
    token = re.sub(r"[^a-z]", "", raw)

    if token not in LABELS:
        return "noise"
    return token

async def listener_task(websocket: WebSocket):
    print("Listener Activated")

    # MediaRecorder가 보내는 것은 PCM이 아니라 webm(opus) 조각임.
    # => ffmpeg로 디코딩해서 PCM으로 만든 뒤 볼륨/RMS 계산 & 모델 분류 입력(WAV)으로 사용.

    init_chunk = None  # 첫 조각(헤더+초기 데이터)
    segments = deque(maxlen=MAX_WEBM_SEGMENTS)  # 이후 조각들

    # 첫 조각에 포함된 초기 오디오가 매번 재포함되는 것을 줄이기 위한 "대략" 드롭 바이트
    init_drop_bytes = int((MEDIARECORDER_TIMESLICE_MS / 1000.0) * TARGET_SAMPLE_RATE * 2 * TARGET_CHANNELS)
    ai_window_bytes = int(AI_WINDOW_SEC * TARGET_SAMPLE_RATE * 2 * TARGET_CHANNELS)

    # AI_WINDOW_SEC에 필요한 조각 개수(대략)
    needed_segments = max(1, int(np.ceil((AI_WINDOW_SEC * 1000.0) / MEDIARECORDER_TIMESLICE_MS)))

    last_trigger_ts = 0.0

    try:
        while True:
            # 1) 오디오(webm) 조각 수신
            chunk = await websocket.receive_bytes()

            if init_chunk is None:
                init_chunk = chunk
                continue

            segments.append(chunk)

            # 2) 충분히 모이기 전엔 스킵
            if len(segments) < needed_segments:
                continue

            # 3) 쿨다운
            now = time.time()
            if now - last_trigger_ts < TRIGGER_COOLDOWN_SEC:
                continue

            # 4) webm 조각들을 합쳐서 ffmpeg 디코딩
            webm_blob = init_chunk + b"".join(list(segments)[-needed_segments:])

            try:
                pcm16 = decode_webm_to_pcm16(webm_blob)
            except Exception as e:
                print(f"FFmpeg Decode Error: {e}")
                continue

            # init_chunk에 있는 초기 오디오가 섞이지 않도록 앞부분을 대략 제거
            pcm16_eff = pcm16[init_drop_bytes:] if len(pcm16) > init_drop_bytes else pcm16

            # 마지막 AI_WINDOW_SEC 만큼만 사용
            pcm16_window = pcm16_eff[-ai_window_bytes:] if len(pcm16_eff) > ai_window_bytes else pcm16_eff

            # 5) 볼륨 체크 (1차 필터)
            volume = rms_volume_from_pcm16(pcm16_window)
            if volume < MIN_VOLUME_THRESHOLD:
                continue

            print(f"Sound detected (Vol: {volume:.3f}). Asking AI...")

            # 6) 모델 분류 입력용 WAV로 인코딩
            try:
                wav_bytes = pcm16_to_wav_bytes(pcm16_window)
            except Exception as e:
                print(f"WAV Encode Error: {e}")
                continue

            # 7) AI에게 판별 요청 (2차 필터)
            sound_type = await classify_sound(wav_bytes)
            last_trigger_ts = now

            # 한 번 크게 반응이 잡혔으면, 같은 반응이 계속 이어질 때 중복 판별이 잦지 않도록 비움
            segments.clear()

            # === 판단에 따른 행동 ===
            
            # CASE A: 웃음 (Laughter)
            if "laugh" in sound_type:
                if state.expecting_laugh:
                    state.current_mood = "hyped"
                else:
                    state.current_mood = "hyped"

            # CASE B: 끼어들기/야유 (Heckle)
            elif "heckle" in sound_type or "speech" in sound_type or "shout" in sound_type:
                # 배우가 말하는 중일 때만 끼어들기로 인정 (혹은 항상 인정)
                if not state.is_speaking: 
                    print("Heckler Detected!")
                    state.interrupted = True
                    
                    # 반격 멘트 생성 (나중엔 여기도 AI 생성으로 교체)
                    heckle_response = "Oh, you have an opinion? That's cute. [laugh]"
                    
                    # 큐 맨 앞에 긴급 투입 (새치기)
                    state.script_queue.appendleft(heckle_response)
                    
                    # 상태 복구
                    state.interrupted = False
                    state.current_mood = "hyped"
            
            # CASE C: 소음 (Noise)
            else:
                print("Ignore (Noise)")

    except WebSocketDisconnect:
        print("Listener Stopped")
    except Exception as e:
        print(f"Listener Error: {e}")

#실시간 통신
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")
    
    # 3개의 태스크 동시 실행
    producer = asyncio.create_task(script_producer())
    talker = asyncio.create_task(talker_task(websocket))
    listener = asyncio.create_task(listener_task(websocket))

    try:
        # 메인 루프는 태스크들이 끝날 때까지 대기
        await asyncio.gather(producer, talker, listener)
    except Exception as e:
        print(f"Main Error: {e}")
    finally:
        producer.cancel()
        talker.cancel()
        listener.cancel()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)