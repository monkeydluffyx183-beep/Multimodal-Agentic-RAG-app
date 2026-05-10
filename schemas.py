"""Voice Router — POST /voice/transcribe | /voice/agent"""

from __future__ import annotations
import base64
import uuid
from fastapi import APIRouter

from app.models.schemas import (
    TranscribeRequest, TranscribeResponse,
    VoiceAgentRequest, VoiceAgentResponse, Message,
)
from app.services import memory_service
from app.services.provider import complete

router = APIRouter()


def _mock_transcribe(audio_b64: str, language: str) -> str:
    """
    Stub — replace with:
      - OpenAI Whisper API  (cloud)
      - faster-whisper      (local)
      - whisper.cpp         (local, C extension)
    """
    return "[Transcription placeholder — integrate Whisper here]"


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(req: TranscribeRequest):
    try:
        base64.b64decode(req.audio_b64, validate=True)
    except Exception:
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail="Invalid base64 audio data")
    transcript = _mock_transcribe(req.audio_b64, req.language)
    return TranscribeResponse(transcript=transcript, language=req.language)


@router.post("/agent", response_model=VoiceAgentResponse)
async def voice_agent(req: VoiceAgentRequest):
    session_id = req.session_id or str(uuid.uuid4())
    model = req.config.model or (
        "claude-sonnet-4-20250514" if req.config.provider == "claude" else "llama3.1:8b"
    )

    # Load history
    history = memory_service.get_as_messages(session_id)
    history.append(Message(role="user", content=req.transcript))

    system = (
        "You are a concise, friendly voice assistant. "
        "Keep replies short (1-3 sentences) since they will be spoken aloud."
    )
    reply, usage = await complete(history, req.config, system)

    # Persist
    memory_service.add_messages(session_id, [
        Message(role="user", content=req.transcript),
        Message(role="assistant", content=reply),
    ])

    return VoiceAgentResponse(
        provider=req.config.provider,
        model=model,
        session_id=session_id,
        reply=reply,
        tts_audio_b64=None,   # Integrate ElevenLabs / OpenAI TTS here
        usage=usage,
    )
