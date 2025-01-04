import json
import logging
import os
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import List, Optional

import ffmpeg
import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from pydub import AudioSegment, silence
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

logger = logging.getLogger()


# Logging Setup


def setup_logging(log_file_path):
    log_format = "%(asctime)s - %(levelname)s - %(message)s"

    # Create root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File handler (rotates daily, keeps 14 backups)
    file_handler = TimedRotatingFileHandler(
        log_file_path, when="midnight", interval=1, backupCount=14
    )
    file_handler.setFormatter(logging.Formatter(log_format))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))

    # Ensure no duplicate handlers
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    # Suppress log propagation to prevent duplication
    logger.propagate = False

    # Apply same config to Uvicorn loggers
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.handlers = [file_handler, console_handler]
    uvicorn_logger.setLevel(logging.INFO)
    uvicorn_logger.propagate = False

    uvicorn_error_logger = logging.getLogger("uvicorn.error")
    uvicorn_error_logger.handlers = [file_handler, console_handler]
    uvicorn_error_logger.setLevel(logging.INFO)
    uvicorn_error_logger.propagate = False


setup_logging("process_log.txt")


try:
    log_file_path = os.path.abspath("process_log.txt")
    setup_logging(log_file_path)
    print("""
    
    
    ██╗    ██╗██╗  ██╗██╗███████╗██████╗ ███████╗██████╗     ████████╗██╗   ██╗██████╗ ██████╗  ██████╗      █████╗ ██████╗ ██╗
    ██║    ██║██║  ██║██║██╔════╝██╔══██╗██╔════╝██╔══██╗    ╚══██╔══╝██║   ██║██╔══██╗██╔══██╗██╔═══██╗    ██╔══██╗██╔══██╗██║
    ██║ █╗ ██║███████║██║███████╗██████╔╝█████╗  ██████╔╝       ██║   ██║   ██║██████╔╝██████╔╝██║   ██║    ███████║██████╔╝██║
    ██║███╗██║██╔══██║██║╚════██║██╔═══╝ ██╔══╝  ██╔══██╗       ██║   ██║   ██║██╔══██╗██╔══██╗██║   ██║    ██╔══██║██╔═══╝ ██║
    ╚███╔███╔╝██║  ██║██║███████║██║     ███████╗██║  ██║       ██║   ╚██████╔╝██║  ██║██████╔╝╚██████╔╝    ██║  ██║██║     ██║
     ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝╚══════╝╚═╝     ╚══════╝╚═╝  ╚═╝       ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚═════╝  ╚═════╝     ╚═╝  ╚═╝╚═╝     ╚═╝
                                                                                                                               
    
    
    Whisper Turbo API 0.1 - Can you hear me whisper? 

    """)
    logging.info(f"Logging setup completed. Log file path: {log_file_path}")
except Exception as e:
    print(f"Error setting up logging: {e}")

# Environment Variables
load_dotenv()
ADMIN_KEY = (
    os.environ.get("ADMIN_KEY") if os.environ.get("ADMIN_KEY") else None
)  # Optional
if ADMIN_KEY:
    logger.info(
        "Admin key found in environment variables, please use key for api requests"
    )
elif ADMIN_KEY is None:
    logger.info(
        "No Admin key found in environment variables, endpoints are available without access restrictions"
    )


ALLOWED_EXTENSIONS = {"mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"}

app = FastAPI(title="Whisper v3 Turbo API")

# Check for CUDA/MPS availability
device = (
    "cuda:0"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


# Check for Flash Attention 2 support
def is_flash_attention_available():
    try:
        import flash_attn

        return True
    except ImportError:
        return False


# Check if Flash Attention 2 is available
ATTN_IMPLEMENTATION = "flash_attention_2" if is_flash_attention_available() else "eager"

# Model setup
MODEL_ID = "openai/whisper-large-v3-turbo"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
    attn_implementation=ATTN_IMPLEMENTATION,  # Use FA2 if available, fallback to SDPA
)
model.to(device)

processor = AutoProcessor.from_pretrained(MODEL_ID)

# Initialize pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)


# Middleware for Optional Admin Token
@app.middleware("http")
async def admin_key_auth_check(request: Request, call_next):
    if ADMIN_KEY and request.headers.get("x-admin-api-key") != ADMIN_KEY:
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    return await call_next(request)


def transcribe_audio(
    file_path: str,
    model_name: str,
    language: Optional[str],
    task: str,
    batch_size: int,
    timestamp: str,
):
    """Performs transcription or translation using the Whisper model."""
    generate_kwargs = {"task": task, "language": language}
    ts = "word" if timestamp == "word" else True

    outputs = pipe(
        file_path,
        chunk_length_s=30,
        batch_size=batch_size,
        generate_kwargs=generate_kwargs,
        return_timestamps=ts,
    )
    return outputs


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form("openai/whisper-large-v3-turbo"),
    language: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    timestamp_granularities: Optional[List[str]] = Form(None),
    batch_size: Optional[int] = Form(24),
):
    """Transcribes an audio file"""
    ext = file.filename.split(".")[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file format")

    with open(f"temp_audio.{ext}", "wb") as temp_audio:
        temp_audio.write(await file.read())

    try:
        result = transcribe_audio(
            temp_audio.name, model, language, "transcribe", batch_size, "chunk"
        )
        Path(temp_audio.name).unlink(missing_ok=True)
        return result if response_format == "json" else result["text"]

    except Exception as e:
        Path(temp_audio.name).unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/audio/translations")
async def translate(
    file: UploadFile = File(...),
    model: str = Form("openai/whisper-large-v3-turbo"),
    response_format: Optional[str] = Form("json"),
    timestamp_granularities: Optional[List[str]] = Form(None),
    batch_size: Optional[int] = Form(24),
):
    """Translates an audio file to English"""
    ext = file.filename.split(".")[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file format")

    with open(f"temp_audio.{ext}", "wb") as temp_audio:
        temp_audio.write(await file.read())

    try:
        result = transcribe_audio(
            temp_audio.name, model, None, "translate", batch_size, "chunk"
        )
        Path(temp_audio.name).unlink(missing_ok=True)
        return result if response_format == "json" else result["text"]

    except Exception as e:
        Path(temp_audio.name).unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(e))


async def process_audio_chunk(audio_chunk):
    """Transcribe a single chunk and yield results word by word."""
    try:
        result = pipe(audio_chunk, return_timestamps="word")
        for word_data in result["chunks"]:
            word = word_data["text"]
            timestamp = word_data["timestamp"]
            yield f"data: {json.dumps({'word': word, 'timestamp': timestamp})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': f'Chunk processing failed: {str(e)}'})}\n\n"


async def stream_audio_chunks(file_path: str):
    """Splits audio based on silences and transcribes dynamically, ensuring temp file cleanup."""
    temp_files = []  # Track temporary files

    try:
        # Convert to WAV (16kHz, mono) for better processing
        converted_audio = "temp_audio_fixed.wav"
        ffmpeg.input(file_path).output(
            converted_audio, format="wav", acodec="pcm_s16le", ar="16000", ac="1"
        ).run(overwrite_output=True, quiet=True)
        temp_files.append(converted_audio)  # Track the converted file

        # Load audio with PyDub
        audio = AudioSegment.from_wav(converted_audio)

        # Detect silence-based chunks
        silence_thresh = -40  # dB threshold for silence
        min_silence_len = 300  # Silence length in ms to detect breaks
        chunks = silence.split_on_silence(
            audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh
        )

        if not chunks:
            yield f"data: {json.dumps({'error': 'No valid audio chunks detected'})}\n\n"
            return

        for i, chunk in enumerate(chunks):
            chunk_path = f"temp_chunk_{i}.wav"
            chunk.export(chunk_path, format="wav")
            temp_files.append(chunk_path)  # Track chunk files

            try:
                # Process the chunk asynchronously
                async for response in process_audio_chunk(chunk_path):
                    yield response
            finally:
                # Ensure chunk is deleted after use
                Path(chunk_path).unlink(missing_ok=True)

    except Exception as e:
        yield f"data: {json.dumps({'error': f'Critical error: {str(e)}'})}\n\n"
    finally:
        # Cleanup all temp files including converted audio
        for temp_file in temp_files:
            Path(temp_file).unlink(missing_ok=True)


@app.get("/v1/audio/stream_transcription")
async def stream_transcription(file_path: str):
    return StreamingResponse(
        stream_audio_chunks(file_path), media_type="text/event-stream"
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8383, log_config=None)