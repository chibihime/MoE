"""api/routes/stream.py — WS /predict/stream"""

import json
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends

from api.dependencies import get_pipeline
from inference.pipeline import EmotionPipeline

logger = logging.getLogger(__name__)
router = APIRouter(tags=["predict"])


@router.websocket("/stream")
async def predict_stream(
    websocket: WebSocket,
    pipe: EmotionPipeline = Depends(get_pipeline),
):
    """
    Real-time emotion prediction over WebSocket.

    Protocol:
        Client sends alternating frames:
          1. Binary frame  — raw Float32 PCM bytes at 16kHz (one 500ms chunk)
          2. Text frame    — optional JSON {"text": "..."} for concurrent ASR text
                             or simply send binary-only chunks

        Server responds after each binary frame with a JSON text frame:
          {
            "emotion":      "sad",
            "confidence":   0.51,
            "valence":      2.8,
            "arousal":      3.1,
            "all_probs":    {"happy": 0.05, "sad": 0.51, ...},
            "text_weight":  0.0,
            "audio_weight": 1.0,
            "chunk_index":  3
          }

        On error, server sends:
          {"error": "description", "chunk_index": N}

        Connection closes when client disconnects.

    Client-side Web Audio API setup:
        const ctx       = new AudioContext({ sampleRate: 16000 });
        const processor = ctx.createScriptProcessor(8000, 1, 1);  // 500ms chunks
        processor.onaudioprocess = (e) => {
            const pcm = e.inputBuffer.getChannelData(0);  // Float32Array
            ws.send(pcm.buffer);                          // send as binary
        };
        ws.onmessage = (e) => {
            const result = JSON.parse(e.data);
            console.log(result.emotion, result.confidence);
        };
    """
    await websocket.accept()
    chunk_index = 0
    pending_text: str | None = None

    logger.info(f"WebSocket connection opened: {websocket.client}")

    try:
        while True:
            # Receive the next frame — either binary (PCM) or text (JSON metadata)
            message = await websocket.receive()

            if "bytes" in message and message["bytes"]:
                # Binary frame: PCM audio chunk
                pcm_bytes = message["bytes"]

                try:
                    result = pipe.predict_stream_chunk(
                        pcm_bytes=pcm_bytes,
                        text=pending_text,
                    )
                    pending_text = None   # consume the queued text

                    await websocket.send_text(json.dumps({
                        "emotion":      result.emotion,
                        "confidence":   result.confidence,
                        "valence":      result.valence,
                        "arousal":      result.arousal,
                        "all_probs":    result.all_probs,
                        "text_weight":  result.text_weight,
                        "audio_weight": result.audio_weight,
                        "chunk_index":  chunk_index,
                    }))

                except ValueError as e:
                    await websocket.send_text(json.dumps({
                        "error":       str(e),
                        "chunk_index": chunk_index,
                    }))
                except Exception as e:
                    logger.exception("Stream chunk prediction failed")
                    await websocket.send_text(json.dumps({
                        "error":       "Prediction failed.",
                        "chunk_index": chunk_index,
                    }))

                chunk_index += 1

            elif "text" in message and message["text"]:
                # Text frame: optional JSON with concurrent ASR text
                try:
                    data = json.loads(message["text"])
                    pending_text = data.get("text") or None
                except json.JSONDecodeError:
                    # If plain text (not JSON), use it directly
                    pending_text = message["text"].strip() or None

    except WebSocketDisconnect:
        logger.info(
            f"WebSocket disconnected: {websocket.client} "
            f"after {chunk_index} chunks"
        )
    except Exception as e:
        logger.exception("WebSocket error")
        try:
            await websocket.send_text(json.dumps({"error": "Server error."}))
            await websocket.close(code=1011)
        except Exception:
            pass