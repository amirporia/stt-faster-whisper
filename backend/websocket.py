import asyncio
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from backend.utils.ffmpeg_utils import start_ffmpeg_decoder
from model.whisper_asr import backend_factory
from backend.logging_config import logger
from config.settings import settings
from time import time

BYTES_PER_SAMPLE = 1
BYTES_PER_SEC = 16000 * BYTES_PER_SAMPLE
asr, _ = backend_factory(settings)

async def handle_websocket(websocket: WebSocket):

    await websocket.accept()    
    ffmpeg_process = await start_ffmpeg_decoder()

    # Initialize buffer and start processing loop
    pcm_buffer = bytearray()  

    try:
       
        if ffmpeg_process is None:
            logger.error("FFmpeg process failed to start.")
            return

        logger.info("Starting audio processing loop...")
        async def read_ffmpeg_stdout():
            loop = asyncio.get_event_loop()
            nonlocal pcm_buffer
            transcribe = ""
            beg = time()

            while True:

                logger.info("Inside audio processing loop...")

                try:
                    elapsed_time = int(time() - beg)
                    beg = time()
                    # Read audio chunk from FFmpeg process
                    chunk = await loop.run_in_executor(None, ffmpeg_process.stdout.read, BYTES_PER_SEC * 2 * elapsed_time)
                    if not chunk:
                        chunk = await loop.run_in_executor(
                            None, ffmpeg_process.stdout.read, 4096
                        )
                        if not chunk:
                            logger.warning("FFmpeg stdout closed.")
                            break

                    pcm_buffer.extend(chunk)
                    logger.info(f"Buffer size: {len(pcm_buffer)} bytes")

                    if len(pcm_buffer) >= BYTES_PER_SEC:
                        pcm_array = np.frombuffer(pcm_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                        pcm_buffer = bytearray()
                        logger.info(f"Buffer ready for transcription, size: {len(pcm_array)}")

                        # Transcribe the audio and send back a response
                        transcription = asr.transcribe(pcm_array)
                        tokenize_transcription = asr.ts_words(transcription)
                        print(tokenize_transcription)
                        if len(tokenize_transcription) >= 2:
                            tokenize_transcription.pop()

                        transcribe = transcribe + " ".join([t for (a, b, t) in tokenize_transcription])

                        logger.info(f"Transcription result: {transcribe}")
                        response = {"lines": [{"speaker": "0", "text": transcribe}]}
                        await websocket.send_json(response)

                except Exception as e:
                    logger.error(f"Error processing audio: {e}")
                    break

        stdout_reader_task = asyncio.create_task(read_ffmpeg_stdout())

        # Main WebSocket loop to receive audio data and send to FFmpeg
        try:
            while True:
                # Receive incoming WebM audio chunks from the client
                message = await websocket.receive_bytes()
                # Pass them to ffmpeg via stdin
                ffmpeg_process.stdin.write(message)
                ffmpeg_process.stdin.flush()

        except WebSocketDisconnect:
            print("WebSocket connection closed.")
        except Exception as e:
            print(f"Error in websocket loop: {e}")
        finally:
            # Clean up ffmpeg and the reader task
            try:
                ffmpeg_process.stdin.close()
            except:
                pass
            stdout_reader_task.cancel()

            try:
                ffmpeg_process.stdout.close()
            except:
                pass

            ffmpeg_process.wait()
            del online
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")