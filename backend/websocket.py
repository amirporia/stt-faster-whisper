import asyncio
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from backend.utils.ffmpeg_utils import start_ffmpeg_decoder
from model.whisper_asr import backend_factory
from backend.logging_config import logger
from time import time

BYTES_PER_SAMPLE = 1
BYTES_PER_SEC = 16000 * BYTES_PER_SAMPLE

async def handle_websocket(websocket: WebSocket, args):
    await websocket.accept()
    logger.info("WebSocket connection opened.")
    
     # Start FFmpeg decoder
    logger.info("Starting FFmpeg decoder...")
    ffmpeg_process = await start_ffmpeg_decoder()

    logger.info("FFmpeg decoder started!")

        # Initialize buffer and start processing loop
    pcm_buffer = bytearray()
    loop = asyncio.get_event_loop()
    asr, _ = backend_factory(args)

    try:
       
        if ffmpeg_process is None:
            logger.error("FFmpeg process failed to start.")
            return

        logger.info("Starting audio processing loop...")
        # Handle audio data from FFmpeg
        async def read_ffmpeg_stdout():

            nonlocal pcm_buffer
            transcribe = ""
            init_transcribe = ""
            beg = time()

            while True:

                logger.info("Inside audio processing loop...")

                try:
                    elapsed_time = int(time() - beg)
                    beg = time()
                    # Read audio chunk from FFmpeg process
                    chunk = await loop.run_in_executor(None, ffmpeg_process.stdout.read, BYTES_PER_SEC * elapsed_time)
                    if not chunk:
                        chunk = await loop.run_in_executor(
                            None, ffmpeg_process.stdout.read, 4096
                        )
                        if not chunk:
                            logger.warning("FFmpeg stdout closed.")
                            break

                    logger.info(f"Received chunk: {len(chunk)} bytes")
                    pcm_buffer.extend(chunk)

                    # Process PCM data if buffer has enough data
                    logger.info(f"Buffer size: {len(pcm_buffer)} bytes")
                    if len(pcm_buffer) >= BYTES_PER_SEC:
                        pcm_array = np.frombuffer(pcm_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                        pcm_buffer = bytearray()
                        logger.info(f"Buffer ready for transcription, size: {len(pcm_array)}")

                        # Transcribe the audio and send back a response
                        transcription = asr.transcribe(pcm_array, init_prompt=(" ".join(init_transcribe.split()[-4:]) if len(init_transcribe.split()) > 4 else init_transcribe))
                        tokenize_transcription = asr.ts_words(transcription)

                        init_transcribe = init_transcribe + " ".join([t for (a, b, t) in tokenize_transcription])


                        if len(tokenize_transcription) >= 2:
                            print(len(tokenize_transcription))
                            tokenize_transcription.pop()
                            print(len(tokenize_transcription))

                        transcribe = transcribe + " ".join([t for (a, b, t) in tokenize_transcription])

                        logger.info(f"Transcription result: {transcribe}")
                        response = {"lines": [{"speaker": "0", "text": transcribe}]}
                        await websocket.send_json(response)

                except Exception as e:
                    logger.error(f"Error processing audio: {e}")
                    break

        stdout_reader_task = asyncio.create_task(read_ffmpeg_stdout())

        # Main WebSocket loop to receive audio data and send to FFmpeg
        while True:
            try:
                logger.info("Waiting for WebSocket data...")
                message = await websocket.receive_bytes()
                logger.info(f"Received {len(message)} bytes from WebSocket.")
                ffmpeg_process.stdin.write(message)
                ffmpeg_process.stdin.flush()

            except WebSocketDisconnect:
                logger.info("WebSocket connection closed.")
                break
            except Exception as e:
                logger.error(f"Error in WebSocket loop: {e}")
                break

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    
    finally:
        ffmpeg_process.stdin.close()
        stdout_reader_task.cancel()
        await ffmpeg_process.wait()
        logger.info("FFmpeg process closed.")