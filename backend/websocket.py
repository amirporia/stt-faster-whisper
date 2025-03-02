import asyncio
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from backend.utils.ffmpeg_utils import start_ffmpeg_decoder
from model.whisper_asr import backend_factory
from backend.logging_config import logger
from config.settings import settings
from time import time
from backend.utils.methods import confirmation_process, sentence_trim_buffer, threshold_trim_buffer

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
BYTES_PER_SEC = SAMPLE_RATE * BYTES_PER_SAMPLE

asr, _ = backend_factory(settings)

def model_transcribe(pcm_array):

    transcription = asr.transcribe(pcm_array)
    tokenize_transcription = asr.ts_words(transcription)
    return tokenize_transcription

async def handle_websocket(websocket: WebSocket):

    await websocket.accept()    
    ffmpeg_process = await start_ffmpeg_decoder()
    pcm_buffer = bytearray()  

    try:
       
        if ffmpeg_process is None:
            logger.error("FFmpeg process failed to start.")
            return
        
        if ffmpeg_process.poll() is not None:  
            logger.error("FFmpeg process exited unexpectedly.1")

        logger.info("Starting audio processing loop...")

        async def read_ffmpeg_stdout():

            loop = asyncio.get_event_loop()
            nonlocal pcm_buffer
            transcribe = []
            confirmed_transciption = []
            beg = time()

            while True:
                try:
                    logger.info("Inside audio processing loop...")
                    elapsed_time = int(time() - beg)
                    beg = time()

                    if ffmpeg_process is None:
                        logger.error("FFmpeg process failed to start.")
                        return
                    
                    if ffmpeg_process.poll() is not None:  
                        logger.error("FFmpeg process exited unexpectedly.2")

                    # Read audio chunk from FFmpeg process
                    chunk = await loop.run_in_executor(None, ffmpeg_process.stdout.read, 32000 * max(1, elapsed_time))
                    if not chunk:
                        chunk = await loop.run_in_executor(
                            None, ffmpeg_process.stdout.read, 4096
                        )
                        if not chunk:
                            logger.warning("FFmpeg stdout closed.")
                            break

                    # Add to buffer from ffmpeg
                    pcm_buffer.extend(chunk)

                    if len(pcm_buffer) >= BYTES_PER_SEC:
                        # Convert audio buffer to numpy 
                        pcm_array = np.frombuffer(pcm_buffer, dtype=np.int16).astype(np.float32) / 32768.0

                        # Transcribe the audio and send back a response
                        tokenize_transcription = model_transcribe(pcm_array)

                        # Confirmed the transcribe by reviewing two times
                        transcribe, confirmed_transciption = confirmation_process(transcribe, tokenize_transcription, confirmed_transciption)

                        # Trimming buffer when reach to end of sentence
                        pcm_buffer_idx, transcribe = sentence_trim_buffer(tokenize_transcription=tokenize_transcription, confirmed_transcription=confirmed_transciption, non_confirmed_transcription=transcribe, sample_rate=SAMPLE_RATE, bytes_per_sample=BYTES_PER_SAMPLE)

                        if pcm_buffer_idx == -1:
                            pcm_buffer = bytearray()
                        else:
                            pcm_buffer = pcm_buffer[pcm_buffer_idx:]
                        
                        # Trimming buffer when reach to 30s
                        pcm_buffer, confirmed_transciption, transcribe = threshold_trim_buffer(tokenize_transcription=tokenize_transcription, confirmed_transcription=confirmed_transciption, non_confirmed_transcription=transcribe, buffer=pcm_buffer, sample_rate=SAMPLE_RATE, bytes_per_sample=BYTES_PER_SAMPLE)

                        response = {"lines": [{"speaker": "0", "text": " ".join(confirmed_transciption)}]}
                        await websocket.send_json(response)

                except Exception as e:
                    logger.error(f"Error processing audio: {e}")
                    break

        stdout_reader_task = asyncio.create_task(read_ffmpeg_stdout())

        # Main WebSocket loop to receive audio data and send to FFmpeg
        try:
            while True:
                try:
                    # Receive incoming WebM audio chunks from the client
                    message = await websocket.receive_bytes()
                    if ffmpeg_process.stdin:
                        ffmpeg_process.stdin.write(message)
                        logger.info(f"Writing to FFmpeg stdin... ******************** message length = {len(message)}")
                        await asyncio.sleep(0.01)  # Small delay to prevent overload
                        ffmpeg_process.stdin.flush()
                except Exception as e:
                    logger.error(f"Unexpected error: {e}")
                
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
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")