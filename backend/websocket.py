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

def model_transcribe(pcm_array, trans):

    transcription = asr.transcribe(pcm_array, init_prompt=trans)
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
                    # logger.info("Inside audio processing loop...")
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
                        chunk = await loop.run_in_executor(None, ffmpeg_process.stdout.read, 4096)
                        if not chunk:
                            logger.warning("FFmpeg stdout closed.")
                            break

                    # Add to buffer from ffmpeg
                    pcm_buffer.extend(chunk)

                    if len(pcm_buffer) >= BYTES_PER_SEC:

                        # Convert audio buffer to numpy 
                        if len(pcm_buffer) % 2 != 0:
                            pcm_array = np.frombuffer(pcm_buffer[:-1], dtype=np.int16).astype(np.float32) / 32768.0
                        else:
                            pcm_array = np.frombuffer(pcm_buffer, dtype=np.int16).astype(np.float32) / 32768.0

                        # Transcribe the audio and send back a response
                        tokenize_transcription = model_transcribe(pcm_array, " ".join(confirmed_transciption))
                        if len(tokenize_transcription) > 0:
                            offset_ts = tokenize_transcription[0][0]
                            tokenize_transcription = [(a-offset_ts, b-offset_ts, t) for a,b,t in tokenize_transcription]
                        print(f"11111111111111111: {tokenize_transcription}")
                        print(f"22222222222222222: {" ".join([" ".join(w.split()) for a,b,w in tokenize_transcription])}")
                        print(f"33333333333333333: {" ".join(transcribe)}")
                        print(f"44444444444444444: {" ".join(confirmed_transciption)}")
                        # Confirmed the transcribe by reviewing two times
                        transcribe, confirmed_transciption = confirmation_process(transcribe, tokenize_transcription, confirmed_transciption)
                        print(f"55555555555555555: {" ".join(transcribe)}")
                        print(f"66666666666666666: {" ".join(confirmed_transciption)}")
                    
  
                        if int(len(pcm_buffer)/(SAMPLE_RATE * BYTES_PER_SAMPLE)) >= 30 and len(tokenize_transcription) > 0:
                            # Trimming buffer when reach to 30s
                            pcm_buffer, confirmed_transciption, transcribe = threshold_trim_buffer(tokenize_transcription=tokenize_transcription, confirmed_transcription=confirmed_transciption, non_confirmed_transcription=transcribe, buffer=pcm_buffer, sample_rate=SAMPLE_RATE, bytes_per_sample=BYTES_PER_SAMPLE)
                        else:
                            # Trimming buffer when reach to end of sentence
                            pcm_buffer_idx, transcribe = sentence_trim_buffer(tokenize_transcription=tokenize_transcription, confirmed_transcription=confirmed_transciption, non_confirmed_transcription=transcribe, sample_rate=SAMPLE_RATE, bytes_per_sample=BYTES_PER_SAMPLE)

                            if pcm_buffer_idx == -1:
                                pcm_buffer.clear()
                            elif pcm_buffer_idx != 0:
                                pcm_buffer = bytearray(pcm_buffer[pcm_buffer_idx:])
                        print(f"777777777777777777: {pcm_buffer_idx}")        
                        response = {"lines": [{"speaker": "0", "text": " ".join(confirmed_transciption)}]}
                        await websocket.send_json(response)

                except Exception as e:
                    logger.error(f"Error processing audio: {e}")
                    break

        reader_task = asyncio.create_task(read_ffmpeg_stdout())

        # Main WebSocket loop to receive audio data and send to FFmpeg
        try:
            while True:
                # Receive incoming WebM audio chunks from the client
                message = await websocket.receive_bytes()
                if ffmpeg_process.stdin:
                    ffmpeg_process.stdin.write(message)
                    # logger.info(f"Writing to FFmpeg stdin... ******************** message length = {len(message)}")
                    await asyncio.sleep(0.01)  # Small delay to prevent overload
                    ffmpeg_process.stdin.flush()
                
        except WebSocketDisconnect:
            print("WebSocket connection closed.")
        except Exception as e:
            print(f"Error in websocket loop: {e}")
        finally:
            # Clean up ffmpeg and the reader task
            await websocket.close()
            reader_task.cancel()
            if ffmpeg_process.stdin:
                ffmpeg_process.stdin.close()
            if ffmpeg_process.stdout:
                ffmpeg_process.stdout.close()
            ffmpeg_process.wait()
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")