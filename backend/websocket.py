import asyncio
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from backend.utils.ffmpeg_utils import start_ffmpeg_decoder
from model.whisper_asr import backend_factory
from backend.logging_config import logger
from config.settings import settings
from time import time
from backend.utils.methods import remove_punctuation

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
BYTES_PER_SEC = SAMPLE_RATE * BYTES_PER_SAMPLE
asr, _ = backend_factory(settings)

def confirmation_process(non_confirmed_transcription, tokenize_transcription, confirmed_transciption):
    sliced_tokenize_transcription = [remove_punctuation(" ".join(t.split())) for a,b,t in tokenize_transcription]

    if len(confirmed_transciption) > 0:
        idx = len(confirmed_transciption) - 1
        while idx != -1:
            if confirmed_transciption[idx].strip().endswith(('.', '?', '!')):
                if idx != len(confirmed_transciption) - 1:
                    confirmed_transciption = confirmed_transciption[:idx + 1]
                break
            idx -= 1
        if idx == -1:
            confirmed_transciption = []        

    if len(non_confirmed_transcription) == 0 or non_confirmed_transcription[0] != sliced_tokenize_transcription[0]:
        non_confirmed_transcription = sliced_tokenize_transcription
    elif len(non_confirmed_transcription) > 0:
        idx = 0
        while idx < min(len(non_confirmed_transcription), len(sliced_tokenize_transcription)):
            if (non_confirmed_transcription[idx]).lower() == (sliced_tokenize_transcription[idx]).lower():
                confirmed_transciption.append(non_confirmed_transcription[idx])
                idx += 1
            else:
                break
        if idx > len(non_confirmed_transcription):
            non_confirmed_transcription.extend(sliced_tokenize_transcription[idx:])
        else:
            non_confirmed_transcription[idx:] = sliced_tokenize_transcription[idx:]

    return non_confirmed_transcription, confirmed_transciption


def trim_audio_buffer_offset(tokenize_transcription, non_confirmed_transcription, confirmed_transcription, sample_rate=16000, bytes_per_sample=2):
    """
    Remove audio from pcm_buffer when a complete sentence (ending with . , ? , ! ) is confirmed.
    
    :param pcm_buffer: The bytearray containing the audio buffer.
    :param tokenize_transcription: List of tuples (start_time, end_time, word) from FasterWhisperASR.
    :param confirmed_transcription: List of confirmed sentences.
    :param sample_rate: Sample rate of the audio (default: 16,000 Hz).
    :param bytes_per_sample: Number of bytes per sample (default: 2 for 16-bit PCM).
    """
    if not confirmed_transcription:
        return 0, non_confirmed_transcription  # No confirmed sentences to remove
    
    # Find the last confirmed sentence ending with ., ?, or !
    last_sentence = None
    for sentence in reversed(confirmed_transcription):
        if sentence.strip().endswith(('.', '?', '!')):
            last_sentence = sentence
            break


    if not last_sentence:
        return 0, non_confirmed_transcription  # No complete sentence found, do not trim buffer
    
    confirmed_words = last_sentence.strip()
    end_time = None
    end_word_idx = 0
    
    # Find the corresponding end timestamp of the last word in the sentence
    for start, end, word in tokenize_transcription:
        if confirmed_words in remove_punctuation(word):
            end_time = end
    
    for idx in range(len(non_confirmed_transcription)):
        if confirmed_words in non_confirmed_transcription[idx]:
            end_word_idx = idx
    
    if end_time is None:
        return 0, non_confirmed_transcription  # No match found, do not trim buffer
    
    # Compute bytes to remove
    bytes_to_remove = int(end_time * sample_rate * bytes_per_sample)
    # Trim buffer   
    return bytes_to_remove, non_confirmed_transcription[end_word_idx + 1:]



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
            offset_buffer = 0
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

                    pcm_buffer.extend(chunk)
                    logger.info(f"chunk size: {len(chunk)} bytes")
                    logger.info(f"Buffer size: {len(pcm_buffer)} bytes")

                    if len(pcm_buffer) >= BYTES_PER_SEC:
                        pcm_array = np.frombuffer(pcm_buffer, dtype=np.int16).astype(np.float32) / 32768.0

                        # Transcribe the audio and send back a response
                        transcription = asr.transcribe(pcm_array)
                        tokenize_transcription = asr.ts_words(transcription)
                        transcribe, confirmed_transciption = confirmation_process(transcribe, tokenize_transcription, confirmed_transciption)
                        offset_buffer, transcribe = trim_audio_buffer_offset(tokenize_transcription=tokenize_transcription, confirmed_transcription=confirmed_transciption, non_confirmed_transcription=transcribe, sample_rate=SAMPLE_RATE, bytes_per_sample=BYTES_PER_SAMPLE)
                        pcm_buffer = pcm_buffer[offset_buffer:]
                        logger.info(f"confirmed_transciption: {' '.join(confirmed_transciption)}")
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