import ffmpeg
import asyncio

SAMPLE_RATE = 16000
CHANNELS = 1

async def start_ffmpeg_decoder():
    return ffmpeg.input("pipe:0", format="webm")\
                 .output("pipe:1", format="s16le", acodec="pcm_s16le", ac=CHANNELS, ar=str(SAMPLE_RATE))\
                 .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
