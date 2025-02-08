import argparse


def get_settings():
    parser = argparse.ArgumentParser(description="Whisper FastAPI Online Server")
    parser.add_argument("--host", type=str, default="localhost", help="The host address.")
    parser.add_argument("--port", type=int, default=8000, help="The port number.")
    parser.add_argument("--warmup-file", type=str, dest="warmup_file", help="Path to warm-up file.")
    parser.add_argument("--diarization", action="store_true", help="Enable speaker diarization.")

    from model.asr_base import add_shared_args
    add_shared_args(parser)

    return parser.parse_args()


settings = get_settings()
