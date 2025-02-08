from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from backend.utils.html_loader import load_html
from backend.websocket import handle_websocket

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def get():
    return HTMLResponse(load_html())

@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    await handle_websocket(websocket)
