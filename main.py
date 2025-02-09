import uvicorn
from backend.app import app
from config.settings import settings

if __name__ == "__main__":
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8030, reload=True)
