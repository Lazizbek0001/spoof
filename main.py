import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        app="src.app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "0") == "1",  # dev only
        workers=1,
        ws_max_size=4 * 1024 * 1024,  # 4 MB per frame message is plenty
    )