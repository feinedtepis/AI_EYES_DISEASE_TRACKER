"""Start the EyeVision AI web app:  python run_app.py   ->  http://127.0.0.1:8000"""
import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "127.0.0.1")
    print(f"EyeVision AI running at http://{host}:{port}  (Ctrl+C to stop)")
    uvicorn.run("app.main:app", host=host, port=port, reload=False)
