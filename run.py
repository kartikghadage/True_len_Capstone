"""TruthLens - local entry point. Run: python run.py -> http://localhost:8000
(In production/Azure the container uses: uvicorn backend.main:app --host 0.0.0.0 --port $PORT)"""
import os 
import uvicorn
if __name__=="__main__":
    port=int(os.getenv("PORT","8000"))
    print("="*55);print("  TruthLens starting...");print(f"  Open:  http://localhost:{port}");print("="*55)
    uvicorn.run("backend.main:app",host="0.0.0.0",port=port,reload=True)
