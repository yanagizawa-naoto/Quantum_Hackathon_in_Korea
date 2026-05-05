import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from router import optimization_router, optimization_v1_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://quantum-guardians.github.io",
        "https://mr2s.vercel.app",
        "https://qi4uinpnu.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

app.include_router(optimization_router)
app.include_router(optimization_v1_router)

@app.get("/")
async def root():
    return {"message": "Quantum Hackathon API"}

if '__main__' == __name__:
    uvicorn.run(app, host='0.0.0.0', port=8000)