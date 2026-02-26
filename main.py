import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from router.optimization_router import router as optimization_router
from graph_generator.router import router as graph_router

app = FastAPI()

app.include_router(optimization_router)
app.include_router(graph_router)
app.mount("/graph/static", StaticFiles(directory="graph_generator/static"), name="graph_static")

@app.get("/")
async def root():
    return {"message": "Quantum Hackathon API"}

if '__main__' == __name__:
    uvicorn.run(app, host='0.0.0.0', port=8000)