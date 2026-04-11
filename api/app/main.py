from fastapi import FastAPI
app = FastAPI(title="Thornet API", description="Microservice Health Check")
@app.get("/")
def read_root():
 return {"status": "success", "message": "FastAPI Microservice is online and ready for MLflow integration!"}