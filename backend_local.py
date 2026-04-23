from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

latest_data = None
history = []


class Telemetry(BaseModel):
    lat: float
    lon: float
    alt: float
    vel: float
    temp: float
    press: float
    alt_press: float
    temps_txt: str
    temps: float


@app.get("/")
def root():
    return {"ok": True, "message": "Backend actiu"}


@app.post("/telemetry")
def receive_telemetry(data: Telemetry):
    global latest_data, history

    latest_data = data.dict()
    history.append(latest_data)

    if len(history) > 500:
        history = history[-500:]

    return {"ok": True}


@app.get("/telemetry/latest")
def get_latest():
    return latest_data if latest_data else {}


@app.get("/telemetry/history")
def get_history():
    return history
