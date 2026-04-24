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
    camX: str = "center"
    camY: str = "center"
    pc_rebut_ts: float | None = None


@app.get("/")
def root():
    return {"ok": True, "message": "Backend actiu"}


@app.post("/telemetry")
def receive_telemetry(data: Telemetry):
    global latest_data, history

    latest_data = data.dict()
    # Normalitza camX/camY com strings (left/right/center)
    def _norm_cam(v):
        v = str(v).strip().lower()
        return v if v in {"left", "right", "center", "up", "down"} else "center"

    latest_data["camX"] = _norm_cam(latest_data.get("camX", "center"))
    latest_data["camY"] = _norm_cam(latest_data.get("camY", "center"))
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
