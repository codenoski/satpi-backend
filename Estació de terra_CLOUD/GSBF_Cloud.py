import math
import time
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import streamlit.components.v1 as components

# =========================
# CONFIG
# =========================
API_BASE = "https://satpi-backend.onrender.com"

MAX_HISTORIAL = 300
REFRESH_SECONDS = 1

SMOOTH_WINDOW = 5
ASCENS_CONFIRM_POINTS = 4
ASCENS_THRESHOLD = 0.8
ASCENS_GAIN_MIN = 3.0

MAP_HEIGHT = 650
MAP_ZOOM = 18
MAP_MOVE_THRESHOLD_METERS = 15.0
MAP_FORCE_REFRESH_SECONDS = 30

TAULA_COLUMNS = [
    "temps", "alt", "alt_suav", "altura_guanyada", "altura_maxima_total",
    "temp", "press", "lat", "lon", "rssi", "servoX", "servoY",
    "vel_calc", "vel_lineal_calc"
]

PLOTLY_CONFIG = {
    "displayModeBar": False,
    "scrollZoom": False,
    "responsive": True,
}


# =========================
# UI BASE
# =========================
st.set_page_config(page_title="Estació de terra", layout="wide")
st.title("Estació de terra Bernat el Ferrer - Satpi")

st.markdown(
    """
    <style>
    .info-card {
        background: #0f1724;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 18px 20px;
        margin-bottom: 14px;
    }

    .info-card h3 {
        margin-top: 0;
        margin-bottom: 14px;
        font-size: 1.35rem;
    }

    .info-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px 20px;
    }

    .info-item {
        padding: 8px 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        font-size: 1rem;
        line-height: 1.35;
    }

    .map-wrap {
        background: #0f1724;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# STATE
# =========================
def init_state():
    if "init" in st.session_state:
        return

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    st.session_state.init = True
    st.session_state.session_id = timestamp
    st.session_state.historial = deque(maxlen=MAX_HISTORIAL)
    st.session_state.altura_base = None
    st.session_state.ha_descendit = False
    st.session_state.last_valid_gps = None

    st.session_state.map_html_cached = ""
    st.session_state.map_last_render_time = 0.0
    st.session_state.map_lat_render = None
    st.session_state.map_lon_render = None


def reset_missio():
    st.session_state.historial = deque(maxlen=MAX_HISTORIAL)
    st.session_state.altura_base = None
    st.session_state.ha_descendit = False
    st.session_state.last_valid_gps = None
    st.session_state.map_html_cached = ""
    st.session_state.map_last_render_time = 0.0
    st.session_state.map_lat_render = None
    st.session_state.map_lon_render = None


init_state()

if not isinstance(st.session_state.historial, deque):
    st.session_state.historial = deque(st.session_state.historial, maxlen=MAX_HISTORIAL)


# =========================
# API
# =========================
def processar_lectura_api():
    try:
        r = requests.get(f"{API_BASE}/telemetry/latest", timeout=10)

        if r.status_code != 200:
            st.warning(f"Backend no disponible: {r.status_code}")
            return

        data = r.json()
        if not data or "temps" not in data:
            return

        data = {
            "temps": float(data["temps"]),
            "alt": float(data["alt"]),
            "temp": float(data["temp"]),
            "press": float(data["press"]),
            "lat": float(data["lat"]),
            "lon": float(data["lon"]),
            "rssi": float(data["rssi"]),
            "servoX": int(data["servoX"]),
            "servoY": int(data["servoY"]),
        }

        if st.session_state.historial:
            ultim_temps = st.session_state.historial[-1]["temps"]

            if data["temps"] == ultim_temps:
                return

            if data["temps"] < ultim_temps:
                reset_missio()

        if coords_valides(data["lat"], data["lon"]):
            st.session_state.last_valid_gps = {
                "lat": data["lat"],
                "lon": data["lon"],
                "temps": data["temps"],
            }

        st.session_state.historial.append(data)

    except Exception as e:
        st.warning(f"No s'han pogut llegir dades del backend: {e}")


# =========================
# GPS / DISTÀNCIA
# =========================
def coords_valides(lat, lon):
    try:
        lat = float(lat)
        lon = float(lon)
    except Exception:
        return False
    return np.isfinite(lat) and np.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180


def metres_per_grau(lat):
    m_lat = 111320.0
    m_lon = 111320.0 * math.cos(math.radians(lat))
    return m_lat, m_lon


def distancia_metres(lat1, lon1, lat2, lon2):
    if not (coords_valides(lat1, lon1) and coords_valides(lat2, lon2)):
        return 0.0

    m_lat = 111320.0
    m_lon = 111320.0 * math.cos(math.radians((lat1 + lat2) / 2.0))
    dx = (lon2 - lon1) * m_lon
    dy = (lat2 - lat1) * m_lat
    return float(math.hypot(dx, dy))


# =========================
# CÀLCULS
# =========================
def calcular_velocitat_lineal_df(df):
    if len(df) < 2:
        return pd.Series(0.0, index=df.index)

    dt = df["temps"].diff()
    valid_actual = df["lat"].between(-90, 90) & df["lon"].between(-180, 180)
    valid_anterior = valid_actual.shift(fill_value=False)
    valid = valid_actual & valid_anterior & (dt > 0)

    m_lat = 111320.0
    m_lon = 111320.0 * np.cos(np.radians(df["lat"]))
    dx = df["lon"].diff() * m_lon
    dy = df["lat"].diff() * m_lat

    vel = pd.Series(np.hypot(dx, dy) / dt, index=df.index)
    return vel.where(valid, 0.0).replace([np.inf, -np.inf], 0).fillna(0.0)


def afegir_variables_altura(df):
    df = df.copy()

    df["alt_suav"] = df["alt"].rolling(window=SMOOTH_WINDOW, min_periods=1).mean()
    df["vel_calc"] = df["alt_suav"].diff() / df["temps"].diff()
    df["vel_calc"] = df["vel_calc"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["vel_lineal_calc"] = calcular_velocitat_lineal_df(df)

    if st.session_state.altura_base is None and len(df) >= ASCENS_CONFIRM_POINTS + 1:
        ultimes_vels = df["vel_calc"].tail(ASCENS_CONFIRM_POINTS)
        guany_finestra = float(
            df["alt_suav"].iloc[-1] - df["alt_suav"].iloc[-ASCENS_CONFIRM_POINTS - 1]
        )

        if (ultimes_vels > ASCENS_THRESHOLD).all() and guany_finestra >= ASCENS_GAIN_MIN:
            idx_ref = max(0, len(df) - ASCENS_CONFIRM_POINTS - 1)
            st.session_state.altura_base = float(df.iloc[idx_ref]["alt_suav"])

    if st.session_state.altura_base is None:
        df["altura_guanyada"] = 0.0
    else:
        df["altura_guanyada"] = (df["alt_suav"] - st.session_state.altura_base).clip(lower=0)

    df["altura_maxima_total"] = df["alt"].cummax()
    return df, st.session_state.altura_base


def calcular_velocitat_vertical(df):
    if len(df) < 2:
        return 0.0

    a = df.iloc[-1]
    b = df.iloc[-2]
    dt = a["temps"] - b["temps"]
    if dt <= 0:
        return 0.0

    return float((a["alt_suav"] - b["alt_suav"]) / dt)


def calcular_temps_aprox_aterratge(df, altura_guanyada, fase):
    if fase != "🪂 Descens" or altura_guanyada <= 0 or len(df) < 5:
        return None

    vels = df["vel_calc"].tail(8)
    vels_negatives = vels[vels < -0.25]

    if len(vels_negatives) < 2:
        return None

    velocitat_descens = abs(float(vels_negatives.mean()))
    if velocitat_descens <= 0:
        return None

    return altura_guanyada / velocitat_descens


def format_temps_aprox(segons):
    if segons is None:
        return "-"

    total = max(0, int(round(segons)))
    minuts, segons_restants = divmod(total, 60)
    hores, minuts_restants = divmod(minuts, 60)

    if hores > 0:
        return f"{hores}h {minuts_restants:02d}m"
    if minuts > 0:
        return f"{minuts}m {segons_restants:02d}s"
    return f"{segons_restants}s"


def obtenir_fase_intelligent(df):
    if len(df) < 6 or st.session_state.altura_base is None:
        return "⏳ Esperant enlairament"

    diffs_alt = df.tail(6)["alt_suav"].diff().dropna()
    avg_diff = diffs_alt.mean()

    altura_guanyada = float(df.iloc[-1]["altura_guanyada"])

    th_estable = 0.25
    th_ascens = 0.8
    th_descens = -0.8

    if avg_diff > th_ascens:
        return "🚀 Ascens"

    if avg_diff < th_descens:
        st.session_state.ha_descendit = True
        return "🪂 Descens"

    if st.session_state.ha_descendit:
        ultimes_vels = df["vel_calc"].tail(6)

        condicions_aterrat = (
            altura_guanyada < 3.0 and
            ultimes_vels.abs().mean() < 0.3
        )

        if condicions_aterrat:
            return "✅ Aterrat"

    return "📡 Vol actiu"


def moviment_estable():
    return {
        "mov_x": "⏺ X estable",
        "mov_y": "⏺ Y estable",
        "mov_z": "⏺ Altitud estable",
        "vel_lineal": 0.0,
        "direccio": "sense moviment",
    }


def calcular_moviment_i_velocitat_lineal(df):
    if len(df) < 2:
        return moviment_estable()

    a = df.iloc[-1]
    b = df.iloc[-2]
    dt = a["temps"] - b["temps"]
    if dt <= 0:
        return moviment_estable()

    delta_lon = a["lon"] - b["lon"]
    delta_lat = a["lat"] - b["lat"]
    delta_alt = a["alt_suav"] - b["alt_suav"]

    th_gps = 0.00001
    th_alt = 0.3

    mov_x = "➡️ X: cap a l'est" if delta_lon > th_gps else "⬅️ X: cap a l'oest" if delta_lon < -th_gps else "⏺ X estable"
    mov_y = "⬆️ Y: cap al nord" if delta_lat > th_gps else "⬇️ Y: cap al sud" if delta_lat < -th_gps else "⏺ Y estable"
    mov_z = "🔼 Z: pujant" if delta_alt > th_alt else "🔽 Z: baixant" if delta_alt < -th_alt else "⏺ Altitud estable"

    if coords_valides(a["lat"], a["lon"]) and coords_valides(b["lat"], b["lon"]):
        m_lat, m_lon = metres_per_grau(a["lat"])
        dx_m = delta_lon * m_lon
        dy_m = delta_lat * m_lat
        vel_lineal = float(a["vel_lineal_calc"])
    else:
        dx_m = dy_m = vel_lineal = 0.0

    comp = []
    if dy_m > 0.3:
        comp.append("nord")
    elif dy_m < -0.3:
        comp.append("sud")

    if dx_m > 0.3:
        comp.append("est")
    elif dx_m < -0.3:
        comp.append("oest")

    return {
        "mov_x": mov_x,
        "mov_y": mov_y,
        "mov_z": mov_z,
        "vel_lineal": vel_lineal,
        "direccio": "-".join(comp) if comp else "sense moviment",
    }


def text_servo_x(valor):
    if valor == 1:
        return "➡️ Servo X: dreta"
    if valor == -1:
        return "⬅️ Servo X: esquerra"
    return "⏺ Servo X: aturat"


def text_servo_y(valor):
    if valor == 1:
        return "⬆️ Servo Y: amunt"
    if valor == -1:
        return "⬇️ Servo Y: avall"
    return "⏺ Servo Y: aturat"


# =========================
# GRÀFIQUES
# =========================
def mini_grafic(df, y, title):
    fig = px.line(df, x="temps", y=y, title=title)
    fig.update_layout(
        height=260,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        transition={"duration": 0},
        uirevision=f"graf-{y}",
    )
    return fig


# =========================
# MAPA LEAFLET
# =========================
def generar_html_mapa_leaflet(lat, lon, zoom=18, height=650):
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8"/>
        <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
        <style>
            html, body {{
                margin: 0;
                padding: 0;
                background: transparent;
            }}

            #map {{
                width: 100%;
                height: {height}px;
                border-radius: 14px;
                overflow: hidden;
                background: #d9d9d9;
            }}

            .pulse-marker {{
                width: 16px;
                height: 16px;
                background: #ff3b30;
                border: 3px solid rgba(255,255,255,0.95);
                border-radius: 50%;
                box-shadow: 0 0 0 0 rgba(255, 59, 48, 0.45);
            }}
        </style>
    </head>
    <body>
        <div id="map"></div>

        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <script>
            const lat = {lat};
            const lon = {lon};
            const zoom = {zoom};

            const map = L.map("map", {{
                zoomControl: true,
                attributionControl: true,
                preferCanvas: true,
                zoomAnimation: false,
                fadeAnimation: false,
                markerZoomAnimation: false
            }}).setView([lat, lon], zoom);

            L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
                maxZoom: 19,
                maxNativeZoom: 19,
                detectRetina: false,
                attribution: "&copy; OpenStreetMap contributors"
            }}).addTo(map);

            const redIcon = L.divIcon({{
                className: "",
                html: '<div class="pulse-marker"></div>',
                iconSize: [22, 22],
                iconAnchor: [11, 11]
            }});

            L.marker([lat, lon], {{ icon: redIcon }}).addTo(map);

            L.circle([lat, lon], {{
                color: "#ff3b30",
                weight: 2,
                fillColor: "#ff3b30",
                fillOpacity: 0.10,
                radius: 8
            }}).addTo(map);

            setTimeout(() => {{
                map.invalidateSize();
            }}, 150);
        </script>
    </body>
    </html>
    """


def renderitzar_mapa():
    gps = st.session_state.last_valid_gps

    if gps is None:
        st.warning("No hi ha coordenades GPS vàlides per mostrar el mapa.")
        return

    lat = float(gps["lat"])
    lon = float(gps["lon"])

    if not coords_valides(lat, lon):
        st.warning("Coordenades invàlides.")
        return

    now = time.time()

    if st.session_state.map_lat_render is None or st.session_state.map_lon_render is None:
        dist_m = 999999.0
    else:
        dist_m = distancia_metres(
            st.session_state.map_lat_render,
            st.session_state.map_lon_render,
            lat,
            lon
        )

    elapsed = now - st.session_state.map_last_render_time

    cal_actualitzar = (
        st.session_state.map_html_cached == ""
        or dist_m >= MAP_MOVE_THRESHOLD_METERS
        or elapsed >= MAP_FORCE_REFRESH_SECONDS
    )

    if cal_actualitzar:
        st.session_state.map_lat_render = lat
        st.session_state.map_lon_render = lon
        st.session_state.map_last_render_time = now
        st.session_state.map_html_cached = generar_html_mapa_leaflet(
            lat=lat,
            lon=lon,
            zoom=MAP_ZOOM,
            height=MAP_HEIGHT
        )

    st.markdown('<div class="map-wrap">', unsafe_allow_html=True)
    components.html(
        st.session_state.map_html_cached,
        height=MAP_HEIGHT,
        scrolling=False,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# RENDER
# =========================
def renderitzar_bloc_gps_i_mapa(
    dada,
    fase,
    altura_guanyada,
    altura_maxima_total,
    vel_lineal,
    direccio_lineal,
    temps_aterratge_txt,
    altura_base,
):
    st.subheader("📍 Posició GPS en temps real")
    gps = st.session_state.last_valid_gps

    col_info, col_map = st.columns([1, 1.55], gap="large")

    with col_info:
        if gps is None:
            st.warning("No hi ha coordenades GPS vàlides.")
        else:
            html_info = f"""
            <div class="info-card">
                <h3>Posició actual</h3>
                <div class="info-grid">
                    <div class="info-item"><b>Latitud:</b> {gps['lat']:.6f}</div>
                    <div class="info-item"><b>Longitud:</b> {gps['lon']:.6f}</div>
                    <div class="info-item"><b>Temps:</b> {dada['temps']:.1f} s</div>
                    <div class="info-item"><b>Etapa:</b> {fase}</div>
                    <div class="info-item"><b>Altitud absoluta:</b> {dada['alt']:.1f} m</div>
                    <div class="info-item"><b>Altura guanyada:</b> {altura_guanyada:.1f} m</div>
                    <div class="info-item"><b>Altura màxima total:</b> {altura_maxima_total:.1f} m</div>
                    <div class="info-item"><b>Velocitat lineal:</b> {vel_lineal:.2f} m/s</div>
                    <div class="info-item"><b>Direcció:</b> {direccio_lineal}</div>
                    <div class="info-item"><b>Temps aprox. aterratge:</b> {temps_aterratge_txt}</div>
                    <div class="info-item"><b>Altura de llançament:</b> {"pendent" if altura_base is None else f"{altura_base:.1f} m"}</div>
                    <div class="info-item"><b>RSSI:</b> {dada['rssi']:.1f} dBm</div>
                </div>
            </div>
            """
            st.markdown(html_info, unsafe_allow_html=True)

    with col_map:
        renderitzar_mapa()


def renderitzar_dashboard():
    if not st.session_state.historial:
        st.write("Encara no hi ha dades")
        return

    df = pd.DataFrame(st.session_state.historial)
    df, altura_base = afegir_variables_altura(df)

    dada = df.iloc[-1]
    fase = obtenir_fase_intelligent(df)
    vel_vertical = calcular_velocitat_vertical(df)
    moviment = calcular_moviment_i_velocitat_lineal(df)

    altura_guanyada = float(dada["altura_guanyada"])
    altura_maxima_total = float(dada["altura_maxima_total"])
    vel_lineal = moviment["vel_lineal"]
    direccio_lineal = moviment["direccio"]

    temps_aterratge_s = calcular_temps_aprox_aterratge(df, altura_guanyada, fase)
    temps_aterratge_txt = format_temps_aprox(temps_aterratge_s)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Temps", f"{dada['temps']:.1f} s")
    m2.metric("Altitud", f"{dada['alt']:.1f} m")
    m3.metric("Velocitat vertical", f"{vel_vertical:.2f} m/s")
    m4.metric("Velocitat lineal", f"{vel_lineal:.2f} m/s")
    m5.metric("Altura guanyada", f"{altura_guanyada:.1f} m")
    m6.metric("Temps aprox. aterratge", temps_aterratge_txt)

    m7, m8, m9, m10 = st.columns(4)
    m7.metric("Altura màxima total", f"{altura_maxima_total:.1f} m")
    m8.metric("Temperatura", f"{dada['temp']:.1f} °C")
    m9.metric("Pressió", f"{dada['press']:.1f} hPa")
    m10.metric("RSSI", f"{dada['rssi']:.1f} dBm")

    col_estat, col_mov = st.columns(2)

    with col_estat:
        st.subheader("🛰️ Etapa de la missió")
        st.info(fase)

        if dada["rssi"] < -95:
            st.error("🔴 Senyal molt dèbil")
        elif dada["rssi"] < -85:
            st.warning("🟡 Senyal dèbil")
        else:
            st.success("🟢 Senyal OK")

        st.info(text_servo_x(dada["servoX"]))
        st.info(text_servo_y(dada["servoY"]))

    with col_mov:
        st.subheader("🧭 Moviment")
        st.success(moviment["mov_x"])
        st.info(moviment["mov_y"])
        st.warning(moviment["mov_z"])
        st.info(f"➡️ Velocitat lineal: {vel_lineal:.2f} m/s cap a {direccio_lineal}")

    renderitzar_bloc_gps_i_mapa(
        dada=dada,
        fase=fase,
        altura_guanyada=altura_guanyada,
        altura_maxima_total=altura_maxima_total,
        vel_lineal=vel_lineal,
        direccio_lineal=direccio_lineal,
        temps_aterratge_txt=temps_aterratge_txt,
        altura_base=altura_base,
    )

    if len(df) >= 2:
        st.subheader("📈 Gràfiques principals")
        st.plotly_chart(
            mini_grafic(df, "alt", "Altitud vs Temps"),
            use_container_width=True,
            config=PLOTLY_CONFIG,
            key="fig_alt",
        )

        st.subheader("📊 Altres dades")
        a1, a2, a3 = st.columns(3)
        with a1:
            st.plotly_chart(
                mini_grafic(df, "temp", "Temperatura"),
                use_container_width=True,
                config=PLOTLY_CONFIG,
                key="fig_temp",
            )
        with a2:
            st.plotly_chart(
                mini_grafic(df, "press", "Pressió"),
                use_container_width=True,
                config=PLOTLY_CONFIG,
                key="fig_press",
            )
        with a3:
            st.plotly_chart(
                mini_grafic(df, "rssi", "RSSI"),
                use_container_width=True,
                config=PLOTLY_CONFIG,
                key="fig_rssi",
            )

        b1, b2 = st.columns(2)
        with b1:
            st.plotly_chart(
                mini_grafic(df, "vel_calc", "Velocitat vertical calculada"),
                use_container_width=True,
                config=PLOTLY_CONFIG,
                key="fig_vel",
            )
        with b2:
            st.plotly_chart(
                mini_grafic(df, "vel_lineal_calc", "Velocitat lineal"),
                use_container_width=True,
                config=PLOTLY_CONFIG,
                key="fig_vlin",
            )

    st.subheader("📋 Últimes dades")
    st.dataframe(df[TAULA_COLUMNS].tail(10), use_container_width=True)


# =========================
# LOOP
# =========================
if hasattr(st, "fragment"):

    @st.fragment(run_every=f"{REFRESH_SECONDS}s")
    def bloc_temps_real():
        processar_lectura_api()
        renderitzar_dashboard()

    bloc_temps_real()

else:
    processar_lectura_api()
    renderitzar_dashboard()
    time.sleep(REFRESH_SECONDS)
    st.rerun()