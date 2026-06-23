import streamlit as st
import streamlit.components.v1 as components
import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import holidays
import requests
import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
from io import StringIO
from entsoe import EntsoePandasClient

st.set_page_config(
    page_title="MVM Energiafogyasztás Előrejelző",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@700;800&family=Inter:wght@400;500&display=swap" rel="stylesheet">
<style>
.stApp { background-color: #050d1a; }
.block-container { padding: 1rem 2rem; max-width: 100%; }
.stButton > button {
    background: linear-gradient(135deg, #0066CC, #0044aa) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important;
    font-family: Montserrat, sans-serif !important;
    font-weight: 600 !important; padding: 10px 24px !important;
}
.stTabs [data-baseweb="tab"] { color: #94a3b8 !important; font-family: Inter, sans-serif !important; font-size: 14px !important; }
.stTabs [aria-selected="true"] { color: #FF6600 !important; border-bottom: 2px solid #FF6600 !important; background: transparent !important; }
[data-testid="stMetricValue"] { color: #FF6600 !important; font-weight: 700 !important; font-size: 15px !important; text-align: center !important; }
[data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 10px !important; text-align: center !important; letter-spacing: 0.5px !important; }
[data-testid="metric-container"] {
    background: #0a1628 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 12px !important;
    padding: 12px 8px !important;
    text-align: center !important;
}
footer { display: none !important; }
#MainMenu { display: none !important; }
header { display: none !important; }
.kartya-sor {
    display: grid;
    grid-template-columns: repeat(8, 1fr);
    gap: 8px;
    margin-bottom: 16px;
    width: 100%;
}
.kartya {
    background: #0a1628;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 10px 8px;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0,102,204,0.15);
    min-height: 90px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    cursor: default;
    transition: border-color 0.25s ease, box-shadow 0.25s ease, transform 0.2s ease;
}
.kartya:hover {
    border-color: #0066CC;
    box-shadow: 0 4px 24px rgba(0,102,204,0.45), 0 0 0 1px rgba(0,102,204,0.3);
    transform: translateY(-3px);
}
.kartya-cim {
    color: #64748b; font-size: 8px; font-family: Inter,sans-serif;
    letter-spacing: 0.8px; text-transform: uppercase; margin-bottom: 4px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.kartya-ertek {
    color: #FF6600; font-size: 14px; font-weight: 700;
    font-family: Montserrat,sans-serif; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis;
}
.kartya-trend {
    font-size: 11px; margin-top: 3px; white-space: nowrap;
    cursor: help;
    border-bottom: 1px dotted transparent;
    transition: border-color 0.2s;
}
.kartya-trend:hover { border-bottom-color: currentColor; }
.kartya-sub { color: #94a3b8; font-size: 10px; margin-top: 3px; white-space: nowrap; }
</style>
""", unsafe_allow_html=True)

RIADOKUSZOB = 6812.0
LAG_KEZDO = 4796.52
hu_holidays = holidays.Hungary(years=[2026, 2027])
ENTSOE_API_KEY = os.environ.get("ENTSOE_API_KEY", "")
DATABASE_URL = os.environ.get("DATABASE_URL", "")
BASE = os.path.dirname(os.path.abspath(__file__))

class FogyasztasModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(9, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.model(x)

@st.cache_resource
def modellek_betoltese():
    sx1 = joblib.load(f"{BASE}/scaler_X_v1.pkl")
    sy1 = joblib.load(f"{BASE}/scaler_y_v1.pkl")
    m1 = FogyasztasModel()
    m1.load_state_dict(torch.load(f"{BASE}/fogyasztas_model_v1.pth", map_location="cpu"))
    m1.eval()
    sx2 = joblib.load(f"{BASE}/scaler_X_v2.pkl")
    sy2 = joblib.load(f"{BASE}/scaler_y_v2.pkl")
    m2 = FogyasztasModel()
    m2.load_state_dict(torch.load(f"{BASE}/fogyasztas_model_v2.pth", map_location="cpu"))
    m2.eval()
    return sx1, sy1, m1, sx2, sy2, m2

scaler_X_v1, scaler_y_v1, model_v1, scaler_X_v2, scaler_y_v2, model_v2 = modellek_betoltese()

def magyar_ma():
    return pd.Timestamp.now(tz="Europe/Budapest").normalize().tz_localize(None).to_pydatetime()

def ments_db_be(eredmenyek, eur_huf):
    if not DATABASE_URL:
        return True, ""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        ma = pd.Timestamp.now(tz="Europe/Budapest").date()
        cur.execute("DELETE FROM predikciok WHERE joslas_datuma = %s", (ma,))
        for e in eredmenyek:
            koltseg = e["fogyasztas"] * 24 * e["dam_ar"] * eur_huf / 1_000_000
            cur.execute("""
                INSERT INTO predikciok 
                (joslas_datuma, cel_datuma, fogyasztas_mwh, homerseklet, 
                 dam_ar, koltseg_mft, modell, riado, eur_huf)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (ma, e["datum"].date(), float(e["fogyasztas"]), float(e["homerseklet"]),
                  float(e["dam_ar"]), float(koltseg), e["modell"], bool(e["riado"]), float(eur_huf)))
        conn.commit()
        cur.close()
        conn.close()
        return True, f"{len(eredmenyek)} sor sikeresen mentve"
    except Exception as ex:
        return False, f"Hiba: {str(ex)}"

def olvas_db_bol(datum_str):
    if not DATABASE_URL:
        return None, None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT * FROM predikciok 
            WHERE joslas_datuma = %s 
            ORDER BY cel_datuma
        """, (datum_str,))
        sorok = cur.fetchall()
        cur.close()
        conn.close()
        return [dict(s) for s in sorok], None
    except Exception as ex:
        return None, str(ex)

def db_osszes_datum():
    if not DATABASE_URL:
        return []
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT joslas_datuma FROM predikciok ORDER BY joslas_datuma DESC")
        datumok = [str(r[0]) for r in cur.fetchall()]
        cur.close()
        conn.close()
        return datumok
    except:
        return []

@st.cache_data(ttl=300)
def get_eur_huf(_datum_kulcs=None):
    try:
        url = "https://data-api.ecb.europa.eu/service/data/EXR/D.HUF.EUR.SP00.A"
        params = {"startPeriod": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
                  "endPeriod": datetime.now().strftime("%Y-%m-%d"), "format": "csvdata"}
        r = requests.get(url, params=params, timeout=10)
        df = pd.read_csv(StringIO(r.text))
        df = df[["TIME_PERIOD", "OBS_VALUE"]].dropna()
        df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
        return float(df["OBS_VALUE"].dropna().iloc[-1])
    except:
        return 395.0

@st.cache_data(ttl=600)
def get_idojaras(datum_kulcs=None):
    try:
        ma = magyar_ma()
        url = "https://api.open-meteo.com/v1/forecast"
        params = {"latitude": 47.5, "longitude": 19.0,
                  "daily": "temperature_2m_mean",
                  "timezone": "Europe/Budapest",
                  "start_date": (ma + timedelta(days=1)).strftime("%Y-%m-%d"),
                  "end_date": (ma + timedelta(days=7)).strftime("%Y-%m-%d")}
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        return [{"datum": pd.to_datetime(d), "homerseklet": t}
                for d, t in zip(data["daily"]["time"], data["daily"]["temperature_2m_mean"])]
    except:
        ma = magyar_ma()
        return [{"datum": pd.to_datetime(ma + timedelta(days=i+1)), "homerseklet": 12.0}
                for i in range(7)]

@st.cache_data(ttl=300)
@st.cache_data(ttl=600)
def get_valos_kezdo_lag(_datum_kulcs=None):
    if not ENTSOE_API_KEY:
        return LAG_KEZDO, False
    try:
        client = EntsoePandasClient(api_key=ENTSOE_API_KEY)
        ma = magyar_ma()
        ma = pd.Timestamp(ma).normalize()
        for delta in [1, 2, 3]:
            nap = ma - pd.Timedelta(days=delta)
            try:
                load = client.query_load("HU", start=nap, end=nap + pd.Timedelta(days=1))
                s = load.iloc[:, 0].dropna()
                if len(s) >= 20:
                    return float(s.mean()), True
            except:
                continue
        return LAG_KEZDO, False
    except:
        return LAG_KEZDO, False
def get_dam_ar(_datum_kulcs=None):
    if not ENTSOE_API_KEY:
        return 104.93, 98.91, False
    try:
        client = EntsoePandasClient(api_key=ENTSOE_API_KEY)
        ma = magyar_ma()
        legfrissebb = 104.93
        for delta in [-1, -2, -3]:
            try:
                nap = ma + timedelta(days=delta)
                start = pd.Timestamp(nap.strftime("%Y-%m-%d"), tz="Europe/Budapest")
                end = start + pd.Timedelta(days=1)
                dam = client.query_day_ahead_prices("HU", start=start, end=end)
                if dam is not None and len(dam) > 0:
                    dam_csucside = dam.between_time("08:00", "20:00")
                    if len(dam_csucside) > 0:
                        ertek = float(dam_csucside.mean())
                    else:
                        ertek = float(dam.mean())
                    if ertek >= 20.0:
                        legfrissebb = ertek
                        break
            except:
                continue
        start_30 = pd.Timestamp((ma - timedelta(days=30)).strftime("%Y-%m-%d"), tz="Europe/Budapest")
        end_30 = pd.Timestamp(ma.strftime("%Y-%m-%d"), tz="Europe/Budapest")
        dam_30 = client.query_day_ahead_prices("HU", start=start_30, end=end_30)
        atlag_30 = float(dam_30.mean()) if dam_30 is not None and len(dam_30) > 0 else 98.91
        if legfrissebb == 104.93:
            legfrissebb = atlag_30
        return legfrissebb, atlag_30, True
    except:
        return 104.93, 98.91, False

def ensemble_joslas(idojaras_lista, dam_ar_1nap, dam_atlag_30, kezdo_lag=LAG_KEZDO):
    eredmenyek = []
    lag = kezdo_lag
    for i, nap in enumerate(idojaras_lista):
        datum = nap["datum"]
        homerseklet = nap["homerseklet"]
        dam_ar = dam_ar_1nap if i == 0 else dam_atlag_30
        X = np.array([[lag, homerseklet, datum.weekday() + 1, datum.month, dam_ar,
                        1 if datum.date() in hu_holidays else 0,
                        1 if datum.weekday() >= 5 else 0,
                        1 if homerseklet < -5 else 0,
                        1 if homerseklet > 30 else 0]])
        if homerseklet < -5:
            X_sc = scaler_X_v2.transform(X)
            X_t = torch.tensor(X_sc, dtype=torch.float32)
            with torch.no_grad():
                pred = model_v2(X_t)
            josolt = float(scaler_y_v2.inverse_transform(pred.numpy())[0][0])
            modell = "V2"
        else:
            X_sc = scaler_X_v1.transform(X)
            X_t = torch.tensor(X_sc, dtype=torch.float32)
            with torch.no_grad():
                pred = model_v1(X_t)
            josolt = float(scaler_y_v1.inverse_transform(pred.numpy())[0][0])
            modell = "V1"
        lag = josolt
        eredmenyek.append({"datum": datum, "homerseklet": homerseklet,
                           "fogyasztas": josolt, "dam_ar": dam_ar,
                           "modell": modell, "riado": josolt >= RIADOKUSZOB})
    return eredmenyek

def plotly_chart(adatok_json, layout_json, chart_id, height=420):
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdn.plot.ly/plotly-2.26.0.min.js" charset="utf-8"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#0a1628; overflow:hidden; }}
#{chart_id} {{
    width:100%; height:{height}px;
    opacity: 0;
    transition: opacity 0.6s ease-out;
}}
#{chart_id}.fade-in {{ opacity: 1; }}
</style>
</head>
<body>
<div id="{chart_id}"></div>
<script>
(function() {{
    var finalData = {adatok_json};
    var layout = {layout_json};
    var config = {{responsive: true, displayModeBar: false}};
    layout.dragmode = false;
    layout.hovermode = 'closest';
    var chartEl = document.getElementById('{chart_id}');
    var animating = false;
    var lastVisible = false;
    var animDone = false;
    var START_DELAY = 500;
    var ANIM_DURATION = 2500;
    var FRAME_RATE = 60;
    var TOTAL_STEPS = Math.round(ANIM_DURATION * FRAME_RATE / 1000);
    var FRAME_INTERVAL = 1000 / FRAME_RATE;
    function makeStartData() {{
        return finalData.map(function(trace) {{
            var t = JSON.parse(JSON.stringify(trace));
            if (t.y) t.y = t.y.map(function() {{ return 0; }});
            if (t.text) {{
                t.text = t.text.map(function() {{ return ''; }});
            }}
            return t;
        }});
    }}
    function easeOutCubic(t) {{
        return 1 - Math.pow(1 - t, 3);
    }}
    function isVisible() {{
        if (document.hidden) return false;
        var rect = chartEl.getBoundingClientRect();
        return rect.width > 0 && rect.height > 0;
    }}
    function runAnimation() {{
        if (animating) return;
        animating = true;
        chartEl.style.opacity = '0';
        Plotly.react('{chart_id}', makeStartData(), layout, config).then(function() {{
            setTimeout(function() {{
                chartEl.style.opacity = '1';
                var current = 0;
                var timer = setInterval(function() {{
                    current++;
                    var t = current / TOTAL_STEPS;
                    var eased = easeOutCubic(t);
                    var animData = finalData.map(function(trace) {{
                        var copy = JSON.parse(JSON.stringify(trace));
                        if (copy.y) copy.y = copy.y.map(function(v) {{ return v * eased; }});
                        if (copy.text && t < 0.9) {{
                            copy.text = copy.text.map(function() {{ return ''; }});
                        }}
                        return copy;
                    }});
                    Plotly.react('{chart_id}', animData, layout, config);
                    if (current >= TOTAL_STEPS) {{
                        clearInterval(timer);
                        Plotly.react('{chart_id}', finalData, layout, config);
                        animating = false;
                        animDone = true;
                    }}
                }}, FRAME_INTERVAL);
            }}, 100);
        }});
    }}
    Plotly.newPlot('{chart_id}', makeStartData(), layout, config).then(function() {{
        setInterval(function() {{
            var vis = isVisible();
            if (vis && !lastVisible) {{
                setTimeout(runAnimation, START_DELAY);
            }}
            if (!vis && animDone) {{
                animDone = false;
            }}
            lastVisible = vis;
        }}, 300);
    }});
}})();
</script>
</body>
</html>"""

def fogyasztas_chart(datumok, fogyasztasok_gwh, modellek, riadok, eur_huf, height=420):
    colors = ["#FF6600" if r else "#0066CC" for r in riadok]
    feliratok = [f"{v:.1f} ({m})" for v, m in zip(fogyasztasok_gwh, modellek)]
    x_min = (pd.to_datetime(datumok[0]) - pd.Timedelta(hours=18)).isoformat()
    x_max = (pd.to_datetime(datumok[-1]) + pd.Timedelta(hours=18)).isoformat()
    kuszob_gwh = RIADOKUSZOB * 24 / 1000
    adatok = json.dumps([{
        "type": "bar", "x": datumok, "y": fogyasztasok_gwh,
        "marker": {"color": colors, "opacity": 0.9},
        "text": feliratok, "textposition": "outside",
        "textfont": {"color": "#ffffff", "size": 11},
        "hovertemplate": "%{x}<br>%{text}<extra></extra>",
        "name": "Fogyasztás", "cliponaxis": False
    }])
    layout = json.dumps({
        "paper_bgcolor": "#0a1628", "plot_bgcolor": "#0f2040",
        "font": {"color": "#cbd5e1", "family": "Inter"},
        "title": {"text": "7 napos fogyasztás előrejelzés (GWh)",
                  "font": {"size": 14, "color": "#f1f5f9"}},
        "margin": {"l": 100, "r": 80, "t": 70, "b": 50},
        "xaxis": {"gridcolor": "#1e3a5f", "tickformat": "%m.%d", "color": "#cbd5e1",
                  "range": [x_min, x_max], "type": "date"},
        "yaxis": {"gridcolor": "#1e3a5f", "title": "GWh", "color": "#cbd5e1",
                  "range": [0, max(max(fogyasztasok_gwh) * 1.25, kuszob_gwh * 1.05)]},
        "bargap": 0.55, "showlegend": False,
        "shapes": [{"type": "line", "xref": "paper", "x0": 0, "x1": 1,
                    "y0": kuszob_gwh, "y1": kuszob_gwh,
                    "line": {"color": "#FF6600", "width": 2, "dash": "dash"}}],
        "annotations": [{"xref": "paper", "x": 0.5, "y": kuszob_gwh,
                         "text": f"Riasztási küszöb ({kuszob_gwh:.1f} GWh)",
                         "showarrow": False, "font": {"color": "#FF6600", "size": 10},
                         "yanchor": "bottom",
                         "bgcolor": "#0a1628", "borderpad": 2}]
    })
    return plotly_chart(adatok, layout, "fogyasztas", height)

def koltseg_chart(datumok, koltsegek, eur_huf, height=420):
    x_min = (pd.to_datetime(datumok[0]) - pd.Timedelta(hours=18)).isoformat()
    x_max = (pd.to_datetime(datumok[-1]) + pd.Timedelta(hours=18)).isoformat()
    adatok = json.dumps([{
        "type": "scatter", "x": datumok, "y": koltsegek,
        "mode": "lines+markers+text",
        "line": {"color": "#FF6600", "width": 3},
        "marker": {"size": 10, "color": "#FF6600", "line": {"color": "#ffffff", "width": 2}},
        "fill": "tozeroy", "fillcolor": "rgba(255,102,0,0.15)",
        "text": [f"{v:.1f}" for v in koltsegek],
        "textposition": "top center",
        "textfont": {"color": "#ffffff", "size": 10},
        "hovertemplate": "%{x}<br>%{y:.1f} M Ft<extra></extra>",
        "name": "Költség", "cliponaxis": False
    }])
    y_min = min(koltsegek) * 0.80
    y_max = max(koltsegek) * 1.20
    layout = json.dumps({
        "paper_bgcolor": "#0a1628", "plot_bgcolor": "#0f2040",
        "font": {"color": "#cbd5e1", "family": "Inter"},
        "title": {"text": f"Becsült napi energiaköltség (M Ft) | EUR/HUF: {eur_huf:.1f}",
                  "font": {"size": 14, "color": "#f1f5f9"}},
        "margin": {"l": 100, "r": 80, "t": 70, "b": 50},
        "xaxis": {"gridcolor": "#1e3a5f", "tickformat": "%m.%d", "color": "#cbd5e1",
                  "range": [x_min, x_max], "type": "date"},
        "yaxis": {"gridcolor": "#1e3a5f", "title": "Millió Ft", "color": "#cbd5e1",
                  "range": [y_min, y_max]},
        "showlegend": False
    })
    return plotly_chart(adatok, layout, "koltseg", height)

def homerseklet_chart(datumok, homersekletek, height=420):
    x_min = (pd.to_datetime(datumok[0]) - pd.Timedelta(hours=18)).isoformat()
    x_max = (pd.to_datetime(datumok[-1]) + pd.Timedelta(hours=18)).isoformat()
    adatok = json.dumps([{
        "type": "scatter", "x": datumok, "y": homersekletek,
        "mode": "lines+markers+text",
        "line": {"color": "#10b981", "width": 3},
        "marker": {"size": 10, "color": "#10b981", "line": {"color": "#ffffff", "width": 2}},
        "fill": "tozeroy", "fillcolor": "rgba(16,185,129,0.15)",
        "text": [f"{v:.1f}°" for v in homersekletek],
        "textposition": "top center",
        "textfont": {"color": "#ffffff", "size": 10},
        "hovertemplate": "%{x}<br>%{y:.1f}°C<extra></extra>",
        "name": "Hőmérséklet", "cliponaxis": False
    }])
    layout = json.dumps({
        "paper_bgcolor": "#0a1628", "plot_bgcolor": "#0f2040",
        "font": {"color": "#cbd5e1", "family": "Inter"},
        "title": {"text": "Hőmérséklet előrejelzés (°C)",
                  "font": {"size": 14, "color": "#f1f5f9"}},
        "margin": {"l": 100, "r": 80, "t": 70, "b": 50},
        "xaxis": {"gridcolor": "#1e3a5f", "tickformat": "%m.%d", "color": "#cbd5e1",
                  "range": [x_min, x_max], "type": "date"},
        "yaxis": {"gridcolor": "#1e3a5f", "title": "°C", "color": "#cbd5e1",
                  "range": [min(homersekletek) - 5, max(homersekletek) + 6]},
        "showlegend": False,
        "shapes": [{"type": "line", "xref": "paper", "x0": 0, "x1": 1,
                    "y0": -5, "y1": -5,
                    "line": {"color": "#FF6600", "width": 1.5, "dash": "dot"}}],
        "annotations": [{"xref": "paper", "x": 0.98, "y": -5,
                         "text": "Extrém hideg (-5°C)",
                         "showarrow": False, "font": {"color": "#FF6600", "size": 10},
                         "yanchor": "top", "xanchor": "right",
                         "bgcolor": "#0a1628", "borderpad": 2}]
    })
    return plotly_chart(adatok, layout, "homerseklet", height)

st.markdown("""
<div style="background:linear-gradient(135deg,#003366 0%,#0055aa 100%);
            padding:20px 32px; border-radius:12px; margin-bottom:16px;
            border-bottom:3px solid #FF6600;
            box-shadow:0 4px 20px rgba(0,102,204,0.3);">
    <div style="display:flex; align-items:center; gap:16px;">
        <div style="background:#0066CC; border-radius:50%; width:52px; height:52px;
                    display:flex; align-items:center; justify-content:center;
                    box-shadow:0 0 15px rgba(0,102,204,0.5);">
            <span style="color:#FF6600; font-size:28px; line-height:1;">⚡</span>
        </div>
        <div>
            <div style="font-family:Montserrat,sans-serif; font-size:26px;
                        font-weight:800; color:#ffffff; letter-spacing:1px;">
                MVM Energiafogyasztás Előrejelző
            </div>
            <div style="font-family:Inter,sans-serif; font-size:13px;
                        color:#93c5fd; margin-top:4px; letter-spacing:0.5px;">
                CrewAI + PyTorch Ensemble (V1 + V2) &nbsp;|&nbsp; 7 napos előrejelzés &nbsp;|&nbsp; Azure PostgreSQL
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📊 Napi Predikció", "📅 Korábbi jóslatok"])

with tab1:
    col_gomb, col_allapot = st.columns([1, 3])
    with col_gomb:
        frissit = st.button("🔄 Előrejelzés frissítése", type="primary")
    with col_allapot:
        allapot_ph = st.empty()

    if frissit:
        st.session_state.frissites_ideje = pd.Timestamp.now(tz="Europe/Budapest").strftime("%Y-%m-%d %H:%M:%S")
        get_eur_huf.clear()
        get_idojaras.clear()
        get_dam_ar.clear()
        for key in ["eredmenyek", "eur_huf", "dam_ar_1nap", "dam_atlag_30", "dam_valodi", "db_mentes_uzenet"]:
            st.session_state.pop(key, None)
        st.rerun()

    if "eredmenyek" not in st.session_state:
        with st.spinner("Adatok lekérése..."):
            ora_kulcs = magyar_ma().strftime("%Y-%m-%d-%H")
            eur_huf = get_eur_huf(_datum_kulcs=ora_kulcs)
            idojaras_lista = get_idojaras(datum_kulcs=ora_kulcs)
            dam_ar_1nap, dam_atlag_30, dam_valodi = get_dam_ar(_datum_kulcs=ora_kulcs)
            kezdo_lag, lag_valodi = get_valos_kezdo_lag(_datum_kulcs=ora_kulcs)
            eredmenyek = ensemble_joslas(idojaras_lista, dam_ar_1nap, dam_atlag_30, kezdo_lag)
            st.session_state.eredmenyek = eredmenyek
            st.session_state.eur_huf = eur_huf
            st.session_state.dam_ar_1nap = dam_ar_1nap
            st.session_state.dam_atlag_30 = dam_atlag_30
            st.session_state.dam_valodi = dam_valodi
            st.session_state.frissites_ideje = pd.Timestamp.now(tz="Europe/Budapest").strftime("%Y-%m-%d %H:%M:%S")
            siker, uzenet = ments_db_be(eredmenyek, eur_huf)
            st.session_state.db_mentes_uzenet = (siker, uzenet)

    if "frissites_ideje" in st.session_state:
        allapot_ph.success(f"✅ Frissítve: {st.session_state.frissites_ideje}")

    if "db_mentes_uzenet" in st.session_state:
        siker, uzenet = st.session_state.db_mentes_uzenet
        if uzenet:
            if siker:
                st.markdown(f"""
                <div style="background:#0a1628; border:1px solid #10b981; border-radius:8px;
                            padding:8px 16px; margin-bottom:12px; text-align:center;">
                    <span style="color:#10b981; font-weight:600; font-family:Montserrat,sans-serif;">
                    💾 Adatbázis: {uzenet}</span>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background:#0a1628; border:1px solid #FF6600; border-radius:8px;
                            padding:8px 16px; margin-bottom:12px; text-align:center;">
                    <span style="color:#FF6600; font-weight:600; font-family:Montserrat,sans-serif;">
                    ⚠️ Adatbázis: {uzenet}</span>
                </div>""", unsafe_allow_html=True)

    if "eredmenyek" in st.session_state:
        eredmenyek = st.session_state.eredmenyek
        eur_huf = st.session_state.eur_huf
        dam_ar_1nap = st.session_state.dam_ar_1nap
        dam_atlag_30 = st.session_state.dam_atlag_30
        dam_valodi = st.session_state.get("dam_valodi", False)

        datumok = [e["datum"].strftime("%Y-%m-%d") for e in eredmenyek]
        fogyasztasok_gwh = [e["fogyasztas"] * 24 / 1000 for e in eredmenyek]
        homersekletek = [e["homerseklet"] for e in eredmenyek]
        modellek_lista = [e["modell"] for e in eredmenyek]
        koltsegek = [e["fogyasztas"] * 24 * e["dam_ar"] * eur_huf / 1_000_000 for e in eredmenyek]
        riadok = [e["riado"] for e in eredmenyek]

        heti_fogyasztas_gwh = sum(fogyasztasok_gwh)
        heti_koltseg = sum(koltsegek)
        max_idx = max(range(len(eredmenyek)), key=lambda i: eredmenyek[i]["fogyasztas"])
        min_idx = min(range(len(eredmenyek)), key=lambda i: eredmenyek[i]["fogyasztas"])
        max_nap = eredmenyek[max_idx]
        min_nap = eredmenyek[min_idx]
        max_fog_gwh = fogyasztasok_gwh[max_idx]
        min_fog_gwh = fogyasztasok_gwh[min_idx]
        riado_napok = [e for e in eredmenyek if e["riado"]]
        v1 = sum(1 for e in eredmenyek if e["modell"] == "V1")
        v2 = sum(1 for e in eredmenyek if e["modell"] == "V2")
        kuszob_gwh = RIADOKUSZOB * 24 / 1000

        if v2 > 0:
            modell_szin = "#FF6600"
            modell_szoveg = f"Aktív modell: V2 — {v2} nap (extrém hideg) | V1 — {v1} nap"
        else:
            modell_szin = "#10b981"
            modell_szoveg = f"Aktív modell: V1 — mind a {v1} nap (normál időjárás)"

        st.markdown(f"""
        <div style="background:#0a1628; border:1px solid #1e3a5f; border-radius:8px;
                    padding:8px 16px; margin-bottom:12px; text-align:center;">
            <span style="color:{modell_szin}; font-weight:600; font-family:Montserrat,sans-serif;">
            {modell_szoveg}</span>
        </div>""", unsafe_allow_html=True)

        elso3_fog = sum(fogyasztasok_gwh[:3]) / 3
        utolso3_fog = sum(fogyasztasok_gwh[-3:]) / 3
        trend_fog = "▲" if utolso3_fog > elso3_fog else "▼"
        trend_fog_szin = "#FF6600" if utolso3_fog > elso3_fog else "#10b981"

        elso3_kolt = sum(koltsegek[:3]) / 3
        utolso3_kolt = sum(koltsegek[-3:]) / 3
        trend_koltseg = "▲" if utolso3_kolt > elso3_kolt else "▼"
        trend_koltseg_szin = "#FF6600" if utolso3_kolt > elso3_kolt else "#10b981"

        trend_dam = "▲" if dam_ar_1nap > dam_atlag_30 else "▼"
        trend_dam_szin = "#FF6600" if dam_ar_1nap > dam_atlag_30 else "#10b981"

        riado_ertek = ", ".join([e["datum"].strftime("%m.%d") for e in riado_napok]) if riado_napok else "Nincs"
        riado_szin = "#FF6600" if riado_napok else "#10b981"
        riado_ikon = "🚨" if riado_napok else "✅"

        fog_valt = ((utolso3_fog - elso3_fog) / elso3_fog) * 100
        kolt_valt = ((utolso3_kolt - elso3_kolt) / elso3_kolt) * 100
        dam_valt = ((dam_ar_1nap - dam_atlag_30) / dam_atlag_30) * 100

        tooltip_fog = f"Heti fogyasztás trend: első 3 nap átlaga ({elso3_fog:.1f} GWh) vs utolsó 3 nap átlaga ({utolso3_fog:.1f} GWh), változás: {fog_valt:+.1f}%"
        tooltip_kolt = f"Heti költség trend: első 3 nap átlaga ({elso3_kolt:.0f} M Ft) vs utolsó 3 nap átlaga ({utolso3_kolt:.0f} M Ft), változás: {kolt_valt:+.1f}%"
        tooltip_dam = f"DAM ár vs 30 napos átlag: holnapi ({dam_ar_1nap:.2f}) vs 30 napos ({dam_atlag_30:.2f} EUR/MWh), változás: {dam_valt:+.1f}%"

        st.markdown(f"""
        <div class="kartya-sor">
          <div class="kartya">
            <div class="kartya-cim">⚡ Heti fogyasztás</div>
            <div class="kartya-ertek">{heti_fogyasztas_gwh:.1f} GWh</div>
            <div class="kartya-trend" style="color:{trend_fog_szin};" title="{tooltip_fog}">{trend_fog} heti trend</div>
          </div>
          <div class="kartya">
            <div class="kartya-cim">💰 Heti költség</div>
            <div class="kartya-ertek">{heti_koltseg/1000:.1f} Mrd Ft</div>
            <div class="kartya-trend" style="color:{trend_koltseg_szin};" title="{tooltip_kolt}">{trend_koltseg} heti trend</div>
          </div>
          <div class="kartya">
            <div class="kartya-cim">📈 Csúcs</div>
            <div class="kartya-ertek">{max_fog_gwh:.1f} GWh</div>
            <div class="kartya-sub">{max_nap['datum'].strftime('%m.%d')}</div>
          </div>
          <div class="kartya">
            <div class="kartya-cim">📉 Minimum</div>
            <div class="kartya-ertek">{min_fog_gwh:.1f} GWh</div>
            <div class="kartya-sub">{min_nap['datum'].strftime('%m.%d')}</div>
          </div>
          <div class="kartya">
            <div class="kartya-cim">🏦 DAM valódi</div>
            <div class="kartya-ertek">{dam_ar_1nap:.2f} EUR/MWh</div>
            <div class="kartya-trend" style="color:{trend_dam_szin};" title="{tooltip_dam}">{trend_dam} vs 30 napos</div>
          </div>
          <div class="kartya">
            <div class="kartya-cim">🏦 DAM 30 napos</div>
            <div class="kartya-ertek">{dam_atlag_30:.2f} EUR/MWh</div>
            <div class="kartya-sub">30 napos átlag</div>
          </div>
          <div class="kartya">
            <div class="kartya-cim">💱 EUR/HUF</div>
            <div class="kartya-ertek">{eur_huf:.1f} Ft</div>
            <div class="kartya-sub">MNB középárfolyam</div>
          </div>
          <div class="kartya" style="border-color:{'#FF6600' if riado_napok else '#1e3a5f'};">
            <div class="kartya-cim">{riado_ikon} Riasztás</div>
            <div class="kartya-ertek" style="color:{riado_szin};">{riado_ertek}</div>
            <div class="kartya-sub">küszöb: {kuszob_gwh:.1f} GWh</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div id='grafikon-fogyasztas' style='margin-top:12px;'></div>", unsafe_allow_html=True)

        gt1, gt2, gt3, gt4 = st.tabs([
            "📊 Fogyasztás", "💰 Költség", "🌡️ Hőmérséklet", "📈 Összes grafikon"])

        with gt1:
            components.html(
                fogyasztas_chart(datumok, fogyasztasok_gwh, modellek_lista, riadok, eur_huf),
                height=440, scrolling=False)

        with gt2:
            components.html(
                koltseg_chart(datumok, koltsegek, eur_huf),
                height=440, scrolling=False)

        with gt3:
            components.html(
                homerseklet_chart(datumok, homersekletek),
                height=440, scrolling=False)

        with gt4:
            c1, c2, c3 = st.columns(3)
            with c1:
                components.html(
                    fogyasztas_chart(datumok, fogyasztasok_gwh, modellek_lista, riadok, eur_huf, height=300),
                    height=320, scrolling=False)
            with c2:
                components.html(
                    koltseg_chart(datumok, koltsegek, eur_huf, height=300),
                    height=320, scrolling=False)
            with c3:
                components.html(
                    homerseklet_chart(datumok, homersekletek, height=300),
                    height=320, scrolling=False)
    else:
        st.info("Kattints az Előrejelzés frissítése gombra!")

with tab2:
    st.markdown("### 📅 Korábbi jóslatok az adatbázisból")
    
    elerheto_datumok = db_osszes_datum()
    
    if not elerheto_datumok:
        st.info("ℹ️ Még nincsenek mentett jóslatok az adatbázisban. Az automatikus mentés akkor indul, "
                "amikor a 'Napi Predikció' tab-on új jóslat készül.")
    else:
        st.markdown(f"**{len(elerheto_datumok)} jóslási nap** elérhető az adatbázisban.")
        col_d, col_b = st.columns([2, 1])
        with col_d:
            valasztott_datum = st.selectbox(
                "Válassz jóslási dátumot:",
                options=elerheto_datumok,
                label_visibility="collapsed",
                key="datum_selectbox"
            )
        with col_b:
            betoltes = st.button("📂 Betöltés", key="betoltes_gomb")
        
        if betoltes and valasztott_datum:
            sorok, hiba = olvas_db_bol(valasztott_datum)
            if hiba:
                st.error(f"Hiba: {hiba}")
            elif not sorok:
                st.warning("Nincs adat erre a dátumra.")
            else:
                st.success(f"✅ {len(sorok)} sor betöltve a {valasztott_datum} jóslásból")
                
                df = pd.DataFrame(sorok)
                df = df.sort_values("cel_datuma")
                
                datumok_t = [str(d) for d in df["cel_datuma"].tolist()]
                fogy_t_gwh = [float(v) * 24 / 1000 for v in df["fogyasztas_mwh"].tolist()]
                hom_t = [float(v) for v in df["homerseklet"].tolist()]
                kolt_t = [float(v) for v in df["koltseg_mft"].tolist()]
                modellek_t = df["modell"].tolist()
                riadok_t = [bool(v) for v in df["riado"].tolist()]
                eur_huf_t = float(df["eur_huf"].iloc[0])
                
                st.markdown(f"""
                <div style="background:#0a1628; border:1px solid #1e3a5f; border-radius:8px;
                            padding:12px 16px; margin:12px 0;">
                    <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:12px;">
                        <div style="text-align:center;">
                            <div style="color:#64748b; font-size:10px; text-transform:uppercase;">Jóslás dátuma</div>
                            <div style="color:#FF6600; font-size:18px; font-weight:700;">{valasztott_datum}</div>
                        </div>
                        <div style="text-align:center;">
                            <div style="color:#64748b; font-size:10px; text-transform:uppercase;">Heti fogyasztás</div>
                            <div style="color:#FF6600; font-size:18px; font-weight:700;">{sum(fogy_t_gwh):.1f} GWh</div>
                        </div>
                        <div style="text-align:center;">
                            <div style="color:#64748b; font-size:10px; text-transform:uppercase;">Heti költség</div>
                            <div style="color:#FF6600; font-size:18px; font-weight:700;">{sum(kolt_t)/1000:.1f} Mrd Ft</div>
                        </div>
                        <div style="text-align:center;">
                            <div style="color:#64748b; font-size:10px; text-transform:uppercase;">EUR/HUF (akkor)</div>
                            <div style="color:#FF6600; font-size:18px; font-weight:700;">{eur_huf_t:.1f} Ft</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                tt1, tt2, tt3 = st.tabs(["📊 Fogyasztás", "💰 Költség", "🌡️ Hőmérséklet"])
                with tt1:
                    components.html(
                        fogyasztas_chart(datumok_t, fogy_t_gwh, modellek_t, riadok_t, eur_huf_t),
                        height=440, scrolling=False)
                with tt2:
                    components.html(
                        koltseg_chart(datumok_t, kolt_t, eur_huf_t),
                        height=440, scrolling=False)
                with tt3:
                    components.html(
                        homerseklet_chart(datumok_t, hom_t),
                        height=440, scrolling=False)
                
                st.markdown("### 📋 Részletes adatok")
                df_display = df[["cel_datuma", "fogyasztas_mwh", "homerseklet", 
                                 "dam_ar", "koltseg_mft", "modell", "riado"]].copy()
                df_display.columns = ["Dátum", "Fogyasztás (MWh, óra-átlag)", "Hőmérséklet (°C)",
                                     "DAM ár (EUR/MWh)", "Költség (M Ft)", "Modell", "Riasztás"]
                st.dataframe(df_display, use_container_width=True, hide_index=True)
