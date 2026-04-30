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
</style>
""", unsafe_allow_html=True)

RIADOKUSZOB = 6812.0
LAG_KEZDO = 4796.52
hu_holidays = holidays.Hungary(years=[2026, 2027])
ENTSOE_API_KEY = os.environ.get("ENTSOE_API_KEY", "")
BASE = os.path.dirname(os.path.abspath(__file__))

# ── MODELL DEFINÍCIÓ ─────────────────────────────────────────────
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

# ── API LEKÉRÉSEK ────────────────────────────────────────────────
@st.cache_data(ttl=300)
def get_eur_huf():
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

@st.cache_data(ttl=3600)
def get_idojaras():
    try:
        ma = datetime.now()
        url = "https://api.open-meteo.com/v1/forecast"
        params = {"latitude": 47.5, "longitude": 19.0,
                  "daily": "temperature_2m_mean",
                  "timezone": "Europe/Budapest",
                  "start_date": ma.strftime("%Y-%m-%d"),
                  "end_date": (ma + timedelta(days=6)).strftime("%Y-%m-%d")}
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        return [{"datum": pd.to_datetime(d), "homerseklet": t}
                for d, t in zip(data["daily"]["time"], data["daily"]["temperature_2m_mean"])]
    except:
        ma = datetime.now()
        return [{"datum": pd.to_datetime(ma + timedelta(days=i)), "homerseklet": 12.0}
                for i in range(7)]

@st.cache_data(ttl=300)
def get_dam_ar():
    # FIX: jelezzük ha nincs API kulcs
    if not ENTSOE_API_KEY:
        return 104.93, 98.91, False  # (1nap, 30nap, valodi_adat)
    try:
        client = EntsoePandasClient(api_key=ENTSOE_API_KEY)
        ma = datetime.now()
        legfrissebb = 104.93
        for delta in [-1, -2, -3]:
            try:
                nap = ma + timedelta(days=delta)
                start = pd.Timestamp(nap.strftime("%Y-%m-%d"), tz="Europe/Budapest")
                end = start + pd.Timedelta(days=1)
                dam = client.query_day_ahead_prices("HU", start=start, end=end)
                if dam is not None and len(dam) > 0:
                    legfrissebb = float(dam.mean())
                    break
            except:
                continue
        start_30 = pd.Timestamp((ma - timedelta(days=30)).strftime("%Y-%m-%d"), tz="Europe/Budapest")
        end_30 = pd.Timestamp(ma.strftime("%Y-%m-%d"), tz="Europe/Budapest")
        dam_30 = client.query_day_ahead_prices("HU", start=start_30, end=end_30)
        atlag_30 = float(dam_30.mean()) if dam_30 is not None and len(dam_30) > 0 else 98.91
        return legfrissebb, atlag_30, True
    except:
        return 104.93, 98.91, False

# ── PREDIKCIÓ ────────────────────────────────────────────────────
def ensemble_joslas(idojaras_lista, dam_ar_1nap, dam_atlag_30):
    eredmenyek = []
    lag = LAG_KEZDO
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

# ── CHART SEGÉDFÜGGVÉNY ───────────────────────────────────────────
def plotly_chart(adatok_json, layout_json, chart_id, height=420):
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdn.plot.ly/plotly-2.26.0.min.js" charset="utf-8"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#0a1628; overflow:hidden; }}
#{chart_id} {{ width:100%; height:{height}px; opacity:0; transition: opacity 0.3s ease; }}
</style>
</head>
<body>
<div id="{chart_id}"></div>
<script>
(function() {{
    var finalData = {adatok_json};
    var layout = {layout_json};
    var config = {{responsive: true, displayModeBar: false}};

    // Nullázott kezdőadatok az animációhoz
    var startData = finalData.map(function(trace) {{
        var t = JSON.parse(JSON.stringify(trace));
        if (t.y) t.y = t.y.map(function() {{ return 0; }});
        if (t.marker && t.marker.opacity !== undefined) t.marker.opacity = 0;
        return t;
    }});

    // Először nullával rajzoljuk ki
    Plotly.newPlot('{chart_id}', startData, layout, config).then(function() {{
        // Láthatóvá tesszük
        document.getElementById('{chart_id}').style.opacity = '1';

        // Kis késleltetés után beanimálódnak a valódi értékek
        setTimeout(function() {{
            Plotly.react('{chart_id}', finalData, layout, config);
        }}, 150);
    }});
}})();
</script>
</body>
</html>"""

# ── CHART FÜGGVÉNYEK ─────────────────────────────────────────────
def fogyasztas_chart(datumok, fogyasztasok, modellek, riadok, eur_huf, height=420):
    # FIX: "#ffffff" hexkód a "white" helyett
    colors = ["#FF6600" if r else "#0066CC" for r in riadok]
    feliratok = [f"{v:,.0f} ({m})".replace(",", " ") for v, m in zip(fogyasztasok, modellek)]
    
    adatok = json.dumps([{
        "type": "bar",
        "x": datumok,
        "y": fogyasztasok,
        "marker": {"color": colors, "opacity": 0.9},
        "text": feliratok,
        "textposition": "outside",
        "textfont": {"color": "#ffffff", "size": 11},
        "hovertemplate": "%{x}<br>%{text}<extra></extra>",
        "name": "Fogyasztás"
    }])
    
    layout = json.dumps({
        "paper_bgcolor": "#0a1628",
        "plot_bgcolor": "#0f2040",
        "font": {"color": "#cbd5e1", "family": "Inter"},
        "title": {"text": "7 napos fogyasztás előrejelzés (MWh)",
                  "font": {"size": 14, "color": "#f1f5f9"}},
        "margin": {"l": 60, "r": 20, "t": 50, "b": 50},
        "xaxis": {"gridcolor": "#1e3a5f", "tickformat": "%m.%d", "color": "#cbd5e1"},
        "yaxis": {"gridcolor": "#1e3a5f", "title": "MWh", "color": "#cbd5e1"},
        "bargap": 0.55,
        "showlegend": False,
        "shapes": [{"type": "line", "x0": datumok[0], "x1": datumok[-1],
                    "y0": RIADOKUSZOB, "y1": RIADOKUSZOB,
                    "line": {"color": "#FF6600", "width": 2, "dash": "dash"}}],
        "annotations": [{"x": datumok[3], "y": RIADOKUSZOB,
                         "text": "Riasztási küszöb (6 812 MWh)",
                         "showarrow": False, "font": {"color": "#FF6600", "size": 10},
                         "yanchor": "bottom"}]
    })
    return plotly_chart(adatok, layout, "fogyasztas", height)

def koltseg_chart(datumok, koltsegek, eur_huf, height=420):
    adatok = json.dumps([{
        "type": "scatter",
        "x": datumok,
        "y": koltsegek,
        "mode": "lines+markers+text",
        "line": {"color": "#FF6600", "width": 3},
        "marker": {"size": 10, "color": "#FF6600", "line": {"color": "#ffffff", "width": 2}},
        "fill": "tozeroy",
        "fillcolor": "rgba(255,102,0,0.15)",
        "text": [f"{v:.1f}" for v in koltsegek],
        "textposition": "top center",
        "textfont": {"color": "#ffffff", "size": 10},
        "hovertemplate": "%{x}<br>%{y:.1f} M Ft<extra></extra>",
        "name": "Költség"
    }])
    
    layout = json.dumps({
        "paper_bgcolor": "#0a1628",
        "plot_bgcolor": "#0f2040",
        "font": {"color": "#cbd5e1", "family": "Inter"},
        "title": {"text": f"Becsült napi energiaköltség (M Ft) | EUR/HUF: {eur_huf:.1f}",
                  "font": {"size": 14, "color": "#f1f5f9"}},
        "margin": {"l": 60, "r": 20, "t": 50, "b": 50},
        "xaxis": {"gridcolor": "#1e3a5f", "tickformat": "%m.%d", "color": "#cbd5e1"},
        "yaxis": {"gridcolor": "#1e3a5f", "title": "Millió Ft", "color": "#cbd5e1",
                  "range": [min(koltsegek)*0.95, max(koltsegek)*1.08]},
        "showlegend": False
    })
    return plotly_chart(adatok, layout, "koltseg", height)

def homerseklet_chart(datumok, homersekletek, height=420):
    adatok = json.dumps([{
        "type": "scatter",
        "x": datumok,
        "y": homersekletek,
        "mode": "lines+markers+text",
        "line": {"color": "#10b981", "width": 3},
        "marker": {"size": 10, "color": "#10b981", "line": {"color": "#ffffff", "width": 2}},
        "fill": "tozeroy",
        "fillcolor": "rgba(16,185,129,0.15)",
        "text": [f"{v:.1f}°" for v in homersekletek],
        "textposition": "top center",
        "textfont": {"color": "#ffffff", "size": 10},
        "hovertemplate": "%{x}<br>%{y:.1f}°C<extra></extra>",
        "name": "Hőmérséklet"
    }])
    
    layout = json.dumps({
        "paper_bgcolor": "#0a1628",
        "plot_bgcolor": "#0f2040",
        "font": {"color": "#cbd5e1", "family": "Inter"},
        "title": {"text": "Hőmérséklet előrejelzés (°C)",
                  "font": {"size": 14, "color": "#f1f5f9"}},
        "margin": {"l": 60, "r": 20, "t": 50, "b": 50},
        "xaxis": {"gridcolor": "#1e3a5f", "tickformat": "%m.%d", "color": "#cbd5e1"},
        "yaxis": {"gridcolor": "#1e3a5f", "title": "°C", "color": "#cbd5e1"},
        "showlegend": False,
        "shapes": [{"type": "line", "x0": datumok[0], "x1": datumok[-1],
                    "y0": -5, "y1": -5,
                    "line": {"color": "#FF6600", "width": 1.5, "dash": "dot"}}],
        "annotations": [{"x": datumok[-1], "y": -5,
                         "text": "Extrém hideg (-5°C)",
                         "showarrow": False, "font": {"color": "#FF6600", "size": 10},
                         "yanchor": "top", "xanchor": "right"}]
    })
    return plotly_chart(adatok, layout, "homerseklet", height)

# ── FEJLÉC ────────────────────────────────────────────────────
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
                CrewAI + PyTorch Ensemble modell &nbsp;|&nbsp; 7 napos előrejelzés &nbsp;|&nbsp; V1 + V2
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

    # FIX: gombnyomás pontos ideje mentve
    if frissit:
        st.session_state.frissites_ideje = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        get_eur_huf.clear()
        get_idojaras.clear()
        get_dam_ar.clear()
        for key in ["eredmenyek", "eur_huf", "dam_ar_1nap", "dam_atlag_30", "dam_valodi"]:
            st.session_state.pop(key, None)
        st.rerun()

    if "eredmenyek" not in st.session_state:
        with st.spinner("Adatok lekérése..."):
            eur_huf = get_eur_huf()
            idojaras_lista = get_idojaras()
            dam_ar_1nap, dam_atlag_30, dam_valodi = get_dam_ar()
            eredmenyek = ensemble_joslas(idojaras_lista, dam_ar_1nap, dam_atlag_30)
            st.session_state.eredmenyek = eredmenyek
            st.session_state.eur_huf = eur_huf
            st.session_state.dam_ar_1nap = dam_ar_1nap
            st.session_state.dam_atlag_30 = dam_atlag_30
            st.session_state.dam_valodi = dam_valodi
        allapot_ph.success(f"✅ Frissítve: {st.session_state.get('frissites_ideje', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}")

    if "eredmenyek" in st.session_state:
        eredmenyek = st.session_state.eredmenyek
        eur_huf = st.session_state.eur_huf
        dam_ar_1nap = st.session_state.dam_ar_1nap
        dam_atlag_30 = st.session_state.dam_atlag_30
        dam_valodi = st.session_state.get("dam_valodi", False)



        datumok = [e["datum"].strftime("%Y-%m-%d") for e in eredmenyek]
        fogyasztasok = [e["fogyasztas"] for e in eredmenyek]
        homersekletek = [e["homerseklet"] for e in eredmenyek]
        modellek_lista = [e["modell"] for e in eredmenyek]
        koltsegek = [e["fogyasztas"] * e["dam_ar"] * eur_huf / 1_000_000 for e in eredmenyek]
        riadok = [e["riado"] for e in eredmenyek]

        heti_fogyasztas = sum(fogyasztasok)
        heti_koltseg = sum(koltsegek)
        max_nap = max(eredmenyek, key=lambda e: e["fogyasztas"])
        min_nap = min(eredmenyek, key=lambda e: e["fogyasztas"])
        riado_napok = [e for e in eredmenyek if e["riado"]]
        v1 = sum(1 for e in eredmenyek if e["modell"] == "V1")
        v2 = sum(1 for e in eredmenyek if e["modell"] == "V2")

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
            🤖 {modell_szoveg}</span>
        </div>""", unsafe_allow_html=True)

        # Trend számítás (első 3 nap vs utolsó 3 nap átlaga)
        trend_fog = "▲" if fogyasztasok[-1] > fogyasztasok[0] else "▼"
        trend_fog_szin = "#FF6600" if fogyasztasok[-1] > fogyasztasok[0] else "#10b981"
        trend_koltseg = "▲" if koltsegek[-1] > koltsegek[0] else "▼"
        trend_koltseg_szin = "#FF6600" if koltsegek[-1] > koltsegek[0] else "#10b981"
        trend_dam = "▲" if dam_ar_1nap > dam_atlag_30 else "▼"
        trend_dam_szin = "#FF6600" if dam_ar_1nap > dam_atlag_30 else "#10b981"
        trend_huf = "▲" if eur_huf > 395 else "▼"
        trend_huf_szin = "#FF6600" if eur_huf > 395 else "#10b981"

        riado_ertek = ", ".join([e["datum"].strftime("%m.%d") for e in riado_napok]) if riado_napok else "Nincs"
        riado_szin = "#FF6600" if riado_napok else "#10b981"
        riado_ikon = "🚨" if riado_napok else "✅"

        st.markdown(f"""
        <style>
        .kartya {{
            flex:1; background:#0a1628; border:1px solid #1e3a5f; border-radius:12px;
            padding:14px 12px; text-align:center; min-width:0;
            box-shadow:0 2px 12px rgba(0,102,204,0.15);
            cursor:pointer; transition: border-color 0.2s, box-shadow 0.2s, transform 0.15s;
            text-decoration:none; display:block;
        }}
        .kartya:hover {{
            border-color:#0066CC; box-shadow:0 4px 20px rgba(0,102,204,0.35);
            transform: translateY(-2px);
        }}
        .kartya-cim {{
            color:#64748b; font-size:9px; font-family:Inter,sans-serif;
            letter-spacing:1px; text-transform:uppercase; margin-bottom:6px;
        }}
        .kartya-ertek {{
            color:#FF6600; font-size:16px; font-weight:700;
            font-family:Montserrat,sans-serif;
        }}
        .kartya-trend {{ font-size:13px; margin-top:4px; }}
        .kartya-sub {{ color:#94a3b8; font-size:11px; margin-top:4px; }}
        </style>

        <div style="display:flex; gap:10px; margin-bottom:16px; flex-wrap:nowrap;">

          <a class="kartya" href="#grafikon-fogyasztas" onclick="navigalj('tab-fogyasztas')">
            <div class="kartya-cim">⚡ Heti fogyasztás</div>
            <div class="kartya-ertek">{heti_fogyasztas:,.0f} MWh</div>
            <div class="kartya-trend" style="color:{trend_fog_szin};">{trend_fog} heti trend</div>
          </a>

          <a class="kartya" href="#grafikon-koltseg" onclick="navigalj('tab-koltseg')">
            <div class="kartya-cim">💰 Heti költség</div>
            <div class="kartya-ertek">{heti_koltseg:.1f} M Ft</div>
            <div class="kartya-trend" style="color:{trend_koltseg_szin};">{trend_koltseg} heti trend</div>
          </a>

          <a class="kartya" href="#grafikon-fogyasztas" onclick="navigalj('tab-fogyasztas')">
            <div class="kartya-cim">📈 Csúcs</div>
            <div class="kartya-ertek">{max_nap['fogyasztas']:,.0f} MWh</div>
            <div class="kartya-sub">{max_nap['datum'].strftime('%m.%d')}</div>
          </a>

          <a class="kartya" href="#grafikon-fogyasztas" onclick="navigalj('tab-fogyasztas')">
            <div class="kartya-cim">📉 Minimum</div>
            <div class="kartya-ertek">{min_nap['fogyasztas']:,.0f} MWh</div>
            <div class="kartya-sub">{min_nap['datum'].strftime('%m.%d')}</div>
          </a>

          <a class="kartya" href="#grafikon-koltseg" onclick="navigalj('tab-koltseg')">
            <div class="kartya-cim">🏦 DAM valódi</div>
            <div class="kartya-ertek">{dam_ar_1nap:.2f} EUR/MWh</div>
            <div class="kartya-trend" style="color:{trend_dam_szin};">{trend_dam} vs 30 napos</div>
          </a>

          <a class="kartya" href="#grafikon-koltseg" onclick="navigalj('tab-koltseg')">
            <div class="kartya-cim">🏦 DAM 30 napos</div>
            <div class="kartya-ertek">{dam_atlag_30:.2f} EUR/MWh</div>
            <div class="kartya-sub">30 napos átlag</div>
          </a>

          <a class="kartya" href="#grafikon-koltseg" onclick="navigalj('tab-koltseg')">
            <div class="kartya-cim">💱 EUR/HUF</div>
            <div class="kartya-ertek">{eur_huf:.1f} Ft</div>
            <div class="kartya-trend" style="color:{trend_huf_szin};">{trend_huf} vs 395 alap</div>
          </a>

          <a class="kartya" href="#grafikon-fogyasztas" onclick="navigalj('tab-fogyasztas')"
             style="border-color:{'#FF6600' if riado_napok else '#1e3a5f'};
                    box-shadow:0 2px 12px rgba({'255,102,0' if riado_napok else '0,102,204'},0.2);">
            <div class="kartya-cim">{riado_ikon} Riasztás</div>
            <div class="kartya-ertek" style="color:{riado_szin};">{riado_ertek}</div>
            <div class="kartya-sub">küszöb: 6 812 MWh</div>
          </a>

        </div>

        <script>
        function navigalj(tabId) {{
            // Megkeresi a tab gombokat és rákattint a megfelelőre
            setTimeout(function() {{
                var tabs = window.parent.document.querySelectorAll('[data-baseweb="tab"]');
                var celTab = null;
                tabs.forEach(function(tab) {{
                    var szoveg = tab.textContent.toLowerCase();
                    if (tabId === 'tab-fogyasztas' && szoveg.includes('fogyaszt')) celTab = tab;
                    if (tabId === 'tab-koltseg' && szoveg.includes('lts')) celTab = tab;
                    if (tabId === 'tab-homerseklet' && szoveg.includes('m')) celTab = tab;
                }});
                if (celTab) {{
                    celTab.click();
                    celTab.scrollIntoView({{behavior: 'smooth', block: 'start'}});
                }}
            }}, 100);
        }}
        </script>
        """, unsafe_allow_html=True)

        st.markdown("<div id='grafikon-fogyasztas' style='margin-top:12px;'></div>", unsafe_allow_html=True)

        gt1, gt2, gt3, gt4 = st.tabs([
            "📊 Fogyasztás", "💰 Költség", "🌡️ Hőmérséklet", "📈 Összes grafikon"])

        with gt1:
            components.html(
                fogyasztas_chart(datumok, fogyasztasok, modellek_lista, riadok, eur_huf),
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
                    fogyasztas_chart(datumok, fogyasztasok, modellek_lista, riadok, eur_huf, height=300),
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
    st.markdown("### Válassz egy korábbi dátumot")
    col_d, col_b = st.columns([2, 1])
    with col_d:
        datum_str = st.text_input("Dátum (ÉÉÉÉ-HH-NN)",
                                   placeholder="pl. 2026-04-28",
                                   label_visibility="collapsed",
                                   key="datum_input")
    with col_b:
        betoltes = st.button("📂 Betöltés", key="betoltes_gomb")
    if betoltes and datum_str:
        # FIX: informatívabb üzenet a mentési funkció hiányáról
        st.info("ℹ️ A korábbi jóslatok mentési funkciója még fejlesztés alatt áll. "
                "Ehhez egy adatbázis kapcsolat (pl. SQLite / CSV log) szükséges.")
