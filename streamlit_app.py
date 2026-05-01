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

# ── GLOBÁLIS STÍLUSOK + KÁRTYA STÍLUSOK EGY HELYEN (FIX: szellem kártyák) ──
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

/* ── KÁRTYA STÍLUSOK (CSS GRID, fix 8 oszlop, hover effekttel) ── */
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


@media (max-width: 1200px) {
    .kartya-sor { grid-template-columns: repeat(4, 1fr); }
}
@media (max-width: 640px) {
    .kartya-sor { grid-template-columns: repeat(2, 1fr); }
}
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

# ── AKTUÁLIS DÁTUM SEGÉDFÜGGVÉNY (FIX: magyar időzóna) ────────────
def magyar_ma():
    """Mindig a magyar időzóna szerinti aktuális dátumot adja vissza."""
    return pd.Timestamp.now(tz="Europe/Budapest").normalize().tz_localize(None).to_pydatetime()

# ── API LEKÉRÉSEK ────────────────────────────────────────────────
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
    """FIX: datum_kulcs paraméter beépül a cache kulcsba → óránként új lekérés."""
    try:
        ma = magyar_ma()
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
        ma = magyar_ma()
        return [{"datum": pd.to_datetime(ma + timedelta(days=i)), "homerseklet": 12.0}
                for i in range(7)]

@st.cache_data(ttl=300)
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

# ── CHART SEGÉDFÜGGVÉNY (látványos animáció: 2.5sec, 500ms delay, fade-in) ──
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
    var chartEl = document.getElementById('{chart_id}');
    var animated = false;

    // ── BEÁLLÍTÁSOK ──
    var START_DELAY = 500;       // 500ms várakozás indítás előtt
    var ANIM_DURATION = 2500;    // 2.5 másodperc animáció
    var FRAME_RATE = 60;         // 60 fps
    var TOTAL_STEPS = Math.round(ANIM_DURATION * FRAME_RATE / 1000);  // ~150 lépés
    var FRAME_INTERVAL = 1000 / FRAME_RATE;  // ~16.67ms

    // Nullázott kezdőadatok (minden Y érték = 0)
    function makeStartData() {{
        return finalData.map(function(trace) {{
            var t = JSON.parse(JSON.stringify(trace));
            if (t.y) t.y = t.y.map(function() {{ return 0; }});
            // A szövegfeliratokat is rejtjük el animáció közben
            if (t.text) {{
                t._origText = t.text;
                t.text = t.text.map(function() {{ return ''; }});
            }}
            return t;
        }});
    }}

    // ── EASING FÜGGVÉNY: easeOutBack (kicsi túllendülés a végén) ──
    function easeOutBack(t) {{
        var c1 = 1.70158;
        var c3 = c1 + 1;
        return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
    }}

    // ── easeOutCubic (smooth, no overshoot) — bar charthoz biztonságosabb ──
    function easeOutCubic(t) {{
        return 1 - Math.pow(1 - t, 3);
    }}

    // Animáció lépésenként
    function runAnimation() {{
        if (animated) return;
        animated = true;

        // Fade in a chart (opacity 0 → 1)
        chartEl.classList.add('fade-in');

        var current = 0;
        var timer = setInterval(function() {{
            current++;
            var t = current / TOTAL_STEPS;
            var eased = easeOutCubic(t);

            var animData = finalData.map(function(trace) {{
                var copy = JSON.parse(JSON.stringify(trace));
                if (copy.y) copy.y = copy.y.map(function(v) {{ return v * eased; }});
                // A felirat csak akkor jelenjen meg, amikor majdnem kész (90%+)
                if (copy.text && t < 0.9) {{
                    copy.text = copy.text.map(function() {{ return ''; }});
                }}
                return copy;
            }});

            Plotly.react('{chart_id}', animData, layout, config);

            if (current >= TOTAL_STEPS) {{
                clearInterval(timer);
                // Végállapot: pontosan az eredeti adatok
                Plotly.react('{chart_id}', finalData, layout, config);
            }}
        }}, FRAME_INTERVAL);
    }}

    // ── INDÍTÁS ──
    Plotly.newPlot('{chart_id}', makeStartData(), layout, config).then(function() {{
        if ('IntersectionObserver' in window) {{
            var observer = new IntersectionObserver(function(entries) {{
                entries.forEach(function(entry) {{
                    if (entry.isIntersecting && !animated) {{
                        // 500ms delay, hogy a felhasználó észrevegye az indulást
                        setTimeout(runAnimation, START_DELAY);
                    }}
                }});
            }}, {{threshold: 0.1}});
            observer.observe(chartEl);
        }} else {{
            // Fallback régi böngészőkre
            setTimeout(runAnimation, START_DELAY);
        }}
    }});
}})();
</script>
</body>
</html>"""

# ── CHART FÜGGVÉNYEK ─────────────────────────────────────────────
def fogyasztas_chart(datumok, fogyasztasok, modellek, riadok, eur_huf, height=420):
    colors = ["#FF6600" if r else "#0066CC" for r in riadok]
    feliratok = [f"{v:,.0f} ({m})".replace(",", " ") for v, m in zip(fogyasztasok, modellek)]

    # FIX: 18 órás padding mindkét oldalon (konzisztens a többi chartal)
    x_min = (pd.to_datetime(datumok[0]) - pd.Timedelta(hours=18)).isoformat()
    x_max = (pd.to_datetime(datumok[-1]) + pd.Timedelta(hours=18)).isoformat()

    adatok = json.dumps([{
        "type": "bar",
        "x": datumok,
        "y": fogyasztasok,
        "marker": {"color": colors, "opacity": 0.9},
        "text": feliratok,
        "textposition": "outside",
        "textfont": {"color": "#ffffff", "size": 11},
        "hovertemplate": "%{x}<br>%{text}<extra></extra>",
        "name": "Fogyasztás",
        "cliponaxis": False
    }])

    layout = json.dumps({
        "paper_bgcolor": "#0a1628",
        "plot_bgcolor": "#0f2040",
        "font": {"color": "#cbd5e1", "family": "Inter"},
        "title": {"text": "7 napos fogyasztás előrejelzés (MWh)",
                  "font": {"size": 14, "color": "#f1f5f9"}},
        # FIX: konzisztens margin a többi chartal
        "margin": {"l": 100, "r": 80, "t": 70, "b": 50},
        "xaxis": {"gridcolor": "#1e3a5f", "tickformat": "%m.%d", "color": "#cbd5e1",
                  "range": [x_min, x_max], "type": "date"},
        "yaxis": {"gridcolor": "#1e3a5f", "title": "MWh", "color": "#cbd5e1",
                  "range": [0, max(fogyasztasok) * 1.25]},
        "bargap": 0.55,
        "showlegend": False,
        "shapes": [{"type": "line", "xref": "paper", "x0": 0, "x1": 1,
                    "y0": RIADOKUSZOB, "y1": RIADOKUSZOB,
                    "line": {"color": "#FF6600", "width": 2, "dash": "dash"}}],
        # FIX: xref:paper, így mindig középre kerül, nem lóg ki
        "annotations": [{"xref": "paper", "x": 0.5, "y": RIADOKUSZOB,
                         "text": "Riasztási küszöb (6 812 MWh)",
                         "showarrow": False, "font": {"color": "#FF6600", "size": 10},
                         "yanchor": "bottom",
                         "bgcolor": "#0a1628", "borderpad": 2}]
    })
    return plotly_chart(adatok, layout, "fogyasztas", height)

def koltseg_chart(datumok, koltsegek, eur_huf, height=420):
    # FIX: 18 órás padding mindkét oldalon (volt 12)
    x_min = (pd.to_datetime(datumok[0]) - pd.Timedelta(hours=18)).isoformat()
    x_max = (pd.to_datetime(datumok[-1]) + pd.Timedelta(hours=18)).isoformat()

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
        # FIX: minden pont top center (egységes), a margin és range gondoskodik a helyről
        "textposition": "top center",
        "textfont": {"color": "#ffffff", "size": 10},
        "hovertemplate": "%{x}<br>%{y:.1f} M Ft<extra></extra>",
        "name": "Költség",
        "cliponaxis": False
    }])

    y_min = min(koltsegek) * 0.80
    y_max = max(koltsegek) * 1.20

    layout = json.dumps({
        "paper_bgcolor": "#0a1628",
        "plot_bgcolor": "#0f2040",
        "font": {"color": "#cbd5e1", "family": "Inter"},
        "title": {"text": f"Becsült napi energiaköltség (M Ft) | EUR/HUF: {eur_huf:.1f}",
                  "font": {"size": 14, "color": "#f1f5f9"}},
        # FIX: bal/jobb margin szélesítve 80/30 → 100/80 (felirat befér)
        "margin": {"l": 100, "r": 80, "t": 70, "b": 50},
        "xaxis": {"gridcolor": "#1e3a5f", "tickformat": "%m.%d", "color": "#cbd5e1",
                  "range": [x_min, x_max], "type": "date"},
        "yaxis": {"gridcolor": "#1e3a5f", "title": "Millió Ft", "color": "#cbd5e1",
                  "range": [y_min, y_max]},
        "showlegend": False
    })
    return plotly_chart(adatok, layout, "koltseg", height)

def homerseklet_chart(datumok, homersekletek, height=420):
    # FIX: 18 órás padding mindkét oldalon
    x_min = (pd.to_datetime(datumok[0]) - pd.Timedelta(hours=18)).isoformat()
    x_max = (pd.to_datetime(datumok[-1]) + pd.Timedelta(hours=18)).isoformat()

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
        # FIX: minden pont top center, margin és range padding gondoskodik a helyről
        "textposition": "top center",
        "textfont": {"color": "#ffffff", "size": 10},
        "hovertemplate": "%{x}<br>%{y:.1f}°C<extra></extra>",
        "name": "Hőmérséklet",
        "cliponaxis": False
    }])

    layout = json.dumps({
        "paper_bgcolor": "#0a1628",
        "plot_bgcolor": "#0f2040",
        "font": {"color": "#cbd5e1", "family": "Inter"},
        "title": {"text": "Hőmérséklet előrejelzés (°C)",
                  "font": {"size": 14, "color": "#f1f5f9"}},
        # FIX: bal/jobb margin szélesítve 80/30 → 100/80
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

    if frissit:
        st.session_state.frissites_ideje = pd.Timestamp.now(tz="Europe/Budapest").strftime("%Y-%m-%d %H:%M:%S")
        get_eur_huf.clear()
        get_idojaras.clear()
        get_dam_ar.clear()
        for key in ["eredmenyek", "eur_huf", "dam_ar_1nap", "dam_atlag_30", "dam_valodi"]:
            st.session_state.pop(key, None)
        st.rerun()

    if "eredmenyek" not in st.session_state:
        with st.spinner("Adatok lekérése..."):
            # FIX: dátum-óra kulcs a cache-be → óránként új lekérés
            ora_kulcs = magyar_ma().strftime("%Y-%m-%d-%H")
            eur_huf = get_eur_huf(_datum_kulcs=ora_kulcs)
            idojaras_lista = get_idojaras(datum_kulcs=ora_kulcs)
            dam_ar_1nap, dam_atlag_30, dam_valodi = get_dam_ar(_datum_kulcs=ora_kulcs)
            eredmenyek = ensemble_joslas(idojaras_lista, dam_ar_1nap, dam_atlag_30)
            st.session_state.eredmenyek = eredmenyek
            st.session_state.eur_huf = eur_huf
            st.session_state.dam_ar_1nap = dam_ar_1nap
            st.session_state.dam_atlag_30 = dam_atlag_30
            st.session_state.dam_valodi = dam_valodi
            st.session_state.frissites_ideje = pd.Timestamp.now(tz="Europe/Budapest").strftime("%Y-%m-%d %H:%M:%S")

    if "frissites_ideje" in st.session_state:
        # FIX: Csak utolsó frissítés időpont, garantáltan magyar idő szerint
        allapot_ph.success(f"✅ Frissítve: {st.session_state.frissites_ideje} (magyar idő)")

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

        # ── TREND SZÁMÍTÁS (FIX: első 3 nap átlaga vs utolsó 3 nap átlaga) ──
        elso3_fog = sum(fogyasztasok[:3]) / 3
        utolso3_fog = sum(fogyasztasok[-3:]) / 3
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

        # Trend tooltipek számítása (egérrel ráhúzva mutatja a magyarázatot)
        fog_valt = ((utolso3_fog - elso3_fog) / elso3_fog) * 100
        kolt_valt = ((utolso3_kolt - elso3_kolt) / elso3_kolt) * 100
        dam_valt = ((dam_ar_1nap - dam_atlag_30) / dam_atlag_30) * 100

        tooltip_fog = f"Heti fogyasztás trend: első 3 nap átlaga ({elso3_fog:,.0f} MWh) vs utolsó 3 nap átlaga ({utolso3_fog:,.0f} MWh), változás: {fog_valt:+.1f}%"
        tooltip_kolt = f"Heti költség trend: első 3 nap átlaga ({elso3_kolt:.1f} M Ft) vs utolsó 3 nap átlaga ({utolso3_kolt:.1f} M Ft), változás: {kolt_valt:+.1f}%"
        tooltip_dam = f"DAM ár vs 30 napos átlag: holnapi ({dam_ar_1nap:.2f}) vs 30 napos ({dam_atlag_30:.2f} EUR/MWh), változás: {dam_valt:+.1f}%"

        # ── KÁRTYÁK (FIX: <div>-ek, hover effekt, trend tooltipek) ──
        st.markdown(f"""
        <div class="kartya-sor">

          <div class="kartya">
            <div class="kartya-cim">⚡ Heti fogyasztás</div>
            <div class="kartya-ertek">{heti_fogyasztas:,.0f} MWh</div>
            <div class="kartya-trend" style="color:{trend_fog_szin};" title="{tooltip_fog}">{trend_fog} heti trend</div>
          </div>

          <div class="kartya">
            <div class="kartya-cim">💰 Heti költség</div>
            <div class="kartya-ertek">{heti_koltseg:.1f} M Ft</div>
            <div class="kartya-trend" style="color:{trend_koltseg_szin};" title="{tooltip_kolt}">{trend_koltseg} heti trend</div>
          </div>

          <div class="kartya">
            <div class="kartya-cim">📈 Csúcs</div>
            <div class="kartya-ertek">{max_nap['fogyasztas']:,.0f} MWh</div>
            <div class="kartya-sub">{max_nap['datum'].strftime('%m.%d')}</div>
          </div>

          <div class="kartya">
            <div class="kartya-cim">📉 Minimum</div>
            <div class="kartya-ertek">{min_nap['fogyasztas']:,.0f} MWh</div>
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
            <div class="kartya-sub">küszöb: 6 812 MWh</div>
          </div>

        </div>
        """, unsafe_allow_html=True)

        # ── (A navigációs script eltávolítva — a kártyák most egyszerű
        # információs <div>-ek, nincs onclick, így nem kell JS sem) ──

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
        st.info("ℹ️ A korábbi jóslatok mentési funkciója még fejlesztés alatt áll. "
                "Ehhez egy adatbázis kapcsolat (pl. SQLite / CSV log) szükséges.")
