import streamlit as st
import streamlit.components.v1 as components
import joblib
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
    page_title="MVM Energiafogyasztás Előrejelző V3",
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

# === KONSTANSOK ===
hu_holidays = holidays.Hungary(years=[2026, 2027])
ENTSOE_API_KEY = os.environ.get("ENTSOE_API_KEY", "")
BASE = os.path.dirname(os.path.abspath(__file__))

# === MODELL ÉS STL BETÖLTÉS ===
@st.cache_resource
def modell_betoltese():
    model = joblib.load(f"{BASE}/xgb_model.pkl")
    return model

@st.cache_resource
def stl_betoltese():
    with open(f"{BASE}/stl_params.json", "r") as f:
        params = json.load(f)
    seasonal = pd.read_csv(f"{BASE}/stl_seasonal.csv", index_col=0, parse_dates=True)
    return params, seasonal

@st.cache_resource
def negativ_arak_betoltese():
    return pd.read_csv(f"{BASE}/negativ_arak_summary.csv")

model = modell_betoltese()
stl_params, stl_seasonal = stl_betoltese()
neg_summary = negativ_arak_betoltese()

FEATURE_SORREND = ['DAM_EUR_MWh', 'Homerseklet_C', 'Paratartalom_szazalek', 'Napsugarzas_W_m2',
                   'Szelsebesseg_kmh', 'Csapadek_mm', 'EUR_HUF', 'Ora', 'Het_napja', 'Honap',
                   'Unnepnap', 'Hetvege', 'Extrem_hideg', 'Extrem_meleg',
                   'Fogyasztas_lag1h', 'Fogyasztas_lag24h', 'Fogyasztas_lag168h',
                   'Nap_termeles_MW', 'Szel_termeles_MW']

def magyar_ma():
    return pd.Timestamp.now(tz="Europe/Budapest").normalize().tz_localize(None).to_pydatetime()

# === ADATLEKÉRŐ FÜGGVÉNYEK ===
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
def get_oras_idojaras(_datum_kulcs=None):
    try:
        ma = magyar_ma()
        url = "https://api.open-meteo.com/v1/forecast"
        params = {"latitude": 47.5, "longitude": 19.0,
                  "hourly": "temperature_2m,relative_humidity_2m,direct_radiation,wind_speed_10m,precipitation",
                  "timezone": "Europe/Budapest",
                  "start_date": (ma + timedelta(days=1)).strftime("%Y-%m-%d"),
                  "end_date": (ma + timedelta(days=7)).strftime("%Y-%m-%d")}
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        return pd.DataFrame({
            "Datum": pd.to_datetime(data["hourly"]["time"]),
            "Homerseklet_C": data["hourly"]["temperature_2m"],
            "Paratartalom_szazalek": data["hourly"]["relative_humidity_2m"],
            "Napsugarzas_W_m2": data["hourly"]["direct_radiation"],
            "Szelsebesseg_kmh": data["hourly"]["wind_speed_10m"],
            "Csapadek_mm": data["hourly"]["precipitation"]
        })
    except:
        ma = magyar_ma()
        hours = [ma + timedelta(days=1, hours=h) for d in range(7) for h in range(24)]
        return pd.DataFrame({"Datum": hours[:168], "Homerseklet_C": [20.0]*168,
                            "Paratartalom_szazalek": [60.0]*168, "Napsugarzas_W_m2": [100.0]*168,
                            "Szelsebesseg_kmh": [10.0]*168, "Csapadek_mm": [0.0]*168})

@st.cache_data(ttl=600)
def get_oras_load_history(_datum_kulcs=None):
    if not ENTSOE_API_KEY:
        return None
    try:
        client = EntsoePandasClient(api_key=ENTSOE_API_KEY)
        ma = magyar_ma()
        start = pd.Timestamp((ma - timedelta(days=8)).strftime("%Y-%m-%d"), tz="Europe/Budapest")
        end = pd.Timestamp(ma.strftime("%Y-%m-%d"), tz="Europe/Budapest")
        load = client.query_load("HU", start=start, end=end)
        if isinstance(load, pd.DataFrame):
            load = load.iloc[:, 0]
        return load
    except:
        return None

@st.cache_data(ttl=300)
def get_dam_ar_oras(_datum_kulcs=None):
    if not ENTSOE_API_KEY:
        return 100.0, 100.0, False, None
    try:
        client = EntsoePandasClient(api_key=ENTSOE_API_KEY)
        ma = magyar_ma()
        holnap_start = pd.Timestamp((ma + timedelta(days=1)).strftime("%Y-%m-%d"), tz="Europe/Budapest")
        holnap_end = holnap_start + pd.Timedelta(days=1)
        try:
            dam_holnap = client.query_day_ahead_prices("HU", start=holnap_start, end=holnap_end)
            dam_1nap = float(dam_holnap.mean())
            holnap_oras = dam_holnap
        except:
            dam_1nap = 100.0
            holnap_oras = None
        start_30 = pd.Timestamp((ma - timedelta(days=30)).strftime("%Y-%m-%d"), tz="Europe/Budapest")
        end_30 = pd.Timestamp(ma.strftime("%Y-%m-%d"), tz="Europe/Budapest")
        dam_30 = client.query_day_ahead_prices("HU", start=start_30, end=end_30)
        atlag_30 = float(dam_30.mean()) if dam_30 is not None and len(dam_30) > 0 else 100.0
        return dam_1nap, atlag_30, True, holnap_oras
    except:
        return 100.0, 100.0, False, None

@st.cache_data(ttl=600)
def get_megujulo_utolso_het(_datum_kulcs=None):
    if not ENTSOE_API_KEY:
        return 500.0, 200.0
    try:
        client = EntsoePandasClient(api_key=ENTSOE_API_KEY)
        ma = magyar_ma()
        start = pd.Timestamp((ma - timedelta(days=7)).strftime("%Y-%m-%d"), tz="Europe/Budapest")
        end = pd.Timestamp(ma.strftime("%Y-%m-%d"), tz="Europe/Budapest")
        gen = client.query_generation("HU", start=start, end=end)
        solar_cols = [c for c in gen.columns if 'Solar' in str(c)]
        wind_cols = [c for c in gen.columns if 'Wind' in str(c)]
        nap_atlag = float(gen[solar_cols].sum(axis=1).mean()) if solar_cols else 500.0
        szel_atlag = float(gen[wind_cols].sum(axis=1).mean()) if wind_cols else 200.0
        return nap_atlag, szel_atlag
    except:
        return 500.0, 200.0

# === ELŐREJELZÉS ===
def oras_elorejelzes(idojaras_df, dam_1nap, dam_atlag_30, eur_huf, load_history, nap_atlag, szel_atlag, holnap_oras):
    eredmenyek = []
    if load_history is not None and len(load_history) >= 168:
        lag_buffer = list(load_history.values[-168:])
    else:
        lag_buffer = [4800.0] * 168

    stl_std = stl_params.get("reziduum_std", 233.87)
    stl_mean = stl_params.get("reziduum_atlag", -28.89)
    last_seasonal = stl_seasonal.iloc[-168:].values.flatten() if len(stl_seasonal) >= 168 else [0]*168

    for i, row in idojaras_df.iterrows():
        datum = row["Datum"]
        ora = datum.hour
        if i < 24 and holnap_oras is not None and len(holnap_oras) > ora:
            dam_ar = float(holnap_oras.iloc[ora]) if ora < len(holnap_oras) else dam_atlag_30
        elif i < 24:
            dam_ar = dam_1nap
        else:
            dam_ar = dam_atlag_30

        lag1h = lag_buffer[-1]
        lag24h = lag_buffer[-24] if len(lag_buffer) >= 24 else lag1h
        lag168h = lag_buffer[-168] if len(lag_buffer) >= 168 else lag1h

        X = np.array([[dam_ar, row["Homerseklet_C"], row["Paratartalom_szazalek"],
                       row["Napsugarzas_W_m2"], row["Szelsebesseg_kmh"], row["Csapadek_mm"],
                       eur_huf, ora, datum.weekday() + 1, datum.month,
                       1 if datum.date() in hu_holidays else 0,
                       1 if datum.weekday() >= 5 else 0,
                       1 if row["Homerseklet_C"] < -5 else 0,
                       1 if row["Homerseklet_C"] > 30 else 0,
                       lag1h, lag24h, lag168h,
                       nap_atlag if row["Napsugarzas_W_m2"] > 50 else 0,
                       szel_atlag]])

        josolt = float(model.predict(X)[0])
        josolt = max(josolt, 2000)

        seasonal_idx = i % len(last_seasonal)
        reziduum = josolt - last_seasonal[seasonal_idx]
        riado = abs(reziduum - stl_mean) > 2 * stl_std

        lag_buffer.append(josolt)

        eredmenyek.append({
            "datum": datum, "ora": ora, "homerseklet": row["Homerseklet_C"],
            "fogyasztas": josolt, "dam_ar": dam_ar,
            "koltseg_mft": josolt * dam_ar * eur_huf / 1_000_000,
            "riado": riado, "reziduum": reziduum,
            "napsugarzas": row["Napsugarzas_W_m2"],
            "szelsebesseg": row["Szelsebesseg_kmh"]
        })
    return eredmenyek

# === PLOTLY CHART ALAP ===
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
            if (t.text) {{ t.text = t.text.map(function() {{ return ''; }}); }}
            return t;
        }});
    }}
    function easeOutCubic(t) {{ return 1 - Math.pow(1 - t, 3); }}
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
                        if (copy.text && t < 0.9) {{ copy.text = copy.text.map(function() {{ return ''; }}); }}
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
            if (vis && !lastVisible) {{ setTimeout(runAnimation, START_DELAY); }}
            if (!vis && animDone) {{ animDone = false; }}
            lastVisible = vis;
        }}, 300);
    }});
}})();
</script>
</body>
</html>"""

# === CHART FÜGGVÉNYEK ===
def napi_fogyasztas_chart(datumok, fogyasztasok_gwh, riadok, height=420):
    colors = ["#FF6600" if r else "#0066CC" for r in riadok]
    feliratok = [f"{v:.1f}" for v in fogyasztasok_gwh]
    x_min = (pd.to_datetime(datumok[0]) - pd.Timedelta(hours=18)).isoformat()
    x_max = (pd.to_datetime(datumok[-1]) + pd.Timedelta(hours=18)).isoformat()
    adatok = json.dumps([{
        "type": "bar", "x": datumok, "y": fogyasztasok_gwh,
        "marker": {"color": colors, "opacity": 0.9},
        "text": feliratok, "textposition": "outside",
        "textfont": {"color": "#ffffff", "size": 11},
        "hovertemplate": "%{x}<br>%{y:.1f} GWh<extra></extra>",
        "name": "Fogyasztás", "cliponaxis": False
    }])
    layout = json.dumps({
        "paper_bgcolor": "#0a1628", "plot_bgcolor": "#0f2040",
        "font": {"color": "#cbd5e1", "family": "Inter"},
        "title": {"text": "7 napos napi fogyasztás előrejelzés (GWh)",
                  "font": {"size": 14, "color": "#f1f5f9"}},
        "margin": {"l": 100, "r": 80, "t": 70, "b": 50},
        "xaxis": {"gridcolor": "#1e3a5f", "tickformat": "%m.%d", "color": "#cbd5e1",
                  "range": [x_min, x_max], "type": "date"},
        "yaxis": {"gridcolor": "#1e3a5f", "title": "GWh", "color": "#cbd5e1",
                  "range": [0, max(fogyasztasok_gwh) * 1.25]},
        "bargap": 0.55, "showlegend": False
    })
    return plotly_chart(adatok, layout, "napi_fogyasztas", height)

def napi_koltseg_chart(datumok, koltsegek, eur_huf, height=420):
    x_min = (pd.to_datetime(datumok[0]) - pd.Timedelta(hours=18)).isoformat()
    x_max = (pd.to_datetime(datumok[-1]) + pd.Timedelta(hours=18)).isoformat()
    adatok = json.dumps([{
        "type": "scatter", "x": datumok, "y": koltsegek, "mode": "lines+markers+text",
        "line": {"color": "#FF6600", "width": 3},
        "marker": {"size": 10, "color": "#FF6600", "line": {"color": "#ffffff", "width": 2}},
        "fill": "tozeroy", "fillcolor": "rgba(255,102,0,0.15)",
        "text": [f"{v:.0f}" for v in koltsegek], "textposition": "top center",
        "textfont": {"color": "#ffffff", "size": 10},
        "hovertemplate": "%{x}<br>%{y:.0f} M Ft<extra></extra>",
        "name": "Költség", "cliponaxis": False
    }])
    layout = json.dumps({
        "paper_bgcolor": "#0a1628", "plot_bgcolor": "#0f2040",
        "font": {"color": "#cbd5e1", "family": "Inter"},
        "title": {"text": f"Becsült napi energiaköltség (M Ft) | EUR/HUF: {eur_huf:.1f}",
                  "font": {"size": 14, "color": "#f1f5f9"}},
        "margin": {"l": 100, "r": 80, "t": 70, "b": 50},
        "xaxis": {"gridcolor": "#1e3a5f", "tickformat": "%m.%d", "color": "#cbd5e1",
                  "range": [x_min, x_max], "type": "date"},
        "yaxis": {"gridcolor": "#1e3a5f", "title": "Millió Ft", "color": "#cbd5e1",
                  "range": [min(koltsegek)*0.8, max(koltsegek)*1.2]},
        "showlegend": False
    })
    return plotly_chart(adatok, layout, "napi_koltseg", height)

def oras_fogyasztas_chart(datumok, fogyasztasok, riadok, height=420):
    colors = ["#FF6600" if r else "#0066CC" for r in riadok]
    adatok = json.dumps([{
        "type": "scatter", "x": datumok, "y": fogyasztasok, "mode": "lines",
        "line": {"color": "#0066CC", "width": 2},
        "fill": "tozeroy", "fillcolor": "rgba(0,102,204,0.15)",
        "hovertemplate": "%{x}<br>%{y:.0f} MWh<extra></extra>",
        "name": "Fogyasztás"
    }, {
        "type": "scatter", "x": [d for d, r in zip(datumok, riadok) if r],
        "y": [f for f, r in zip(fogyasztasok, riadok) if r],
        "mode": "markers", "marker": {"color": "#FF6600", "size": 8, "symbol": "diamond"},
        "hovertemplate": "%{x}<br>%{y:.0f} MWh (STL riasztás)<extra></extra>",
        "name": "STL riasztás"
    }])
    layout = json.dumps({
        "paper_bgcolor": "#0a1628", "plot_bgcolor": "#0f2040",
        "font": {"color": "#cbd5e1", "family": "Inter"},
        "title": {"text": "168 órás fogyasztás előrejelzés (MWh) — STL riasztással",
                  "font": {"size": 14, "color": "#f1f5f9"}},
        "margin": {"l": 100, "r": 80, "t": 70, "b": 50},
        "xaxis": {"gridcolor": "#1e3a5f", "tickformat": "%m.%d %H:%M", "color": "#cbd5e1"},
        "yaxis": {"gridcolor": "#1e3a5f", "title": "MWh", "color": "#cbd5e1"},
        "showlegend": True, "legend": {"font": {"color": "#cbd5e1"}}
    })
    return plotly_chart(adatok, layout, "oras_fogyasztas", height)

def oras_homerseklet_chart(datumok, homersekletek, height=420):
    adatok = json.dumps([{
        "type": "scatter", "x": datumok, "y": homersekletek, "mode": "lines",
        "line": {"color": "#10b981", "width": 2},
        "fill": "tozeroy", "fillcolor": "rgba(16,185,129,0.15)",
        "hovertemplate": "%{x}<br>%{y:.1f}°C<extra></extra>",
        "name": "Hőmérséklet"
    }])
    layout = json.dumps({
        "paper_bgcolor": "#0a1628", "plot_bgcolor": "#0f2040",
        "font": {"color": "#cbd5e1", "family": "Inter"},
        "title": {"text": "168 órás hőmérséklet előrejelzés (°C)",
                  "font": {"size": 14, "color": "#f1f5f9"}},
        "margin": {"l": 100, "r": 80, "t": 70, "b": 50},
        "xaxis": {"gridcolor": "#1e3a5f", "tickformat": "%m.%d %H:%M", "color": "#cbd5e1"},
        "yaxis": {"gridcolor": "#1e3a5f", "title": "°C", "color": "#cbd5e1"},
        "showlegend": False
    })
    return plotly_chart(adatok, layout, "oras_homerseklet", height)

def negativ_arak_chart(neg_df, height=420):
    adatok = json.dumps([{
        "type": "bar", "x": neg_df["Ev_Honap"].tolist(), "y": neg_df["negativ_ora"].tolist(),
        "marker": {"color": ["#FF6600" if v > 10 else "#0066CC" for v in neg_df["negativ_ora"]], "opacity": 0.9},
        "hovertemplate": "%{x}<br>%{y} negatív óra<extra></extra>",
        "name": "Negatív órák"
    }])
    layout = json.dumps({
        "paper_bgcolor": "#0a1628", "plot_bgcolor": "#0f2040",
        "font": {"color": "#cbd5e1", "family": "Inter"},
        "title": {"text": "Negatív DAM árak havi trendje (2015–2026) — napelemes túltermelés hatása",
                  "font": {"size": 14, "color": "#f1f5f9"}},
        "margin": {"l": 80, "r": 40, "t": 70, "b": 80},
        "xaxis": {"gridcolor": "#1e3a5f", "color": "#cbd5e1", "tickangle": -45,
                  "dtick": 6},
        "yaxis": {"gridcolor": "#1e3a5f", "title": "Negatív órák száma / hónap", "color": "#cbd5e1"},
        "bargap": 0.3, "showlegend": False
    })
    return plotly_chart(adatok, layout, "negativ_arak", height)

# === FEJLÉC ===
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
                MVM Energiafogyasztás Előrejelző V3
            </div>
            <div style="font-family:Inter,sans-serif; font-size:13px;
                        color:#93c5fd; margin-top:4px; letter-spacing:0.5px;">
                XGBoost (FLAML AutoML) &nbsp;|&nbsp; 168 órás előrejelzés &nbsp;|&nbsp; STL anomáliadetekció
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# === FŐTABOK ===
tab1, tab2, tab3 = st.tabs(["📊 Napi Összesítés", "⏱️ Órás Bontás", "⚡ Negatív Árak"])

# === ADATLEKÉRÉS ===
col_gomb, col_allapot = st.columns([1, 3])
with col_gomb:
    frissit = st.button("🔄 Előrejelzés frissítése", type="primary")
with col_allapot:
    allapot_ph = st.empty()

if frissit:
    st.session_state.frissites_ideje = pd.Timestamp.now(tz="Europe/Budapest").strftime("%Y-%m-%d %H:%M:%S")
    get_eur_huf.clear()
    get_oras_idojaras.clear()
    get_dam_ar_oras.clear()
    get_oras_load_history.clear()
    get_megujulo_utolso_het.clear()
    st.session_state.pop("eredmenyek", None)
    st.rerun()

if "eredmenyek" not in st.session_state:
    with st.spinner("Adatok lekérése (ENTSO-E, Open-Meteo, ECB)..."):
        ora_kulcs = magyar_ma().strftime("%Y-%m-%d-%H")
        eur_huf = get_eur_huf(_datum_kulcs=ora_kulcs)
        idojaras_df = get_oras_idojaras(_datum_kulcs=ora_kulcs)
        dam_1nap, dam_atlag_30, dam_valodi, holnap_oras = get_dam_ar_oras(_datum_kulcs=ora_kulcs)
        load_history = get_oras_load_history(_datum_kulcs=ora_kulcs)
        nap_atlag, szel_atlag = get_megujulo_utolso_het(_datum_kulcs=ora_kulcs)
        eredmenyek = oras_elorejelzes(idojaras_df, dam_1nap, dam_atlag_30, eur_huf,
                                       load_history, nap_atlag, szel_atlag, holnap_oras)
        st.session_state.eredmenyek = eredmenyek
        st.session_state.eur_huf = eur_huf
        st.session_state.dam_1nap = dam_1nap
        st.session_state.dam_atlag_30 = dam_atlag_30
        st.session_state.frissites_ideje = pd.Timestamp.now(tz="Europe/Budapest").strftime("%Y-%m-%d %H:%M:%S")

if "frissites_ideje" in st.session_state:
    allapot_ph.success(f"✅ Frissítve: {st.session_state.frissites_ideje}")

# === ADATOK FELDOLGOZÁSA ===
if "eredmenyek" in st.session_state:
    eredmenyek = st.session_state.eredmenyek
    eur_huf = st.session_state.eur_huf
    dam_1nap = st.session_state.dam_1nap
    dam_atlag_30 = st.session_state.dam_atlag_30

    edf = pd.DataFrame(eredmenyek)
    edf["datum"] = pd.to_datetime(edf["datum"])
    edf["nap"] = edf["datum"].dt.date

    # Napi aggregálás
    napi = edf.groupby("nap").agg(
        fogyasztas_gwh=("fogyasztas", lambda x: x.sum() / 1000),
        koltseg_mft=("koltseg_mft", "sum"),
        homerseklet_atlag=("homerseklet", "mean"),
        riado_orak=("riado", "sum"),
        max_ora_fogyasztas=("fogyasztas", "max"),
        min_ora_fogyasztas=("fogyasztas", "min")
    ).reset_index()
    napi["riado"] = napi["riado_orak"] > 0

    # Modell infó
    st.markdown(f"""
    <div style="background:#0a1628; border:1px solid #1e3a5f; border-radius:8px;
                padding:8px 16px; margin-bottom:12px; text-align:center;">
        <span style="color:#10b981; font-weight:600; font-family:Montserrat,sans-serif;">
        Aktív modell: XGBoost (FLAML AutoML) — 168 órás előrejelzés &nbsp;|&nbsp; STL anomáliadetekció</span>
    </div>""", unsafe_allow_html=True)

    # === KÁRTYÁK ===
    heti_fogyasztas_gwh = napi["fogyasztas_gwh"].sum()
    heti_koltseg = napi["koltseg_mft"].sum()
    max_nap = napi.loc[napi["fogyasztas_gwh"].idxmax()]
    min_nap = napi.loc[napi["fogyasztas_gwh"].idxmin()]
    riado_napok = napi[napi["riado"]]
    riado_ertek = ", ".join([str(n) for n in riado_napok["nap"]]) if len(riado_napok) > 0 else "Nincs"
    riado_szin = "#FF6600" if len(riado_napok) > 0 else "#10b981"
    riado_ikon = "🚨" if len(riado_napok) > 0 else "✅"
    stl_riado_orak = int(edf["riado"].sum())

    st.markdown(f"""
    <div class="kartya-sor">
      <div class="kartya">
        <div class="kartya-cim">⚡ Heti fogyasztás</div>
        <div class="kartya-ertek">{heti_fogyasztas_gwh:.1f} GWh</div>
      </div>
      <div class="kartya">
        <div class="kartya-cim">💰 Heti költség</div>
        <div class="kartya-ertek">{heti_koltseg/1000:.1f} Mrd Ft</div>
      </div>
      <div class="kartya">
        <div class="kartya-cim">📈 Csúcs nap</div>
        <div class="kartya-ertek">{max_nap['fogyasztas_gwh']:.1f} GWh</div>
        <div class="kartya-sub">{max_nap['nap']}</div>
      </div>
      <div class="kartya">
        <div class="kartya-cim">📉 Minimum nap</div>
        <div class="kartya-ertek">{min_nap['fogyasztas_gwh']:.1f} GWh</div>
        <div class="kartya-sub">{min_nap['nap']}</div>
      </div>
      <div class="kartya">
        <div class="kartya-cim">🏦 DAM valódi</div>
        <div class="kartya-ertek">{dam_1nap:.2f} EUR/MWh</div>
      </div>
      <div class="kartya">
        <div class="kartya-cim">🏦 DAM 30 napos</div>
        <div class="kartya-ertek">{dam_atlag_30:.2f} EUR/MWh</div>
        <div class="kartya-sub">30 napos átlag</div>
      </div>
      <div class="kartya">
        <div class="kartya-cim">💱 EUR/HUF</div>
        <div class="kartya-ertek">{eur_huf:.1f} Ft</div>
      </div>
      <div class="kartya" style="border-color:{riado_szin};">
        <div class="kartya-cim">{riado_ikon} STL Riasztás</div>
        <div class="kartya-ertek" style="color:{riado_szin};">{stl_riado_orak} óra</div>
        <div class="kartya-sub">z-score > 2σ</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # === TAB 1: NAPI ÖSSZESÍTÉS ===
    with tab1:
        datumok_napi = [str(n) for n in napi["nap"]]
        gt1, gt2 = st.tabs(["📊 Fogyasztás", "💰 Költség"])
        with gt1:
            components.html(
                napi_fogyasztas_chart(datumok_napi, napi["fogyasztas_gwh"].tolist(),
                                     napi["riado"].tolist()),
                height=440, scrolling=False)
        with gt2:
            components.html(
                napi_koltseg_chart(datumok_napi, napi["koltseg_mft"].tolist(), eur_huf),
                height=440, scrolling=False)

    # === TAB 2: ÓRÁS BONTÁS ===
    with tab2:
        datumok_oras = [d.strftime("%Y-%m-%d %H:%M") for d in edf["datum"]]
        ot1, ot2 = st.tabs(["⚡ Fogyasztás + STL", "🌡️ Hőmérséklet"])
        with ot1:
            components.html(
                oras_fogyasztas_chart(datumok_oras, edf["fogyasztas"].tolist(),
                                     edf["riado"].tolist()),
                height=440, scrolling=False)
            if stl_riado_orak > 0:
                riado_df = edf[edf["riado"]][["datum", "fogyasztas", "reziduum"]].copy()
                riado_df["datum"] = riado_df["datum"].dt.strftime("%m.%d %H:%M")
                riado_df["reziduum"] = riado_df["reziduum"].round(1)
                riado_df.columns = ["Időpont", "Fogyasztás (MWh)", "Reziduum (MWh)"]
                st.markdown(f"""
                <div style="background:#0a1628; border:1px solid #FF6600; border-radius:8px;
                            padding:12px 16px; margin:8px 0;">
                    <span style="color:#FF6600; font-weight:600;">
                    🚨 {stl_riado_orak} órában az STL reziduum meghaladja a 2σ küszöböt
                    (σ = {stl_params.get('reziduum_std', 233.87):.0f} MWh)</span>
                </div>""", unsafe_allow_html=True)
                st.dataframe(riado_df, use_container_width=True, hide_index=True)
        with ot2:
            components.html(
                oras_homerseklet_chart(datumok_oras, edf["homerseklet"].tolist()),
                height=440, scrolling=False)

    # === TAB 3: NEGATÍV ÁRAK ===
    with tab3:
        osszes_neg = int(neg_summary["negativ_ora"].sum())
        utolso_ev = neg_summary[neg_summary["Ev_Honap"] >= "2025-01"]
        utolso_ev_neg = int(utolso_ev["negativ_ora"].sum()) if len(utolso_ev) > 0 else 0
        legrosszabb = neg_summary.loc[neg_summary["negativ_ora"].idxmax()]

        st.markdown(f"""
        <div class="kartya-sor" style="grid-template-columns: repeat(4, 1fr);">
          <div class="kartya">
            <div class="kartya-cim">📊 Összes negatív óra</div>
            <div class="kartya-ertek">{osszes_neg}</div>
            <div class="kartya-sub">2015–2026</div>
          </div>
          <div class="kartya">
            <div class="kartya-cim">📈 2025–2026 negatív óra</div>
            <div class="kartya-ertek">{utolso_ev_neg}</div>
            <div class="kartya-sub">utolsó 18 hónap</div>
          </div>
          <div class="kartya">
            <div class="kartya-cim">📉 Legrosszabb hónap</div>
            <div class="kartya-ertek">{int(legrosszabb['negativ_ora'])} óra</div>
            <div class="kartya-sub">{legrosszabb['Ev_Honap']}</div>
          </div>
          <div class="kartya">
            <div class="kartya-cim">⚠️ Legmélyebb ár</div>
            <div class="kartya-ertek">{neg_summary['min_ar'].min():.0f} EUR/MWh</div>
            <div class="kartya-sub">HUPX padlóár</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        components.html(
            negativ_arak_chart(neg_summary),
            height=480, scrolling=False)

        st.markdown("""
        <div style="background:#0a1628; border:1px solid #1e3a5f; border-radius:8px;
                    padding:12px 16px; margin:8px 0;">
            <span style="color:#93c5fd; font-size:13px; font-family:Inter,sans-serif;">
            💡 <b>A negatív árak oka:</b> hétvégén és ünnepnapokon a napelemes túltermelés olyan szintet ér el,
            hogy az eladó fizet a vevőnek az áram elviteléért. A HUPX tőzsde padlóára -500 EUR/MWh.
            A jelenség 2024-2026-ban drasztikusan növekszik a megújuló kapacitás bővülésével.</span>
        </div>""", unsafe_allow_html=True)
else:
    st.info("Kattints az Előrejelzés frissítése gombra!")
