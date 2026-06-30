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

# === OLDAL BEÁLLÍTÁS ===
st.set_page_config(
    page_title="MVM Energiafogyasztás Előrejelző V3",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# === STÍLUSOK ===
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@700;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
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
footer, #MainMenu, header { display: none !important; }

/* === KÁRTYÁK === */
.kartya-sor {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
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
    display: flex; flex-direction: column; justify-content: center;
    transition: border-color 0.25s ease, box-shadow 0.25s ease, transform 0.2s ease;
}
.kartya:hover {
    border-color: #0066CC;
    box-shadow: 0 4px 24px rgba(0,102,204,0.45);
    transform: translateY(-3px);
}
.kartya-cim {
    color: #64748b; font-size: 8px; font-family: Inter,sans-serif;
    letter-spacing: 0.8px; text-transform: uppercase; margin-bottom: 4px;
}
.kartya-ertek {
    color: #FF6600; font-size: 14px; font-weight: 700;
    font-family: Montserrat,sans-serif;
}
.kartya-sub { color: #94a3b8; font-size: 10px; margin-top: 3px; }
/* Fallback jelzés — szürke háttér + figyelmeztető keret */
.kartya-fallback {
    background: #0d1117 !important;
    border-color: #4a4a2e !important;
}
.kartya-fallback .kartya-ertek { color: #a0a0a0 !important; }

/* === MINI PANELEK === */
.mini-panel {
    background: #0a1628;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 16px;
    min-height: 180px;
    transition: border-color 0.25s ease, box-shadow 0.25s ease;
}
.mini-panel:hover {
    border-color: #0066CC;
    box-shadow: 0 4px 24px rgba(0,102,204,0.35);
}
.mini-panel-cim {
    color: #f1f5f9; font-size: 14px; font-weight: 600;
    font-family: Montserrat,sans-serif; margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

# === KONSTANSOK ===
hu_holidays = holidays.Hungary(years=[2025, 2026, 2027])
ENTSOE_API_KEY = os.environ.get("ENTSOE_API_KEY", "")
BASE = os.path.dirname(os.path.abspath(__file__))

# A FLAML/XGBoost training-ben használt feature sorrend (Colab X.columns alapján)
FEATURE_SORREND = [
    'DAM_EUR_MWh', 'Homerseklet_C', 'Paratartalom_szazalek', 'Napsugarzas_W_m2',
    'Szelsebesseg_kmh', 'Csapadek_mm', 'EUR_HUF', 'Ora', 'Het_napja', 'Honap',
    'Unnepnap', 'Hetvege', 'Extrem_hideg', 'Extrem_meleg',
    'Fogyasztas_lag1h', 'Fogyasztas_lag24h', 'Fogyasztas_lag168h',
    'Nap_termeles_MW', 'Szel_termeles_MW'
]

def magyar_ma():
    return pd.Timestamp.now(tz="Europe/Budapest").normalize().tz_localize(None).to_pydatetime()

# === MODELL BETÖLTÉS ===
@st.cache_resource
def modell_betoltese():
    return joblib.load(f"{BASE}/xgb_model.pkl")

@st.cache_resource
def stl_betoltese():
    with open(f"{BASE}/stl_params.json", "r") as f:
        params = json.load(f)
    seasonal = pd.read_csv(f"{BASE}/stl_seasonal.csv", index_col=0, parse_dates=True)
    return params, seasonal

model = modell_betoltese()
stl_params, stl_seasonal = stl_betoltese()

# === ADATLEKÉRŐ FÜGGVÉNYEK ===
# Minden függvény visszaadja az adatot + egy bool-t: True = valós, False = fallback

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
        ertek = float(df["OBS_VALUE"].dropna().iloc[-1])
        return ertek, True  # valós adat
    except:
        return 395.0, False  # fallback

@st.cache_data(ttl=600)
def get_oras_idojaras(_datum_kulcs=None):
    try:
        ma = magyar_ma()
        url = "https://api.open-meteo.com/v1/forecast"
        params = {"latitude": 47.5, "longitude": 19.0,
                  "hourly": "temperature_2m,relative_humidity_2m,direct_radiation,wind_speed_10m,precipitation",
                  "timezone": "Europe/Budapest",
                  "start_date": (ma + timedelta(days=1)).strftime("%Y-%m-%d"),
                  "end_date": (ma + timedelta(days=2)).strftime("%Y-%m-%d")}  # 48 óra
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        df = pd.DataFrame({
            "Datum": pd.to_datetime(data["hourly"]["time"]),
            "Homerseklet_C": data["hourly"]["temperature_2m"],
            "Paratartalom_szazalek": data["hourly"]["relative_humidity_2m"],
            "Napsugarzas_W_m2": data["hourly"]["direct_radiation"],
            "Szelsebesseg_kmh": data["hourly"]["wind_speed_10m"],
            "Csapadek_mm": data["hourly"]["precipitation"]
        })
        return df, True
    except:
        ma = magyar_ma()
        hours = [ma + timedelta(days=1, hours=h) for h in range(48)]
        return pd.DataFrame({"Datum": hours, "Homerseklet_C": [20.0]*48,
                            "Paratartalom_szazalek": [60.0]*48, "Napsugarzas_W_m2": [100.0]*48,
                            "Szelsebesseg_kmh": [10.0]*48, "Csapadek_mm": [0.0]*48}), False

@st.cache_data(ttl=600)
def get_oras_load_history(_datum_kulcs=None):
    if not ENTSOE_API_KEY:
        return None, False
    try:
        client = EntsoePandasClient(api_key=ENTSOE_API_KEY)
        ma = magyar_ma()
        start = pd.Timestamp((ma - timedelta(days=8)).strftime("%Y-%m-%d"), tz="Europe/Budapest")
        end = pd.Timestamp(ma.strftime("%Y-%m-%d"), tz="Europe/Budapest")
        load = client.query_load("HU", start=start, end=end)
        if isinstance(load, pd.DataFrame):
            load = load.iloc[:, 0]
        return load, True
    except:
        return None, False

@st.cache_data(ttl=300)
def get_dam_ar_oras(_datum_kulcs=None):
    if not ENTSOE_API_KEY:
        return 100.0, 100.0, None, False
    try:
        client = EntsoePandasClient(api_key=ENTSOE_API_KEY)
        ma = magyar_ma()
        # Holnapi árak
        holnap_start = pd.Timestamp((ma + timedelta(days=1)).strftime("%Y-%m-%d"), tz="Europe/Budapest")
        holnap_end = holnap_start + pd.Timedelta(days=1)
        try:
            holnap_oras = client.query_day_ahead_prices("HU", start=holnap_start, end=holnap_end)
            dam_1nap = float(holnap_oras.mean())
        except:
            holnap_oras = None
            dam_1nap = 100.0
        # 30 napos átlag
        start_30 = pd.Timestamp((ma - timedelta(days=30)).strftime("%Y-%m-%d"), tz="Europe/Budapest")
        end_30 = pd.Timestamp(ma.strftime("%Y-%m-%d"), tz="Europe/Budapest")
        dam_30 = client.query_day_ahead_prices("HU", start=start_30, end=end_30)
        atlag_30 = float(dam_30.mean()) if dam_30 is not None and len(dam_30) > 0 else 100.0
        return dam_1nap, atlag_30, holnap_oras, True
    except:
        return 100.0, 100.0, None, False

@st.cache_data(ttl=600)
def get_megujulo_utolso_het(_datum_kulcs=None):
    if not ENTSOE_API_KEY:
        return 500.0, 200.0, False
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
        return nap_atlag, szel_atlag, True
    except:
        return 500.0, 200.0, False

# === ELŐREJELZÉS — DataFrame-mel, nem numpy-jal ===
def oras_elorejelzes(idojaras_df, dam_1nap, dam_atlag_30, eur_huf, load_history,
                     nap_atlag, szel_atlag, holnap_oras):
    eredmenyek = []

    # Lag buffer: valós ENTSO-E adatból, ha van
    if load_history is not None and len(load_history) >= 168:
        lag_buffer = list(load_history.values[-168:])
    else:
        lag_buffer = [4800.0] * 168

    # STL szezonális átlag óránként
    seasonal_by_hour = stl_seasonal.groupby(stl_seasonal.index.hour).mean().iloc[:, 0]
    stl_std = stl_params.get("reziduum_std", 233.87)
    stl_mean = stl_params.get("reziduum_atlag", -28.89)

    for i, row in idojaras_df.iterrows():
        datum = row["Datum"]
        ora = datum.hour

        # DAM ár: holnapi valós ha van, különben 30 napos átlag
        if i < 24 and holnap_oras is not None and ora < len(holnap_oras):
            dam_ar = float(holnap_oras.iloc[ora])
        elif i < 24:
            dam_ar = dam_1nap
        else:
            dam_ar = dam_atlag_30

        # Lag-ek: mindig a valós ENTSO-E buffer-ből (lag24h és lag168h soha nem jósolt)
        lag1h = lag_buffer[-1]
        lag24h = lag_buffer[-24] if len(lag_buffer) >= 24 else lag1h
        lag168h = lag_buffer[-168] if len(lag_buffer) >= 168 else lag1h

        # Megújuló: napsugárzás alapján (éjszaka 0)
        nap_mw = nap_atlag if row["Napsugarzas_W_m2"] > 50 else 0

        # DataFrame-mel hívjuk a modellt — a feature sorrend GARANTÁLT
        X_sor = pd.DataFrame([{
            'DAM_EUR_MWh': dam_ar,
            'Homerseklet_C': row["Homerseklet_C"],
            'Paratartalom_szazalek': row["Paratartalom_szazalek"],
            'Napsugarzas_W_m2': row["Napsugarzas_W_m2"],
            'Szelsebesseg_kmh': row["Szelsebesseg_kmh"],
            'Csapadek_mm': row["Csapadek_mm"],
            'EUR_HUF': eur_huf,
            'Ora': ora,
            'Het_napja': datum.weekday() + 1,
            'Honap': datum.month,
            'Unnepnap': 1 if datum.date() in hu_holidays else 0,
            'Hetvege': 1 if datum.weekday() >= 5 else 0,
            'Extrem_hideg': 1 if row["Homerseklet_C"] < -5 else 0,
            'Extrem_meleg': 1 if row["Homerseklet_C"] > 30 else 0,
            'Fogyasztas_lag1h': lag1h,
            'Fogyasztas_lag24h': lag24h,
            'Fogyasztas_lag168h': lag168h,
            'Nap_termeles_MW': nap_mw,
            'Szel_termeles_MW': szel_atlag
        }], columns=FEATURE_SORREND)  # <-- sorrend biztosítva!

        josolt = float(model.predict(X_sor)[0])
        josolt = max(josolt, 2000)

        # STL riasztás: jósolt - trend_szint - seasonal = reziduum
        seasonal_val = seasonal_by_hour.get(ora, 0)
        trend_szint = stl_params.get("trend_utolso", josolt)  # ha nincs, nem riaszt feleslegesen
        reziduum = josolt - trend_szint - seasonal_val
        riado = abs(reziduum - stl_mean) > 2.5 * stl_std

        # Lag buffer frissítés: a lag1h göngyölít, de lag24h és lag168h valós marad
        lag_buffer.append(josolt)

        eredmenyek.append({
            "datum": datum, "ora": ora,
            "homerseklet": row["Homerseklet_C"],
            "napsugarzas": row["Napsugarzas_W_m2"],
            "szelsebesseg": row["Szelsebesseg_kmh"],
            "fogyasztas": josolt,
            "dam_ar": dam_ar,
            "koltseg_mft": josolt * dam_ar * eur_huf / 1_000_000,
            "riado": riado,
            "reziduum": reziduum
        })

    return eredmenyek

# === PLOTLY MINI CHART (egyszerűsített, gyors) ===
def mini_bar_chart(values, labels, colors, chart_id, height=150):
    adatok = json.dumps([{
        "type": "bar", "x": labels, "y": values,
        "marker": {"color": colors, "opacity": 0.9},
        "text": [f"{v:.0f}" for v in values], "textposition": "outside",
        "textfont": {"color": "#ffffff", "size": 9},
        "hovertemplate": "%{x}<br>%{y:.0f}<extra></extra>"
    }])
    layout = json.dumps({
        "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": "#cbd5e1", "family": "Inter", "size": 10},
        "margin": {"l": 30, "r": 10, "t": 10, "b": 30},
        "xaxis": {"gridcolor": "#1e3a5f", "tickformat": "%m.%d", "color": "#94a3b8"},
        "yaxis": {"gridcolor": "#1e3a5f", "color": "#94a3b8", "showticklabels": False},
        "bargap": 0.4, "showlegend": False
    })
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
<style>*{{margin:0;padding:0}}body{{background:transparent}}#{chart_id}{{width:100%;height:{height}px}}</style>
</head><body><div id="{chart_id}"></div>
<script>Plotly.newPlot('{chart_id}',{adatok},{layout},{{responsive:true,displayModeBar:false}});</script>
</body></html>"""

def mini_line_chart(values, labels, color, chart_id, height=150, suffix=""):
    adatok = json.dumps([{
        "type": "scatter", "x": labels, "y": values, "mode": "lines",
        "line": {"color": color, "width": 2},
        "fill": "tozeroy", "fillcolor": color.replace(")", ",0.15)").replace("rgb", "rgba"),
        "hovertemplate": f"%{{x}}<br>%{{y:.1f}}{suffix}<extra></extra>"
    }])
    layout = json.dumps({
        "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": "#cbd5e1", "family": "Inter", "size": 10},
        "margin": {"l": 30, "r": 10, "t": 10, "b": 30},
        "xaxis": {"gridcolor": "#1e3a5f", "color": "#94a3b8"},
        "yaxis": {"gridcolor": "#1e3a5f", "color": "#94a3b8", "showticklabels": False},
        "showlegend": False
    })
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
<style>*{{margin:0;padding:0}}body{{background:transparent}}#{chart_id}{{width:100%;height:{height}px}}</style>
</head><body><div id="{chart_id}"></div>
<script>Plotly.newPlot('{chart_id}',{adatok},{layout},{{responsive:true,displayModeBar:false}});</script>
</body></html>"""

# === DAM ÁR ZÓNA SZÍNEK ===
def dam_zona_szin(ar, atlag):
    if ar < atlag * 0.7:
        return "#10b981"  # zöld — olcsó
    elif ar < atlag * 1.3:
        return "#f59e0b"  # sárga — átlagos
    else:
        return "#ef4444"  # piros — drága

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
                XGBoost (FLAML AutoML) &nbsp;|&nbsp; 48 órás előrejelzés &nbsp;|&nbsp; Okos Mérő Asszisztens
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# === FRISSÍTÉS GOMB ===
col_gomb, col_allapot = st.columns([1, 3])
with col_gomb:
    frissit = st.button("🔄 Előrejelzés frissítése", type="primary")
with col_allapot:
    allapot_ph = st.empty()

if frissit:
    get_eur_huf.clear()
    get_oras_idojaras.clear()
    get_dam_ar_oras.clear()
    get_oras_load_history.clear()
    get_megujulo_utolso_het.clear()
    st.session_state.pop("eredmenyek", None)
    st.rerun()

# === ADATLEKÉRÉS ===
if "eredmenyek" not in st.session_state:
    with st.spinner("Adatok lekérése (ENTSO-E, Open-Meteo, ECB)..."):
        ora_kulcs = magyar_ma().strftime("%Y-%m-%d-%H")

        eur_huf, eur_valodi = get_eur_huf(_datum_kulcs=ora_kulcs)
        idojaras_df, ido_valodi = get_oras_idojaras(_datum_kulcs=ora_kulcs)
        dam_1nap, dam_atlag_30, holnap_oras, dam_valodi = get_dam_ar_oras(_datum_kulcs=ora_kulcs)
        load_history, load_valodi = get_oras_load_history(_datum_kulcs=ora_kulcs)
        nap_atlag, szel_atlag, megujulo_valodi = get_megujulo_utolso_het(_datum_kulcs=ora_kulcs)

        eredmenyek = oras_elorejelzes(idojaras_df, dam_1nap, dam_atlag_30, eur_huf,
                                       load_history, nap_atlag, szel_atlag, holnap_oras)

        st.session_state.eredmenyek = eredmenyek
        st.session_state.eur_huf = eur_huf
        st.session_state.dam_1nap = dam_1nap
        st.session_state.dam_atlag_30 = dam_atlag_30
        st.session_state.holnap_oras = holnap_oras
        st.session_state.frissites_ideje = pd.Timestamp.now(tz="Europe/Budapest").strftime("%Y-%m-%d %H:%M")
        # Fallback státuszok mentése
        st.session_state.fallback = {
            "eur": not eur_valodi,
            "idojaras": not ido_valodi,
            "dam": not dam_valodi,
            "load": not load_valodi,
            "megujulo": not megujulo_valodi
        }

if "frissites_ideje" in st.session_state:
    fb = st.session_state.get("fallback", {})
    fb_lista = [k for k, v in fb.items() if v]
    if fb_lista:
        allapot_ph.warning(f"⚠️ Frissítve: {st.session_state.frissites_ideje} | Fallback: {', '.join(fb_lista)}")
    else:
        allapot_ph.success(f"✅ Frissítve: {st.session_state.frissites_ideje} | Minden adat valós")

# === FŐ TARTALOM ===
if "eredmenyek" in st.session_state:
    eredmenyek = st.session_state.eredmenyek
    eur_huf = st.session_state.eur_huf
    dam_1nap = st.session_state.dam_1nap
    dam_atlag_30 = st.session_state.dam_atlag_30
    fb = st.session_state.get("fallback", {})

    edf = pd.DataFrame(eredmenyek)
    edf["datum"] = pd.to_datetime(edf["datum"])
    edf["nap"] = edf["datum"].dt.date

    # Napi aggregálás
    napi = edf.groupby("nap").agg(
        fogyasztas_gwh=("fogyasztas", lambda x: x.sum() / 1000),
        koltseg_mft=("koltseg_mft", "sum"),
        homerseklet_atlag=("homerseklet", "mean"),
        homerseklet_max=("homerseklet", "max"),
        homerseklet_min=("homerseklet", "min"),
        riado_orak=("riado", "sum"),
        max_fogyasztas=("fogyasztas", "max"),
        min_fogyasztas=("fogyasztas", "min")
    ).reset_index()

    osszfogyasztas = napi["fogyasztas_gwh"].sum()
    osszkoltseg = napi["koltseg_mft"].sum()
    stl_riado_orak = int(edf["riado"].sum())
    riado_szin = "#FF6600" if stl_riado_orak > 0 else "#10b981"

    # === KÁRTYÁK ===
    def fb_class(kulcs):
        return "kartya-fallback" if fb.get(kulcs, False) else ""

    def fb_ikon(kulcs):
        return " ⚠️" if fb.get(kulcs, False) else ""

    st.markdown(f"""
    <div class="kartya-sor">
      <div class="kartya">
        <div class="kartya-cim">⚡ 48h fogyasztás</div>
        <div class="kartya-ertek">{osszfogyasztas:.1f} GWh</div>
      </div>
      <div class="kartya">
        <div class="kartya-cim">💰 48h költség</div>
        <div class="kartya-ertek">{osszkoltseg:.0f} M Ft</div>
      </div>
      <div class="kartya {fb_class('dam')}">
        <div class="kartya-cim">🏦 DAM holnap{fb_ikon('dam')}</div>
        <div class="kartya-ertek">{dam_1nap:.1f} €/MWh</div>
      </div>
      <div class="kartya {fb_class('dam')}">
        <div class="kartya-cim">🏦 DAM 30 napos{fb_ikon('dam')}</div>
        <div class="kartya-ertek">{dam_atlag_30:.1f} €/MWh</div>
        <div class="kartya-sub">átlag</div>
      </div>
      <div class="kartya {fb_class('eur')}">
        <div class="kartya-cim">💱 EUR/HUF{fb_ikon('eur')}</div>
        <div class="kartya-ertek">{eur_huf:.1f} Ft</div>
      </div>
      <div class="kartya">
        <div class="kartya-cim">🌡️ Hőm. tartomány</div>
        <div class="kartya-ertek">{edf['homerseklet'].min():.0f}–{edf['homerseklet'].max():.0f} °C</div>
      </div>
      <div class="kartya" style="border-color:{riado_szin};">
        <div class="kartya-cim">{'🚨' if stl_riado_orak > 0 else '✅'} STL riasztás</div>
        <div class="kartya-ertek" style="color:{riado_szin};">{stl_riado_orak} óra</div>
        <div class="kartya-sub">/ {len(edf)} órából</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # === TABOK ===
    tab1, tab2, tab3 = st.tabs(["📊 Áttekintés", "⏱️ Előrejelzés", "🔌 Okos Mérő"])

    # === TAB 1: ÁTTEKINTÉS — mini panelek ===
    with tab1:
        col1, col2, col3 = st.columns(3)

        # Mini panel 1: Holnapi fogyasztás
        with col1:
            st.markdown('<div class="mini-panel"><div class="mini-panel-cim">⚡ Holnapi fogyasztás (MWh)</div>', unsafe_allow_html=True)
            holnap_df = edf[edf["datum"].dt.date == edf["datum"].dt.date.iloc[0]]
            if len(holnap_df) > 0:
                labels_h = [d.strftime("%H:%M") for d in holnap_df["datum"]]
                values_h = holnap_df["fogyasztas"].tolist()
                components.html(mini_line_chart(values_h, labels_h, "rgb(0,102,204)", "mini_fogy", 140, " MWh"),
                               height=155, scrolling=False)
                cscs = holnap_df.loc[holnap_df["fogyasztas"].idxmax()]
                st.markdown(f'<div class="kartya-sub">Csúcs: {cscs["fogyasztas"]:.0f} MWh ({cscs["ora"]}:00) | '
                           f'Napi: {holnap_df["fogyasztas"].sum()/1000:.1f} GWh</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Mini panel 2: Holnapi DAM árak — zöld/sárga/piros zónák
        with col2:
            st.markdown('<div class="mini-panel"><div class="mini-panel-cim">💰 Holnapi árazás (EUR/MWh)</div>', unsafe_allow_html=True)
            holnap_arak = edf[edf["datum"].dt.date == edf["datum"].dt.date.iloc[0]]
            if len(holnap_arak) > 0:
                labels_a = [d.strftime("%H:%M") for d in holnap_arak["datum"]]
                values_a = holnap_arak["dam_ar"].tolist()
                szinek = [dam_zona_szin(ar, dam_atlag_30) for ar in values_a]
                components.html(mini_bar_chart(values_a, labels_a, szinek, "mini_dam", 140),
                               height=155, scrolling=False)
                olcso = sum(1 for s in szinek if s == "#10b981")
                draga = sum(1 for s in szinek if s == "#ef4444")
                st.markdown(f'<div class="kartya-sub">🟢 {olcso} olcsó óra | 🔴 {draga} drága óra</div>',
                           unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Mini panel 3: Időjárás
        with col3:
            st.markdown('<div class="mini-panel"><div class="mini-panel-cim">🌡️ Holnapi időjárás</div>', unsafe_allow_html=True)
            holnap_ido = edf[edf["datum"].dt.date == edf["datum"].dt.date.iloc[0]]
            if len(holnap_ido) > 0:
                labels_i = [d.strftime("%H:%M") for d in holnap_ido["datum"]]
                values_i = holnap_ido["homerseklet"].tolist()
                components.html(mini_line_chart(values_i, labels_i, "rgb(16,185,129)", "mini_ido", 140, "°C"),
                               height=155, scrolling=False)
                st.markdown(f'<div class="kartya-sub">Min: {min(values_i):.0f}°C | Max: {max(values_i):.0f}°C | '
                           f'Átlag: {sum(values_i)/len(values_i):.0f}°C</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Modell info
        st.markdown("""
        <div style="background:#0a1628; border:1px solid #1e3a5f; border-radius:8px;
                    padding:8px 16px; margin-top:12px; text-align:center;">
            <span style="color:#64748b; font-size:11px; font-family:Inter,sans-serif;">
            XGBoost (FLAML AutoML) — MAE: 71.55 MWh — R²: 0.9768 — 19 feature, 100k+ tanítóminta
            </span>
        </div>""", unsafe_allow_html=True)

    # === TAB 2: ELŐREJELZÉS (placeholder) ===
    with tab2:
        st.markdown("""
        <div style="background:#0a1628; border:1px solid #1e3a5f; border-radius:12px;
                    padding:40px; text-align:center; margin-top:20px;">
            <div style="color:#FF6600; font-size:48px; margin-bottom:16px;">⏱️</div>
            <div style="color:#f1f5f9; font-size:18px; font-weight:600;
                        font-family:Montserrat,sans-serif; margin-bottom:8px;">
                Részletes 48 órás előrejelzés
            </div>
            <div style="color:#94a3b8; font-size:13px; font-family:Inter,sans-serif;">
                Következő lépés — órás bontású fogyasztás + költség + STL riasztás
            </div>
        </div>""", unsafe_allow_html=True)

    # === TAB 3: OKOS MÉRŐ (placeholder) ===
    with tab3:
        st.markdown("""
        <div style="background:#0a1628; border:1px solid #1e3a5f; border-radius:12px;
                    padding:40px; text-align:center; margin-top:20px;">
            <div style="color:#10b981; font-size:48px; margin-bottom:16px;">🔌</div>
            <div style="color:#f1f5f9; font-size:18px; font-weight:600;
                        font-family:Montserrat,sans-serif; margin-bottom:8px;">
                Okos Mérő Asszisztens
            </div>
            <div style="color:#94a3b8; font-size:13px; font-family:Inter,sans-serif;">
                Következő lépés — "Mikor olcsó az áram?" zónák, megtakarítás-kalkulátor
            </div>
        </div>""", unsafe_allow_html=True)

else:
    st.info("Kattints az 🔄 Előrejelzés frissítése gombra!")
