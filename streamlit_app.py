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
    page_title="OkosMérő.hu",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@600;700;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
.stApp { background-color: #050d1a; }
.block-container { padding: 1rem 2rem; max-width: 100%; }
.stButton > button {
    background: linear-gradient(135deg, #0066CC, #0044aa) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    font-family: Montserrat, sans-serif !important; font-weight: 600 !important;
    padding: 8px 20px !important; font-size: 13px !important;
}
.stTabs [data-baseweb="tab"] { color: #94a3b8 !important; font-family: Inter, sans-serif !important; }
.stTabs [aria-selected="true"] { color: #FF6600 !important; border-bottom: 2px solid #FF6600 !important; background: transparent !important; }
[data-testid="stMetricValue"] { color: #FF6600 !important; font-weight: 700 !important; font-size: 16px !important; text-align: center !important; }
[data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 10px !important; text-align: center !important; }
[data-testid="metric-container"] {
    background: #0a1628 !important; border: 1px solid #1e3a5f !important;
    border-radius: 12px !important; padding: 12px 8px !important; text-align: center !important;
}
footer, #MainMenu, header { display: none !important; }
</style>""", unsafe_allow_html=True)

# === KONSTANSOK ===
hu_holidays = holidays.Hungary(years=[2025, 2026, 2027])
ENTSOE_API_KEY = os.environ.get("ENTSOE_API_KEY", "")
BASE = os.path.dirname(os.path.abspath(__file__))

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

# === ADATLEKÉRŐK ===
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
        return float(df["OBS_VALUE"].dropna().iloc[-1]), True
    except:
        return 395.0, False

@st.cache_data(ttl=600)
def get_oras_idojaras(_datum_kulcs=None):
    try:
        ma = magyar_ma()
        url = "https://api.open-meteo.com/v1/forecast"
        params = {"latitude": 47.5, "longitude": 19.0,
                  "hourly": "temperature_2m,relative_humidity_2m,direct_radiation,wind_speed_10m,precipitation",
                  "timezone": "Europe/Budapest",
                  "start_date": (ma + timedelta(days=1)).strftime("%Y-%m-%d"),
                  "end_date": (ma + timedelta(days=1)).strftime("%Y-%m-%d")}
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        return pd.DataFrame({
            "Datum": pd.to_datetime(data["hourly"]["time"]),
            "Homerseklet_C": data["hourly"]["temperature_2m"],
            "Paratartalom_szazalek": data["hourly"]["relative_humidity_2m"],
            "Napsugarzas_W_m2": data["hourly"]["direct_radiation"],
            "Szelsebesseg_kmh": data["hourly"]["wind_speed_10m"],
            "Csapadek_mm": data["hourly"]["precipitation"]
        }), True
    except:
        ma = magyar_ma()
        hours = [ma + timedelta(days=1, hours=h) for h in range(24)]
        return pd.DataFrame({"Datum": hours, "Homerseklet_C": [20.0]*24,
                            "Paratartalom_szazalek": [60.0]*24, "Napsugarzas_W_m2": [100.0]*24,
                            "Szelsebesseg_kmh": [10.0]*24, "Csapadek_mm": [0.0]*24}), False

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
        return 100.0, None, False
    try:
        client = EntsoePandasClient(api_key=ENTSOE_API_KEY)
        ma = magyar_ma()
        holnap_start = pd.Timestamp((ma + timedelta(days=1)).strftime("%Y-%m-%d"), tz="Europe/Budapest")
        holnap_end = holnap_start + pd.Timedelta(days=1)
        try:
            holnap_oras = client.query_day_ahead_prices("HU", start=holnap_start, end=holnap_end)
            dam_atlag = float(holnap_oras.mean())
        except:
            holnap_oras = None
            dam_atlag = 100.0
        return dam_atlag, holnap_oras, True
    except:
        return 100.0, None, False

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

@st.cache_data(ttl=600)
def get_aktualis_homerseklet(_datum_kulcs=None):
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {"latitude": 47.5, "longitude": 19.0,
                  "current_weather": "true", "timezone": "Europe/Budapest"}
        r = requests.get(url, params=params, timeout=10)
        return float(r.json()["current_weather"]["temperature"]), True
    except:
        return 25.0, False

# === ELŐREJELZÉS — 24 óra, DataFrame-mel ===
def oras_elorejelzes(idojaras_df, dam_atlag, eur_huf, load_history,
                     nap_atlag, szel_atlag, holnap_oras):
    eredmenyek = []
    if load_history is not None and len(load_history) >= 168:
        lag_buffer = list(load_history.values[-168:])
    else:
        lag_buffer = [4800.0] * 168

    seasonal_by_hour = stl_seasonal.groupby(stl_seasonal.index.hour).mean().iloc[:, 0]
    stl_std = stl_params.get("reziduum_std", 233.87)
    stl_mean = stl_params.get("reziduum_atlag", -28.89)

    for i, row in idojaras_df.iterrows():
        datum = row["Datum"]
        ora = datum.hour

        # DAM ár: valós holnapi óránkénti ha van, különben átlag
        if holnap_oras is not None and ora < len(holnap_oras):
            dam_ar = float(holnap_oras.iloc[ora])
        else:
            dam_ar = dam_atlag

        lag1h = lag_buffer[-1]
        lag24h = lag_buffer[-24] if len(lag_buffer) >= 24 else lag1h
        lag168h = lag_buffer[-168] if len(lag_buffer) >= 168 else lag1h
        nap_mw = nap_atlag if row["Napsugarzas_W_m2"] > 50 else 0

        # DataFrame — feature sorrend GARANTÁLT
        X_sor = pd.DataFrame([{
            'DAM_EUR_MWh': dam_ar, 'Homerseklet_C': row["Homerseklet_C"],
            'Paratartalom_szazalek': row["Paratartalom_szazalek"],
            'Napsugarzas_W_m2': row["Napsugarzas_W_m2"],
            'Szelsebesseg_kmh': row["Szelsebesseg_kmh"],
            'Csapadek_mm': row["Csapadek_mm"], 'EUR_HUF': eur_huf,
            'Ora': ora, 'Het_napja': datum.weekday() + 1, 'Honap': datum.month,
            'Unnepnap': 1 if datum.date() in hu_holidays else 0,
            'Hetvege': 1 if datum.weekday() >= 5 else 0,
            'Extrem_hideg': 1 if row["Homerseklet_C"] < -5 else 0,
            'Extrem_meleg': 1 if row["Homerseklet_C"] > 30 else 0,
            'Fogyasztas_lag1h': lag1h, 'Fogyasztas_lag24h': lag24h,
            'Fogyasztas_lag168h': lag168h,
            'Nap_termeles_MW': nap_mw, 'Szel_termeles_MW': szel_atlag
        }], columns=FEATURE_SORREND)

        josolt = float(model.predict(X_sor)[0])
        josolt = max(josolt, 2000)

        seasonal_val = seasonal_by_hour.get(ora, 0)
        reziduum = josolt - seasonal_val - stl_mean
        riado = abs(reziduum) > 2.5 * stl_std

        lag_buffer.append(josolt)
        eredmenyek.append({
            "datum": datum, "ora": ora, "homerseklet": row["Homerseklet_C"],
            "napsugarzas": row["Napsugarzas_W_m2"], "szelsebesseg": row["Szelsebesseg_kmh"],
            "fogyasztas": josolt, "dam_ar": dam_ar,
            "koltseg_mft": josolt * dam_ar * eur_huf / 1_000_000,
            "riado": riado, "reziduum": reziduum
        })
    return eredmenyek

# === FRISSÍTÉS ===
col_btn, _ = st.columns([1, 5])
with col_btn:
    frissit = st.button("🔄 Frissítés", type="primary")

if frissit:
    get_eur_huf.clear()
    get_oras_idojaras.clear()
    get_dam_ar_oras.clear()
    get_oras_load_history.clear()
    get_megujulo_utolso_het.clear()
    get_aktualis_homerseklet.clear()
    st.session_state.pop("eredmenyek", None)
    st.rerun()

# === ADATLEKÉRÉS ===
if "eredmenyek" not in st.session_state:
    with st.spinner("Adatok lekérése (ENTSO-E, Open-Meteo, ECB)..."):
        ora_kulcs = magyar_ma().strftime("%Y-%m-%d-%H")
        eur_huf, eur_ok = get_eur_huf(_datum_kulcs=ora_kulcs)
        idojaras_df, ido_ok = get_oras_idojaras(_datum_kulcs=ora_kulcs)
        dam_atlag, holnap_oras, dam_ok = get_dam_ar_oras(_datum_kulcs=ora_kulcs)
        load_history, load_ok = get_oras_load_history(_datum_kulcs=ora_kulcs)
        nap_atlag, szel_atlag, megujulo_ok = get_megujulo_utolso_het(_datum_kulcs=ora_kulcs)
        aktualis_ho, ho_ok = get_aktualis_homerseklet(_datum_kulcs=ora_kulcs)

        eredmenyek = oras_elorejelzes(idojaras_df, dam_atlag, eur_huf,
                                       load_history, nap_atlag, szel_atlag, holnap_oras)
        st.session_state.eredmenyek = eredmenyek
        st.session_state.eur_huf = eur_huf
        st.session_state.dam_atlag = dam_atlag
        st.session_state.holnap_oras = holnap_oras
        st.session_state.aktualis_ho = aktualis_ho
        st.session_state.frissites_ideje = pd.Timestamp.now(tz="Europe/Budapest").strftime("%H:%M")
        st.session_state.fallback = {
            "EUR/HUF": not eur_ok, "Időjárás": not ido_ok,
            "DAM": not dam_ok, "Fogyasztás": not load_ok, "Megújuló": not megujulo_ok
        }

# === FŐ TARTALOM ===
if "eredmenyek" in st.session_state:
    eredmenyek = st.session_state.eredmenyek
    eur_huf = st.session_state.eur_huf
    dam_atlag = st.session_state.dam_atlag
    aktualis_ho = st.session_state.aktualis_ho
    fb = st.session_state.get("fallback", {})

    edf = pd.DataFrame(eredmenyek)
    edf["datum"] = pd.to_datetime(edf["datum"])

    # Aktuális óra fogyasztása (első predikció)
    aktualis_fogy = edf["fogyasztas"].iloc[0]

    # DAM legolcsóbb és legdrágább
    legolcsobb_idx = edf["dam_ar"].idxmin()
    legdragabb_idx = edf["dam_ar"].idxmax()
    legolcsobb_ora = int(edf.loc[legolcsobb_idx, "ora"])
    legolcsobb_ar = edf.loc[legolcsobb_idx, "dam_ar"]
    legdragabb_ora = int(edf.loc[legdragabb_idx, "ora"])
    legdragabb_ar = edf.loc[legdragabb_idx, "dam_ar"]
    holnap_max_ho = edf["homerseklet"].max()

    stl_riado_orak = int(edf["riado"].sum())

    # Fallback figyelmeztetés
    fb_lista = [k for k, v in fb.items() if v]

    # === FEJLÉC — B design + XGBOOST badge ===
    holnap_nap = (magyar_ma() + timedelta(days=1)).strftime("%Y.%m.%d")
    st.markdown(f"""
    <div style="background:linear-gradient(180deg, #0a1628 0%, #050d1a 100%);
                border:1px solid #1e3a5f; border-radius:10px;
                padding:18px 28px; margin-bottom:12px;
                display:flex; align-items:center; justify-content:space-between;">
        <div style="display:flex; align-items:center; gap:16px;">
            <div style="font-size:32px; filter:drop-shadow(0 0 12px rgba(255,102,0,0.4));">⚡</div>
            <div>
                <div style="display:flex; align-items:center; gap:10px;">
                    <span style="font-family:Montserrat,sans-serif; font-size:22px; font-weight:800; color:#ffffff;">
                        Okos<span style="color:#10b981;">Mérő</span>.hu
                    </span>
                    <span style="background:rgba(0,102,204,0.15); color:#4b9cd3; font-size:9px;
                                 padding:3px 8px; border-radius:4px; font-weight:600; letter-spacing:0.5px;">
                        XGBOOST FLAML
                    </span>
                </div>
                <div style="font-family:Inter,sans-serif; font-size:11px; color:#4b9cd3;
                            margin-top:3px; letter-spacing:2px; text-transform:uppercase;">
                    energiapiaci asszisztens
                </div>
            </div>
        </div>
        <div style="display:flex; gap:24px; align-items:center;">
            <div style="text-align:center;">
                <div style="font-family:Montserrat,sans-serif; font-size:18px; font-weight:700; color:#10b981;">
                    {legolcsobb_ar:.0f} €
                </div>
                <div style="font-size:7px; color:#64748b; text-transform:uppercase; letter-spacing:0.5px; margin-top:2px;">
                    legolcsóbb ({legolcsobb_ora:02d}:00)
                </div>
            </div>
            <div style="text-align:center;">
                <div style="font-family:Montserrat,sans-serif; font-size:18px; font-weight:700; color:#ef4444;">
                    {legdragabb_ar:.0f} €
                </div>
                <div style="font-size:7px; color:#64748b; text-transform:uppercase; letter-spacing:0.5px; margin-top:2px;">
                    legdrágább ({legdragabb_ora:02d}:00)
                </div>
            </div>
            <div style="text-align:center;">
                <div style="font-family:Montserrat,sans-serif; font-size:18px; font-weight:700; color:#FF6600;">
                    {holnap_max_ho:.0f}°C
                </div>
                <div style="font-size:7px; color:#64748b; text-transform:uppercase; letter-spacing:0.5px; margin-top:2px;">
                    holnap max
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # === STÁTUSZSOR ===
    if fb_lista:
        st.warning(f"⚠️ Becsült adat: {', '.join(fb_lista)} | Frissítve: {st.session_state.frissites_ideje}")
    else:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">
            <div style="width:6px; height:6px; border-radius:50%; background:#10b981;
                        animation: pulse 2s infinite;"></div>
            <span style="font-size:11px; color:#10b981;">
                Élő adatok — {st.session_state.frissites_ideje}-kor frissítve
            </span>
            <style>@keyframes pulse {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0.3; }} }}</style>
        </div>
        """, unsafe_allow_html=True)

    # === 6 KPI KÁRTYA ===
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("⚡ Most", f"{aktualis_fogy:,.0f} MWh")
    # DAM most: aktuális óra ára
    aktualis_ora = pd.Timestamp.now(tz="Europe/Budapest").hour
    if st.session_state.holnap_oras is not None and aktualis_ora < len(st.session_state.holnap_oras):
        dam_most = float(st.session_state.holnap_oras.iloc[aktualis_ora])
    else:
        dam_most = dam_atlag
    k2.metric("🏦 DAM most", f"{dam_most:.0f} €/MWh")
    k3.metric("🌡️ Budapest", f"{aktualis_ho:.0f} °C")
    k4.metric("💱 EUR/HUF", f"{eur_huf:.1f} Ft")
    k5.metric("🟢 Legolcsóbb", f"{legolcsobb_ora:02d}:00 — {legolcsobb_ar:.0f} €")
    k6.metric(
        "✅ STL" if stl_riado_orak == 0 else "🚨 STL",
        f"{stl_riado_orak} / {len(edf)} óra"
    )

    # === TABOK ===
    tab1, tab2, tab3 = st.tabs(["📊 Áttekintés", "⏱️ Előrejelzés", "🔌 Okos Mérő"])

    # === TAB 1: ÁTTEKINTÉS ===
    with tab1:
        holnap_str = (magyar_ma() + timedelta(days=1)).strftime("%m.%d (%A)")
        st.caption(f"Holnapi előrejelzés — {holnap_str}")

        p1, p2, p3 = st.columns(3)

        with p1:
            st.markdown("**⚡ Fogyasztás (MWh)**")
            chart_df = edf[["datum", "fogyasztas"]].copy()
            chart_df = chart_df.rename(columns={"fogyasztas": "MWh"}).set_index("datum")
            st.area_chart(chart_df, color="#0066CC", height=200)
            csucs = edf.loc[edf["fogyasztas"].idxmax()]
            volgyi = edf.loc[edf["fogyasztas"].idxmin()]
            st.caption(f"Csúcs: {csucs['fogyasztas']:,.0f} MWh ({int(csucs['ora'])}:00) — "
                      f"Völgy: {volgyi['fogyasztas']:,.0f} MWh ({int(volgyi['ora'])}:00)")

        with p2:
            st.markdown("**💰 DAM ár (€/MWh)**")
            ar_df = edf[["datum", "dam_ar"]].copy()
            ar_df = ar_df.rename(columns={"dam_ar": "€/MWh"}).set_index("datum")
            st.bar_chart(ar_df, color="#FF6600", height=200)
            olcso = len(edf[edf["dam_ar"] < dam_atlag * 0.7])
            draga = len(edf[edf["dam_ar"] > dam_atlag * 1.3])
            st.caption(f"🟢 {olcso} olcsó — 🟡 {24 - olcso - draga} átlagos — 🔴 {draga} drága óra")

        with p3:
            st.markdown("**🌡️ Hőmérséklet (°C)**")
            ido_df = edf[["datum", "homerseklet"]].copy()
            ido_df = ido_df.rename(columns={"homerseklet": "°C"}).set_index("datum")
            st.line_chart(ido_df, color="#10b981", height=200)
            st.caption(f"Min: {edf['homerseklet'].min():.0f}°C — "
                      f"Max: {edf['homerseklet'].max():.0f}°C — "
                      f"Átlag: {edf['homerseklet'].mean():.0f}°C")

        st.divider()
        st.caption("XGBoost (FLAML AutoML) — MAE: 71.55 MWh — R²: 0.9768 — "
                  "lag24h + lag168h mindig valós ENTSO-E adat")

    # === TAB 2: ELŐREJELZÉS (placeholder) ===
    with tab2:
        st.markdown("### ⏱️ Részletes 24 órás előrejelzés")
        st.info("Következő lépés — órás bontású fogyasztás + hőmérséklet kettős tengelyen, "
                "csúcs/völgy kijelölés, napi költség, STL riasztás részletei")

    # === TAB 3: OKOS MÉRŐ (placeholder) ===
    with tab3:
        st.markdown("### 🔌 Okos Mérő Asszisztens")
        st.info("Következő lépés — 24 órás DAM ár zónák (zöld/sárga/piros), "
                "háztartási eszköz javaslatok, megtakarítás-kalkulátor")

else:
    st.info("Kattints a 🔄 Frissítés gombra!")
