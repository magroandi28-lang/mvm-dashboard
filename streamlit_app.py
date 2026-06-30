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

st.set_page_config(page_title="OkosMérő.hu", page_icon="⚡", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@600;700;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
.stApp { background-color: #050d1a; }
.block-container { padding: 1rem 2rem; max-width: 100%; }
.stButton > button {
    background: linear-gradient(135deg, #0066CC, #0044aa) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    font-family: Montserrat, sans-serif !important; font-weight: 600 !important;
    padding: 8px 20px !important;
}
.stTabs [data-baseweb="tab"] { color: #94a3b8 !important; font-family: Inter, sans-serif !important; }
.stTabs [aria-selected="true"] { color: #FF6600 !important; border-bottom: 2px solid #FF6600 !important; background: transparent !important; }
footer, #MainMenu, header { display: none !important; }
</style>""", unsafe_allow_html=True)

hu_holidays = holidays.Hungary(years=[2025, 2026, 2027])
ENTSOE_API_KEY = os.environ.get("ENTSOE_API_KEY", "")
BASE = os.path.dirname(os.path.abspath(__file__))
FEATURE_SORREND = [
    'DAM_EUR_MWh', 'Homerseklet_C', 'Paratartalom_szazalek', 'Napsugarzas_W_m2',
    'Szelsebesseg_kmh', 'Csapadek_mm', 'EUR_HUF', 'Ora', 'Het_napja', 'Honap',
    'Unnepnap', 'Hetvege', 'Extrem_hideg', 'Extrem_meleg',
    'Fogyasztas_lag1h', 'Fogyasztas_lag24h', 'Fogyasztas_lag168h',
    'Nap_termeles_MW', 'Szel_termeles_MW']

def magyar_ma():
    return pd.Timestamp.now(tz="Europe/Budapest").normalize().tz_localize(None).to_pydatetime()

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
def get_eur_huf(_dk=None):
    try:
        r = requests.get("https://data-api.ecb.europa.eu/service/data/EXR/D.HUF.EUR.SP00.A",
            params={"startPeriod": (datetime.now()-timedelta(days=7)).strftime("%Y-%m-%d"),
                    "endPeriod": datetime.now().strftime("%Y-%m-%d"), "format": "csvdata"}, timeout=10)
        df = pd.read_csv(StringIO(r.text))[["TIME_PERIOD","OBS_VALUE"]].dropna()
        df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
        return float(df["OBS_VALUE"].dropna().iloc[-1]), True
    except: return 395.0, False

@st.cache_data(ttl=600)
def get_oras_idojaras(_dk=None):
    try:
        ma = magyar_ma()
        r = requests.get("https://api.open-meteo.com/v1/forecast",
            params={"latitude":47.5,"longitude":19.0,
                    "hourly":"temperature_2m,relative_humidity_2m,direct_radiation,wind_speed_10m,precipitation",
                    "timezone":"Europe/Budapest",
                    "start_date":(ma+timedelta(days=1)).strftime("%Y-%m-%d"),
                    "end_date":(ma+timedelta(days=1)).strftime("%Y-%m-%d")}, timeout=15)
        d = r.json()
        return pd.DataFrame({"Datum":pd.to_datetime(d["hourly"]["time"]),
            "Homerseklet_C":d["hourly"]["temperature_2m"],
            "Paratartalom_szazalek":d["hourly"]["relative_humidity_2m"],
            "Napsugarzas_W_m2":d["hourly"]["direct_radiation"],
            "Szelsebesseg_kmh":d["hourly"]["wind_speed_10m"],
            "Csapadek_mm":d["hourly"]["precipitation"]}), True
    except:
        ma = magyar_ma()
        h = [ma+timedelta(days=1,hours=i) for i in range(24)]
        return pd.DataFrame({"Datum":h,"Homerseklet_C":[20]*24,"Paratartalom_szazalek":[60]*24,
            "Napsugarzas_W_m2":[100]*24,"Szelsebesseg_kmh":[10]*24,"Csapadek_mm":[0]*24}), False

@st.cache_data(ttl=600)
def get_oras_load_history(_dk=None):
    if not ENTSOE_API_KEY: return None, False
    try:
        c = EntsoePandasClient(api_key=ENTSOE_API_KEY); ma = magyar_ma()
        s = pd.Timestamp((ma-timedelta(days=8)).strftime("%Y-%m-%d"),tz="Europe/Budapest")
        e = pd.Timestamp(ma.strftime("%Y-%m-%d"),tz="Europe/Budapest")
        load = c.query_load("HU",start=s,end=e)
        if isinstance(load,pd.DataFrame): load=load.iloc[:,0]
        return load, True
    except: return None, False

@st.cache_data(ttl=300)
def get_dam_ar_oras(_dk=None):
    if not ENTSOE_API_KEY: return 100.0, None, False
    try:
        c = EntsoePandasClient(api_key=ENTSOE_API_KEY); ma = magyar_ma()
        hs = pd.Timestamp((ma+timedelta(days=1)).strftime("%Y-%m-%d"),tz="Europe/Budapest")
        try:
            ho = c.query_day_ahead_prices("HU",start=hs,end=hs+pd.Timedelta(days=1))
            da = float(ho.mean())
        except: ho=None; da=100.0
        return da, ho, True
    except: return 100.0, None, False

@st.cache_data(ttl=600)
def get_megujulo(_dk=None):
    if not ENTSOE_API_KEY: return 500.0, 200.0, False
    try:
        c = EntsoePandasClient(api_key=ENTSOE_API_KEY); ma = magyar_ma()
        s = pd.Timestamp((ma-timedelta(days=7)).strftime("%Y-%m-%d"),tz="Europe/Budapest")
        e = pd.Timestamp(ma.strftime("%Y-%m-%d"),tz="Europe/Budapest")
        g = c.query_generation("HU",start=s,end=e)
        sc = [x for x in g.columns if 'Solar' in str(x)]
        wc = [x for x in g.columns if 'Wind' in str(x)]
        return (float(g[sc].sum(1).mean()) if sc else 500.0,
                float(g[wc].sum(1).mean()) if wc else 200.0, True)
    except: return 500.0, 200.0, False

@st.cache_data(ttl=600)
def get_aktualis_ho(_dk=None):
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast",
            params={"latitude":47.5,"longitude":19.0,"current_weather":"true",
                    "timezone":"Europe/Budapest"}, timeout=10)
        return float(r.json()["current_weather"]["temperature"]), True
    except: return 25.0, False

# === ELŐREJELZÉS ===
def oras_elorejelzes(idojaras_df, dam_atlag, eur_huf, load_history,
                     nap_atlag, szel_atlag, holnap_oras):
    eredmenyek = []
    lag_buffer = list(load_history.values[-168:]) if load_history is not None and len(load_history)>=168 else [4800.0]*168
    seasonal_by_hour = stl_seasonal.groupby(stl_seasonal.index.hour).mean().iloc[:,0]
    stl_std = stl_params.get("reziduum_std", 233.87)
    stl_mean = stl_params.get("reziduum_atlag", -28.89)
    # Trend szint becslés: utolsó ismert lag átlag - seasonal átlag
    trend_szint = np.mean(lag_buffer[-24:]) - seasonal_by_hour.mean()

    for i, row in idojaras_df.iterrows():
        datum = row["Datum"]; ora = datum.hour
        dam_ar = float(holnap_oras.iloc[ora]) if holnap_oras is not None and ora < len(holnap_oras) else dam_atlag
        lag1h = lag_buffer[-1]
        lag24h = lag_buffer[-24] if len(lag_buffer)>=24 else lag1h
        lag168h = lag_buffer[-168] if len(lag_buffer)>=168 else lag1h

        X_sor = pd.DataFrame([{
            'DAM_EUR_MWh':dam_ar,'Homerseklet_C':row["Homerseklet_C"],
            'Paratartalom_szazalek':row["Paratartalom_szazalek"],
            'Napsugarzas_W_m2':row["Napsugarzas_W_m2"],
            'Szelsebesseg_kmh':row["Szelsebesseg_kmh"],
            'Csapadek_mm':row["Csapadek_mm"],'EUR_HUF':eur_huf,
            'Ora':ora,'Het_napja':datum.weekday()+1,'Honap':datum.month,
            'Unnepnap':1 if datum.date() in hu_holidays else 0,
            'Hetvege':1 if datum.weekday()>=5 else 0,
            'Extrem_hideg':1 if row["Homerseklet_C"]<-5 else 0,
            'Extrem_meleg':1 if row["Homerseklet_C"]>30 else 0,
            'Fogyasztas_lag1h':lag1h,'Fogyasztas_lag24h':lag24h,
            'Fogyasztas_lag168h':lag168h,
            'Nap_termeles_MW':nap_atlag if row["Napsugarzas_W_m2"]>50 else 0,
            'Szel_termeles_MW':szel_atlag
        }], columns=FEATURE_SORREND)

        josolt = max(float(model.predict(X_sor)[0]), 2000)
        # STL: reziduum = jósolt - trend - seasonal
        seasonal_val = seasonal_by_hour.get(ora, 0)
        reziduum = josolt - trend_szint - seasonal_val
        riado = abs(reziduum - stl_mean) > 2.5 * stl_std

        lag_buffer.append(josolt)
        eredmenyek.append({"datum":datum,"ora":ora,"homerseklet":row["Homerseklet_C"],
            "napsugarzas":row["Napsugarzas_W_m2"],"szelsebesseg":row["Szelsebesseg_kmh"],
            "fogyasztas":josolt,"dam_ar":dam_ar,
            "koltseg_mft":josolt*dam_ar*eur_huf/1_000_000,
            "riado":riado,"reziduum":reziduum})
    return eredmenyek

# === PLOTLY HELPER ===
def plotly_html(data_json, layout_json, cid, h=220):
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
<style>*{{margin:0;padding:0}}body{{background:transparent}}#{cid}{{width:100%;height:{h}px}}</style>
</head><body><div id="{cid}"></div>
<script>Plotly.newPlot('{cid}',{data_json},{layout_json},{{responsive:true,displayModeBar:false}});</script>
</body></html>"""

def plotly_layout_alap():
    return {"paper_bgcolor":"rgba(0,0,0,0)","plot_bgcolor":"rgba(10,22,40,0.5)",
            "font":{"color":"#cbd5e1","family":"Inter","size":10},
            "margin":{"l":45,"r":15,"t":10,"b":35},
            "xaxis":{"gridcolor":"#1e3a5f","color":"#94a3b8"},
            "yaxis":{"gridcolor":"#1e3a5f","color":"#94a3b8"},
            "showlegend":False}

# === FRISSÍTÉS + ADATLEKÉRÉS ===
if st.button("🔄 Frissítés", type="primary"):
    for fn in [get_eur_huf,get_oras_idojaras,get_dam_ar_oras,get_oras_load_history,get_megujulo,get_aktualis_ho]:
        fn.clear()
    st.session_state.pop("eredmenyek",None)
    st.rerun()

if "eredmenyek" not in st.session_state:
    with st.spinner("Adatok lekérése..."):
        dk = magyar_ma().strftime("%Y-%m-%d-%H")
        eur_huf,eur_ok = get_eur_huf(_dk=dk)
        ido_df,ido_ok = get_oras_idojaras(_dk=dk)
        dam_atlag,holnap_oras,dam_ok = get_dam_ar_oras(_dk=dk)
        lh,load_ok = get_oras_load_history(_dk=dk)
        na,sa,meg_ok = get_megujulo(_dk=dk)
        aho,ho_ok = get_aktualis_ho(_dk=dk)
        ered = oras_elorejelzes(ido_df,dam_atlag,eur_huf,lh,na,sa,holnap_oras)
        st.session_state.update({"eredmenyek":ered,"eur_huf":eur_huf,"dam_atlag":dam_atlag,
            "holnap_oras":holnap_oras,"aktualis_ho":aho,
            "frissites":pd.Timestamp.now(tz="Europe/Budapest").strftime("%H:%M"),
            "fallback":{"EUR/HUF":not eur_ok,"Időjárás":not ido_ok,"DAM":not dam_ok,
                        "Fogyasztás":not load_ok,"Megújuló":not meg_ok}})

if "eredmenyek" not in st.session_state:
    st.info("Kattints a 🔄 Frissítés gombra!")
    st.stop()

# === ADATOK ELŐKÉSZÍTÉS ===
edf = pd.DataFrame(st.session_state.eredmenyek)
edf["datum"] = pd.to_datetime(edf["datum"])
eur_huf = st.session_state.eur_huf
dam_atlag = st.session_state.dam_atlag
aho = st.session_state.aktualis_ho
fb = st.session_state.get("fallback",{})

legolcsobb = edf.loc[edf["dam_ar"].idxmin()]
legdragabb = edf.loc[edf["dam_ar"].idxmax()]
csucs = edf.loc[edf["fogyasztas"].idxmax()]
volgy = edf.loc[edf["fogyasztas"].idxmin()]
stl_riado = int(edf["riado"].sum())
holnap_max_ho = edf["homerseklet"].max()

# === FEJLÉC ===
st.markdown(f"""
<div style="background:linear-gradient(180deg,#0a1628,#050d1a); border:1px solid #1e3a5f;
            border-radius:10px; padding:18px 28px; margin-bottom:4px;
            display:flex; align-items:center; justify-content:space-between;">
    <div style="display:flex; align-items:center; gap:16px;">
        <div style="font-size:32px; filter:drop-shadow(0 0 12px rgba(255,102,0,0.4));">⚡</div>
        <div>
            <div style="display:flex; align-items:center; gap:10px;">
                <span style="font-family:Montserrat; font-size:22px; font-weight:800; color:#fff;">
                    Okos<span style="color:#10b981;">Mérő</span>.hu</span>
                <span style="background:rgba(0,102,204,0.15); color:#4b9cd3; font-size:9px;
                             padding:3px 8px; border-radius:4px; font-weight:600;">XGBOOST FLAML</span>
            </div>
            <div style="font-family:Inter; font-size:11px; color:#4b9cd3; margin-top:3px;
                        letter-spacing:2px; text-transform:uppercase;">energiapiaci asszisztens</div>
        </div>
    </div>
    <div style="display:flex; gap:24px;">
        <div style="text-align:center;">
            <div style="font-family:Montserrat; font-size:18px; font-weight:700; color:#10b981;">
                {legolcsobb['dam_ar']:.0f} €</div>
            <div style="font-size:7px; color:#64748b; text-transform:uppercase; margin-top:2px;">
                legolcsóbb ({int(legolcsobb['ora']):02d}:00)</div>
        </div>
        <div style="text-align:center;">
            <div style="font-family:Montserrat; font-size:18px; font-weight:700; color:#ef4444;">
                {legdragabb['dam_ar']:.0f} €</div>
            <div style="font-size:7px; color:#64748b; text-transform:uppercase; margin-top:2px;">
                legdrágább ({int(legdragabb['ora']):02d}:00)</div>
        </div>
        <div style="text-align:center;">
            <div style="font-family:Montserrat; font-size:18px; font-weight:700; color:#FF6600;">
                {holnap_max_ho:.0f}°C</div>
            <div style="font-size:7px; color:#64748b; text-transform:uppercase; margin-top:2px;">holnap max</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# === STÁTUSZ ===
fb_lista = [k for k,v in fb.items() if v]
if fb_lista:
    st.markdown(f'<div style="font-size:11px;color:#f59e0b;margin:4px 0 8px;">⚠️ Becsült: {", ".join(fb_lista)} — {st.session_state.frissites}-kor frissítve</div>', unsafe_allow_html=True)
else:
    st.markdown(f"""<div style="display:flex;align-items:center;gap:8px;margin:4px 0 8px;">
        <div style="width:6px;height:6px;border-radius:50%;background:#10b981;animation:p 2s infinite;"></div>
        <span style="font-size:11px;color:#10b981;">Élő adatok — {st.session_state.frissites}-kor frissítve</span>
        <style>@keyframes p{{0%,100%{{opacity:1}}50%{{opacity:.3}}}}</style></div>""", unsafe_allow_html=True)

# === 6 KÁRTYA ===
dam_most_ar = dam_atlag
if st.session_state.holnap_oras is not None:
    ora_most = pd.Timestamp.now(tz="Europe/Budapest").hour
    if ora_most < len(st.session_state.holnap_oras):
        dam_most_ar = float(st.session_state.holnap_oras.iloc[ora_most])

def dam_szin(ar, atlag):
    if ar < atlag * 0.7: return "#10b981"
    elif ar > atlag * 1.3: return "#ef4444"
    return "#f59e0b"

riado_szin = "#10b981" if stl_riado == 0 else "#FF6600"
riado_ikon = "✅" if stl_riado == 0 else "🚨"
dam_sz = dam_szin(dam_most_ar, dam_atlag)

st.markdown(f"""
<div style="display:grid; grid-template-columns:repeat(6,1fr); gap:8px; margin-bottom:12px;">
  <div style="background:#0a1628;border:1px solid #1e3a5f;border-radius:10px;padding:10px 8px;text-align:center;">
    <div style="color:#64748b;font-size:8px;text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px;">⚡ Most</div>
    <div style="color:#FF6600;font-size:15px;font-weight:700;font-family:Montserrat;">{edf['fogyasztas'].iloc[0]:,.0f} MWh</div>
    <div style="color:#94a3b8;font-size:9px;margin-top:2px;">aktuális fogyasztás</div>
  </div>
  <div style="background:#0a1628;border:1px solid {dam_sz};border-radius:10px;padding:10px 8px;text-align:center;">
    <div style="color:#64748b;font-size:8px;text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px;">🏦 DAM most</div>
    <div style="color:{dam_sz};font-size:15px;font-weight:700;font-family:Montserrat;">{dam_most_ar:.0f} €/MWh</div>
    <div style="color:{dam_sz};font-size:9px;margin-top:2px;">{'olcsó' if dam_sz=='#10b981' else 'drága' if dam_sz=='#ef4444' else 'átlagos'}</div>
  </div>
  <div style="background:#0a1628;border:1px solid #1e3a5f;border-radius:10px;padding:10px 8px;text-align:center;">
    <div style="color:#64748b;font-size:8px;text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px;">🌡️ Budapest</div>
    <div style="color:#FF6600;font-size:15px;font-weight:700;font-family:Montserrat;">{aho:.0f} °C</div>
    <div style="color:#94a3b8;font-size:9px;margin-top:2px;">most</div>
  </div>
  <div style="background:#0a1628;border:1px solid #1e3a5f;border-radius:10px;padding:10px 8px;text-align:center;">
    <div style="color:#64748b;font-size:8px;text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px;">💱 EUR/HUF</div>
    <div style="color:#FF6600;font-size:15px;font-weight:700;font-family:Montserrat;">{eur_huf:.1f} Ft</div>
  </div>
  <div style="background:#0a1628;border:1px solid #10b981;border-radius:10px;padding:10px 8px;text-align:center;">
    <div style="color:#64748b;font-size:8px;text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px;">🟢 Legolcsóbb</div>
    <div style="color:#10b981;font-size:15px;font-weight:700;font-family:Montserrat;">{int(legolcsobb['ora']):02d}:00 — {legolcsobb['dam_ar']:.0f} €</div>
    <div style="color:#10b981;font-size:9px;margin-top:2px;">holnap</div>
  </div>
  <div style="background:#0a1628;border:1px solid {riado_szin};border-radius:10px;padding:10px 8px;text-align:center;">
    <div style="color:#64748b;font-size:8px;text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px;">{riado_ikon} STL</div>
    <div style="color:{riado_szin};font-size:15px;font-weight:700;font-family:Montserrat;">{stl_riado} / {len(edf)}</div>
    <div style="color:#94a3b8;font-size:9px;margin-top:2px;">óra</div>
  </div>
</div>
""", unsafe_allow_html=True)

# === TABOK ===
tab1, tab2, tab3 = st.tabs(["📊 Áttekintés", "⏱️ Előrejelzés", "🔌 Okos Mérő"])

with tab1:
    holnap_str = (magyar_ma()+timedelta(days=1)).strftime("%m.%d")
    st.markdown(f'<div style="color:#94a3b8;font-size:10px;margin-bottom:8px;text-transform:uppercase;letter-spacing:.5px;">Holnapi előrejelzés — {holnap_str}</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    # Panel 1: Fogyasztás
    with c1:
        st.markdown('<div style="color:#f1f5f9;font-size:12px;font-weight:600;font-family:Montserrat;margin-bottom:4px;">⚡ Fogyasztás (MWh)</div>', unsafe_allow_html=True)
        labels = [f"{int(r['ora']):02d}:00" for _,r in edf.iterrows()]
        vals = edf["fogyasztas"].tolist()
        lay = plotly_layout_alap()
        lay["yaxis"]["title"] = "MWh"
        data = json.dumps([{"type":"scatter","x":labels,"y":vals,"mode":"lines",
            "line":{"color":"#0066CC","width":2},"fill":"tozeroy",
            "fillcolor":"rgba(0,102,204,0.15)",
            "hovertemplate":"%{x}<br>%{y:,.0f} MWh<extra></extra>"}])
        components.html(plotly_html(data, json.dumps(lay), "fogy"), height=235, scrolling=False)
        st.markdown(f'<div style="color:#94a3b8;font-size:9px;">Csúcs: {csucs["fogyasztas"]:,.0f} MWh ({int(csucs["ora"])}:00) — Völgy: {volgy["fogyasztas"]:,.0f} MWh ({int(volgy["ora"])}:00)</div>', unsafe_allow_html=True)

    # Panel 2: DAM ár zónák
    with c2:
        st.markdown('<div style="color:#f1f5f9;font-size:12px;font-weight:600;font-family:Montserrat;margin-bottom:4px;">💰 DAM ár zónák (€/MWh)</div>', unsafe_allow_html=True)
        ar_labels = [f"{int(r['ora']):02d}:00" for _,r in edf.iterrows()]
        ar_vals = edf["dam_ar"].tolist()
        ar_colors = [dam_szin(a, dam_atlag) for a in ar_vals]
        lay2 = plotly_layout_alap()
        lay2["yaxis"]["title"] = "€/MWh"
        lay2["bargap"] = 0.3
        data2 = json.dumps([{"type":"bar","x":ar_labels,"y":ar_vals,
            "marker":{"color":ar_colors,"opacity":0.85},
            "hovertemplate":"%{x}<br>%{y:.0f} €/MWh<extra></extra>"}])
        components.html(plotly_html(data2, json.dumps(lay2), "dam"), height=235, scrolling=False)
        olcso = sum(1 for c in ar_colors if c=="#10b981")
        draga = sum(1 for c in ar_colors if c=="#ef4444")
        st.markdown(f'<div style="color:#94a3b8;font-size:9px;">🟢 {olcso} olcsó — 🟡 {24-olcso-draga} átlagos — 🔴 {draga} drága óra</div>', unsafe_allow_html=True)

    # Panel 3: Időjárás
    with c3:
        st.markdown('<div style="color:#f1f5f9;font-size:12px;font-weight:600;font-family:Montserrat;margin-bottom:4px;">🌡️ Holnapi időjárás</div>', unsafe_allow_html=True)
        ho_labels = [f"{int(r['ora']):02d}:00" for _,r in edf.iterrows()]
        ho_vals = edf["homerseklet"].tolist()
        lay3 = plotly_layout_alap()
        lay3["yaxis"]["title"] = "°C"
        data3 = json.dumps([{"type":"scatter","x":ho_labels,"y":ho_vals,"mode":"lines",
            "line":{"color":"#10b981","width":2},"fill":"tozeroy",
            "fillcolor":"rgba(16,185,129,0.15)",
            "hovertemplate":"%{x}<br>%{y:.1f}°C<extra></extra>"}])
        components.html(plotly_html(data3, json.dumps(lay3), "ido"), height=235, scrolling=False)
        st.markdown(f'<div style="color:#94a3b8;font-size:9px;">Min: {edf["homerseklet"].min():.0f}°C — Max: {edf["homerseklet"].max():.0f}°C — Átlag: {edf["homerseklet"].mean():.0f}°C</div>', unsafe_allow_html=True)

    st.markdown('<div style="border-top:1px solid #1e3a5f;margin-top:12px;padding-top:8px;text-align:center;color:#64748b;font-size:9px;">XGBoost (FLAML AutoML) — MAE: 71.55 MWh — R²: 0.9768 — lag24h + lag168h mindig valós ENTSO-E adat</div>', unsafe_allow_html=True)

with tab2:
    st.markdown("### ⏱️ Részletes 24 órás előrejelzés")
    st.info("Következő lépés — fogyasztás + hőmérséklet kettős tengelyen, csúcs/völgy, napi költség, STL riasztás")

with tab3:
    st.markdown("### 🔌 Okos Mérő Asszisztens")
    st.info("Következő lépés — 24 órás DAM ár zónák, háztartási javaslatok, megtakarítás-kalkulátor")
