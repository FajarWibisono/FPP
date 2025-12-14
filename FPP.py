import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Fundamental Price Prediction", layout="wide")
st.title("📊 Fundamental Price Prediction")

ticker = st.text_input("Masukkan Kode Saham (contoh: BMRI.JK)", value="BMRI.JK").strip().upper()

def safe_get(obj, key, default=0.0):
    try:
        if isinstance(obj, (pd.Series, pd.DataFrame)):
            if key in obj.index:
                val = obj[key]
                return float(val) if pd.notna(val) and val is not None else default
            else:
                return default
        else:
            val = obj.get(key, default)
            return float(val) if pd.notna(val) and val is not None else default
    except:
        return default

def get_equity_from_balance_sheet(bs):
    candidates = [
        'Total Stockholder Equity',
        'Total Equity',
        'Stockholders Equity',
        'Total shareholders\' equity',
        'Total Shareholders Equity',
        'Ordinary Shares',
        'Total liabilities and equity'
    ]
    for key in candidates:
        if key in bs.index:
            series = bs.loc[key]
            if not series.empty:
                return series.iloc[0]
    return 0

def get_from_financials(fin, key_list, idx=0):
    for k in key_list:
        if k in fin.index:
            series = fin.loc[k]
            if not series.empty and idx < len(series):
                return series.iloc[idx]
    return 0

@st.cache_data(ttl=3600)
def fetch_fundamental_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        history = stock.history(period="5y")

        if financials.empty or balance_sheet.empty:
            return None

        shares_outstanding = safe_get(info, 'sharesOutstanding', 0)
        shares = shares_outstanding / 1_000_000 if shares_outstanding > 0 else 0
        last_price = safe_get(info, 'currentPrice', history['Close'].iloc[-1] if not history.empty else 0)

        revenue = get_from_financials(financials, ['Total Revenue', 'Revenue'])
        net_income = get_from_financials(financials, ['Net Income', 'Net Income Common Stockholders'])
        equity = get_equity_from_balance_sheet(balance_sheet)

        roe_annual = (net_income / equity * 100) if equity != 0 else 0

        roes = []
        for i in range(min(5, len(financials.columns))):
            ni = get_from_financials(financials, ['Net Income'], i)
            eq = get_equity_from_balance_sheet(balance_sheet.iloc[:, i:i+1])
            if eq != 0:
                roes.append(ni / eq * 100)
        roe_5y = np.mean(roes) if roes else 0

        eps_growth_5y = safe_get(info, 'earningsGrowth', 0) * 100
        if eps_growth_5y == 0:
            eps_growth_5y = safe_get(info, 'earningsQuarterlyGrowth', 0) * 100

        sps_growth_5y = safe_get(info, 'revenueGrowth', 0) * 100

        payout_ratio = safe_get(info, 'payoutRatio', 0) * 100
        if payout_ratio == 0:
            dividend_yield = safe_get(info, 'dividendYield', 0)
            eps = safe_get(info, 'trailingEps', 0)
            if dividend_yield > 0 and eps > 0:
                dps = dividend_yield * last_price
                payout_ratio = (dps / eps) * 100

        avg_pbv = safe_get(info, 'priceToBook', 0)
        avg_per = safe_get(info, 'trailingPE', 0)
        avg_psr = safe_get(info, 'priceToSalesTrailing12Months', 0)

        return {
            "Shares (in Million)": shares,
            "Last Price": last_price,
            "Pendapatan": revenue,
            "Profit": net_income,
            "Equity": equity,
            "ROE (Annual)": roe_annual,
            "ROE (in 5 Years)": roe_5y,
            "EPS Growth (in 5 Years)": eps_growth_5y,
            "SPS Growth (Annual)": sps_growth_5y,
            "SPS Growth (in 5 Years)": sps_growth_5y,
            "Dividend Payout Ratio (DPR%)": payout_ratio,
            "Average PBV": avg_pbv,
            "Average PER": avg_per,
            "Average PSR": avg_psr,
        }
    except Exception as e:
        st.warning(f"⚠️ Gagal mengambil data dari Yahoo Finance untuk `{ticker}`: {str(e)}")
        return None

data = fetch_fundamental_data(ticker)

st.subheader("🔍 Data Fundamental")
manual_mode = st.checkbox(" Gunakan input manual jika data otomatis tidak lengkap")

def input_with_default(label, default_val, format="%.2f"):
    key = label.replace(" ", "_").replace("(", "").replace(")", "")
    if manual_mode:
        return st.number_input(label, value=float(default_val), format=format, key=f"manual_{key}")
    else:
        st.text_input(label, value=f"{default_val:,.2f}" if isinstance(default_val, (int, float)) else str(default_val), disabled=True)
        return default_val

if data is None:
    st.error("Tidak ada data otomatis. Silakan isi secara manual.")
    manual_mode = True
    defaults = {k: 0.0 for k in [
        "Shares (in Million)", "Last Price", "Pendapatan", "Profit", "Equity",
        "ROE (Annual)", "ROE (in 5 Years)", "EPS Growth (in 5 Years)",
        "SPS Growth (Annual)", "SPS Growth (in 5 Years)", "Dividend Payout Ratio (DPR%)",
        "Average PBV", "Average PER", "Average PSR"
    ]}
else:
    defaults = data

col1, col2 = st.columns(2)

with col1:
    shares = input_with_default("Shares (in Million)", defaults["Shares (in Million)"])
    last_price = input_with_default("Last Price", defaults["Last Price"])
    pendapatan = input_with_default("Pendapatan", defaults["Pendapatan"])
    profit = input_with_default("Profit", defaults["Profit"])
    equity = input_with_default("Equity", defaults["Equity"])
    roe_annual = input_with_default("ROE (Annual) (%)", defaults["ROE (Annual)"])
    roe_5y = input_with_default("ROE (in 5 Years) (%)", defaults["ROE (in 5 Years)"])
    eps_growth_5y = input_with_default("EPS Growth (in 5 Years) (%)", defaults["EPS Growth (in 5 Years)"])

with col2:
    sps_growth_annual = input_with_default("SPS Growth (Annual) (%)", defaults["SPS Growth (Annual)"])
    sps_growth_5y = input_with_default("SPS Growth (in 5 Years) (%)", defaults["SPS Growth (in 5 Years)"])
    dpr = input_with_default("Dividend Payout Ratio (DPR%)", defaults["Dividend Payout Ratio (DPR%)"])
    avg_pbv = input_with_default("Average PBV", defaults["Average PBV"])
    avg_per = input_with_default("Average PER", defaults["Average PER"])
    avg_psr = input_with_default("Average PSR", defaults["Average PSR"])

# ==============================
# ✅ FUTURE VALUE PROJECTION (2025–2029)
# ==============================
st.markdown("---")
st.markdown(f"📈 Proyeksi Future Price (2025–2029) – <span style='color: green; font-weight: bold;'>{ticker}</span>", unsafe_allow_html=True)

# Current per-share metrics
shares_outstanding = shares * 1_000_000
book_value_per_share = equity / shares_outstanding if shares_outstanding > 0 else 0
eps_current = profit / shares_outstanding if shares_outstanding > 0 else 0
sps_current = pendapatan / shares_outstanding if shares_outstanding > 0 else 0

# Growth rates (as decimals)
roe = roe_5y / 100
eps_growth = eps_growth_5y / 100
sps_growth = sps_growth_5y / 100

# Projection years: 2025 to 2029 (5 years)
start_year = 2025
years = [start_year + i for i in range(5)]  # [2025, 2026, 2027, 2028, 2029]

# Project future values (compounded annually)
bv_fv = [book_value_per_share * ((1 + roe) ** (i + 1)) for i in range(5)]
eps_fv = [eps_current * ((1 + eps_growth) ** (i + 1)) for i in range(5)]
sps_fv = [sps_current * ((1 + sps_growth) ** (i + 1)) for i in range(5)]

# Future Price = Metric × Avg Multiple
price_pbv = [bv * avg_pbv for bv in bv_fv]
price_per = [eps * avg_per for eps in eps_fv]
price_psr = [sps * avg_psr for sps in sps_fv]

# Create DataFrame
future_df = pd.DataFrame({
    "Tahun": years,
    "BV Future Value": bv_fv,
    "Future Price (PBV)": price_pbv,
    "EPS Future Value": eps_fv,
    "Future Price (PER)": price_per,
    "SPS Future Value": sps_fv,
    "Future Price (PSR)": price_psr,
})

st.dataframe(future_df.round(2), width='stretch')

# ==============================
# 🎯 SUMMARY & POTENTIAL
# ==============================
st.markdown("---")
st.subheader("🎯 Kesimpulan & Potensi Investasi")

# Use 2029 (last year) future price average
price_2029_pbv = future_df.iloc[-1]["Future Price (PBV)"]
price_2029_per = future_df.iloc[-1]["Future Price (PER)"]
price_2029_psr = future_df.iloc[-1]["Future Price (PSR)"]

future_price_final = np.nanmean([price_2029_pbv, price_2029_per, price_2029_psr])

# Avoid division by zero
gain_loss = ((future_price_final / last_price) - 1) * 100 if last_price > 0 else 0
margin_of_safety = max(0, (future_price_final - last_price) / future_price_final * 100) if future_price_final > 0 else 0
cagr = (future_price_final / last_price) ** (1/5) - 1 if last_price > 0 else 0

summary_data = {
    "Last Price (Saat Ini)": last_price,
    "Future Price (2029)": future_price_final,
    "G&L Potential (%)": gain_loss,
    "Annual Return (%)": gain_loss / 5,
    "Margin of Safety (%)": margin_of_safety,
    "CAGR (%)": cagr * 100,
}

summary_df = pd.DataFrame(list(summary_data.items()), columns=["Metric", "Value"])
st.dataframe(summary_df.round(2), width='stretch')

# Optional: Show average future price across all years (like Excel's "Potensi Price")
st.markdown("### 💡 Catatan")
st.write("""
- Proyeksi menggunakan **ROE 5 tahun** untuk BV, **EPS Growth 5 tahun** untuk EPS, dan **SPS Growth 5 tahun** untuk SPS.
- Future Price dihitung dengan mengalikan proyeksi per-share metric dengan **rata-rata PBV, PER, PSR**.
- Jika data otomatis tidak tersedia, silakan centang **input manual**.
""")