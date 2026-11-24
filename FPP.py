"""Fundamental Price Prediction - versi perbaikan (khusus saham IDX).

Aplikasi ini ditujukan untuk saham IDX (akhiran .JK). Konversi mata uang tetap
wajib karena sebagian emiten IDX (mis. PGAS, MEDC, ENRG, GIAA) melaporkan
laporan keuangan dalam USD, bukan IDR.

Perubahan utama dibanding FPP.py asli:
1. Mata uang disamakan: laporan keuangan (financialCurrency) dikonversi ke
   mata uang harga saham (currency) memakai kurs live dan kurs historis per
   tahun buku. Ini menangani emiten IDX yang melaporkan dalam USD.
2. Nilai NaN pada laporan tidak lagi "meracuni" rata-rata ROE 5 tahun.
3. Tahun laporan keuangan dipasangkan berdasarkan tanggal kolom, bukan posisi.
4. Pertumbuhan EPS/SPS dihitung sebagai CAGR historis (bukan growth 1 tahun),
   dan dibatasi agar tidak menghasilkan proyeksi ekstrem.
5. Payout ratio memakai dividendRate / memperlakukan dividendYield sebagai persen
   (yfinance mengembalikan 5.88 = 5.88%, bukan 0.0588).
6. Kelipatan PBV/PER/PSR dihitung dari data historis (median), divalidasi,
   dan nilai 0/negatif tidak lagi ikut dirata-rata.
7. Tahun proyeksi tidak lagi hardcode 2025.
8. Pertumbuhan Book Value default memakai laba ditahan: ROE x (1 - DPR),
   dengan opsi ROE penuh untuk metode lama.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Fundamental Price Prediction", layout="wide")
st.title("📊 Fundamental Price Prediction")

# Mata uang kuotasi dalam satuan pecahan (pence) -> (mata uang mayor, pengali)
FRACTIONAL = {
    "GBp": ("GBP", 0.01),
    "GBX": ("GBP", 0.01),
    "ZAc": ("ZAR", 0.01),
    "ILA": ("ILS", 0.01),
}

MAX_GROWTH = 0.30        # batas atas/bawah pertumbuhan tahunan untuk proyeksi
MAX_MULTIPLE = 200.0     # kelipatan di atas ini dianggap data rusak
MIN_VALID_PRICE_RATIO = 0.1   # future price / harga sekarang, batas kewajaran bawah
MAX_VALID_PRICE_RATIO = 10.0  # batas kewajaran atas


# ---------------------------------------------------------------- utilitas ---
def to_number(value, default=np.nan):
    try:
        if value is None:
            return default
        value = float(value)
        return value if np.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def split_currency(code):
    """Kembalikan (mata uang mayor, pengali harga)."""
    if not code:
        return None, 1.0
    if code in FRACTIONAL:
        return FRACTIONAL[code]
    return code.upper(), 1.0


@st.cache_data(ttl=3600)
def get_fx_series(source, target):
    """Riwayat kurs: 1 `source` = ? `target`, per tanggal."""
    if source == target:
        return pd.Series([1.0], index=[pd.Timestamp.now().normalize()])
    for pair, invert in ((f"{source}{target}=X", False), (f"{target}{source}=X", True)):
        try:
            hist = yf.Ticker(pair).history(period="6y")["Close"].dropna()
            if hist.empty:
                continue
            index = hist.index
            if getattr(index, "tz", None) is not None:
                index = index.tz_localize(None)
            series = pd.Series(hist.to_numpy(), index=index).sort_index()
            return 1.0 / series if invert else series
        except Exception:
            continue
    return None


def fx_at(series, period, fallback):
    """Kurs pada/ sebelum tanggal tertentu."""
    if series is None or series.empty:
        return fallback
    subset = series[series.index <= pd.Timestamp(period)]
    return float(subset.iloc[-1]) if not subset.empty else fallback


def pick_value(frame, keys, column):
    """Ambil nilai pertama yang ada dan tidak NaN untuk satu kolom periode."""
    if frame is None or column not in frame.columns:
        return np.nan
    for key in keys:
        if key in frame.index:
            try:
                value = frame.loc[key, column]
            except (KeyError, TypeError):
                continue
            if isinstance(value, pd.Series):
                value = value.dropna()
                if value.empty:
                    continue
                value = value.iloc[0]
            value = to_number(value)
            if np.isfinite(value):
                return value
    return np.nan


def price_as_of(history, period):
    """Harga penutupan terakhir pada/ sebelum akhir periode fiskal."""
    if history is None or history.empty:
        return np.nan
    closes = history["Close"].dropna()
    if closes.empty:
        return np.nan
    index = closes.index
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)
    closes = pd.Series(closes.to_numpy(), index=index).sort_index()
    target = pd.Timestamp(period)
    subset = closes[closes.index <= target]
    return float(subset.iloc[-1]) if not subset.empty else np.nan


def cagr(first, last, years):
    if not (np.isfinite(first) and np.isfinite(last)) or first <= 0 or years <= 0:
        return np.nan
    return (last / first) ** (1.0 / years) - 1.0


def clip_growth(value, label, notes):
    value = to_number(value)
    if not np.isfinite(value):
        return np.nan
    clipped = float(np.clip(value, -MAX_GROWTH, MAX_GROWTH))
    if clipped != value:
        notes.append(
            f"{label} {value * 100:,.1f}%/tahun dibatasi menjadi "
            f"{clipped * 100:,.1f}%/tahun agar proyeksi tidak ekstrem."
        )
    return clipped


def median_multiple(values):
    values = [v for v in values if np.isfinite(v) and 0 < v <= MAX_MULTIPLE]
    return float(np.median(values)) if values else np.nan


def fmt(value, decimals=2):
    if value is None or not np.isfinite(to_number(value)):
        return "-"
    return f"{value:,.{decimals}f}"


# ------------------------------------------------------------ data historis ---
@st.cache_data(ttl=3600)
def fetch_fundamental_data(ticker):
    notes = []
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        # auto_adjust=False: harga tetap disesuaikan aksi korporasi (split),
        # tetapi tidak ikut disesuaikan dividen, agar PBV/PER historis valid.
        history = stock.history(period="6y", auto_adjust=False)

        if financials is None or financials.empty or balance_sheet is None or balance_sheet.empty:
            return None

        # Emiten IDX umumnya IDR/IDR, tetapi beberapa melaporkan USD.
        default_currency = "IDR" if ticker.upper().endswith(".JK") else "USD"
        quote_currency = info.get("currency") or default_currency
        financial_currency = info.get("financialCurrency") or default_currency
        quote_major, price_multiplier = split_currency(quote_currency)
        financial_major, _ = split_currency(financial_currency)

        fx_series = get_fx_series(financial_major, quote_major)
        if fx_series is None:
            st.warning(
                f"⚠️ Kurs {financial_major}→{quote_major} tidak tersedia. "
                "Data laporan tidak dapat disamakan dengan harga saham."
            )
            return None
        rate_now = float(fx_series.iloc[-1]) if not fx_series.empty else 1.0
        if financial_major != quote_major:
            notes.append(
                f"Laporan keuangan dalam {financial_major}, harga saham dalam "
                f"{quote_currency}. Nilai laporan dikonversi ke {quote_major} "
                f"(kurs terakhir 1 {financial_major} = {rate_now:,.2f} {quote_major})."
            )

        # --- jumlah lembar saham ---
        shares_raw = to_number(info.get("sharesOutstanding"))
        if not np.isfinite(shares_raw) or shares_raw <= 0:
            shares_raw = to_number(info.get("impliedSharesOutstanding"))
        if not np.isfinite(shares_raw) or shares_raw <= 0:
            for frame in (balance_sheet, financials):
                for key in ("Ordinary Shares Number", "Share Issued"):
                    if frame is not None and key in frame.index:
                        series = frame.loc[key].dropna()
                        if not series.empty:
                            shares_raw = to_number(series.iloc[0])
                            break
                if np.isfinite(shares_raw) and shares_raw > 0:
                    break
        if not np.isfinite(shares_raw) or shares_raw <= 0:
            st.warning("⚠️ Jumlah saham beredar tidak ditemukan. Gunakan input manual.")
            return None

        # --- harga terakhir ---
        last_close = to_number(history["Close"].iloc[-1]) if not history.empty else np.nan
        last_price = np.nan
        for key in ("currentPrice", "regularMarketPrice", "previousClose"):
            last_price = to_number(info.get(key))
            if np.isfinite(last_price) and last_price > 0:
                break
        if not np.isfinite(last_price) or last_price <= 0:
            last_price = last_close
        if not np.isfinite(last_price) or last_price <= 0:
            st.warning("⚠️ Harga saham terakhir tidak ditemukan. Gunakan input manual.")
            return None

        price_value = last_price * price_multiplier  # dalam mata uang mayor

        # --- data per periode fiskal (dipasangkan per tanggal) ---
        revenue_keys = ["Total Revenue", "Operating Revenue"]
        income_keys = ["Net Income Common Stockholders", "Net Income",
                       "Net Income Including Noncontrolling Interests"]
        equity_keys = ["Stockholders Equity", "Total Stockholder Equity",
                       "Common Stock Equity", "Total Equity Gross Minority Interest"]

        periods = sorted(set(financials.columns) & set(balance_sheet.columns), reverse=True)
        rows = []
        for period in periods[:6]:
            revenue_raw = pick_value(financials, revenue_keys, period)
            income_raw = pick_value(financials, income_keys, period)
            equity_raw = pick_value(balance_sheet, equity_keys, period)
            close = price_as_of(history, period)
            period_rate = fx_at(fx_series, period, rate_now)
            rows.append({
                "period": pd.Timestamp(period),
                "rate": period_rate,
                "revenue_raw": revenue_raw,
                "income_raw": income_raw,
                "equity_raw": equity_raw,
                "revenue": revenue_raw * period_rate if np.isfinite(revenue_raw) else np.nan,
                "income": income_raw * period_rate if np.isfinite(income_raw) else np.nan,
                "equity": equity_raw * period_rate if np.isfinite(equity_raw) else np.nan,
                "close": close * price_multiplier if np.isfinite(close) else np.nan,
            })

        if not rows:
            return None

        # --- ROE tahunan & rata-rata 5 tahun (NaN/ekuitas negatif dilewati) ---
        roes = []
        for row in rows[:5]:
            if row["equity"] > 0 and np.isfinite(row["income"]):
                roes.append(row["income"] / row["equity"] * 100)
        roe_5y = float(np.mean(roes)) if roes else np.nan
        roe_annual = roes[0] if roes else roe_5y
        if len(roes) < 3:
            fallback = to_number(info.get("returnOnEquity"))
            if np.isfinite(fallback):
                roe_5y = fallback * 100
                notes.append("ROE 5 tahun dihitung dari data terbatas; memakai ROE Yahoo.")

        # Nilai terkini memakai kurs terakhir (bukan kurs akhir tahun buku).
        latest_revenue_raw = next((r["revenue_raw"] for r in rows if np.isfinite(r["revenue_raw"])), np.nan)
        latest_income_raw = next((r["income_raw"] for r in rows if np.isfinite(r["income_raw"])), np.nan)
        latest_equity_raw = next((r["equity_raw"] for r in rows if np.isfinite(r["equity_raw"])), np.nan)
        latest_revenue = latest_revenue_raw * rate_now if np.isfinite(latest_revenue_raw) else np.nan
        latest_income = latest_income_raw * rate_now if np.isfinite(latest_income_raw) else np.nan
        latest_equity = latest_equity_raw * rate_now if np.isfinite(latest_equity_raw) else np.nan

        # --- CAGR historis (bukan growth 1 periode) ---
        def series_cagr(field):
            points = [(r["period"], r[field]) for r in rows if np.isfinite(r[field]) and r[field] > 0]
            if len(points) < 2:
                return np.nan
            newest_period, newest = points[0]
            oldest_period, oldest = points[-1]
            years = (newest_period - oldest_period).days / 365.25
            return cagr(oldest, newest, years)

        eps_growth = series_cagr("income")
        if not np.isfinite(eps_growth):
            eps_growth = to_number(info.get("earningsGrowth"))
        sps_growth = series_cagr("revenue")
        if not np.isfinite(sps_growth):
            sps_growth = to_number(info.get("revenueGrowth"))

        eps_growth = clip_growth(eps_growth, "Pertumbuhan laba", notes)
        sps_growth = clip_growth(sps_growth, "Pertumbuhan pendapatan", notes)

        # Label "Annual" harus benar-benar pertumbuhan 1 tahun, bukan CAGR 5 tahun.
        sps_growth_annual = to_number(info.get("revenueGrowth"))
        if not np.isfinite(sps_growth_annual):
            sps_growth_annual = sps_growth
            if np.isfinite(sps_growth):
                notes.append("Growth pendapatan 1 tahun tidak tersedia; kolom Annual memakai CAGR.")

        # --- payout ratio ---
        eps_current = latest_income / shares_raw if latest_income > 0 else np.nan
        payout_ratio = np.nan
        raw_payout = to_number(info.get("payoutRatio"))
        if np.isfinite(raw_payout) and 0 < raw_payout <= 5:
            payout_ratio = raw_payout * 100
        if not np.isfinite(payout_ratio):
            dividend_rate = to_number(info.get("dividendRate"))
            if np.isfinite(dividend_rate) and np.isfinite(eps_current) and eps_current > 0:
                # dividendRate mengikuti mata uang kuotasi (termasuk pence)
                payout_ratio = dividend_rate * price_multiplier / eps_current * 100
        if not np.isfinite(payout_ratio):
            dividend_yield = to_number(info.get("dividendYield"))
            if np.isfinite(dividend_yield) and dividend_yield > 0 and np.isfinite(eps_current) and eps_current > 0:
                # yfinance: dividendYield dalam persen (5.88 = 5.88%)
                dy = dividend_yield / 100 if dividend_yield > 1 else dividend_yield
                payout_ratio = dy * price_value / eps_current * 100
        if np.isfinite(payout_ratio) and not 0 <= payout_ratio <= 200:
            notes.append(f"Payout ratio {payout_ratio:,.1f}% tidak wajar; diabaikan.")
            payout_ratio = np.nan

        # --- kelipatan historis (median), per periode memakai kurs periode itu ---
        pbv_values, per_values, psr_values = [], [], []
        for row in rows[:5]:
            if not (np.isfinite(row["close"]) and row["close"] > 0):
                continue
            if row["equity"] > 0:
                pbv_values.append(row["close"] / (row["equity"] / shares_raw))
            if row["income"] > 0:
                per_values.append(row["close"] / (row["income"] / shares_raw))
            if row["revenue"] > 0:
                psr_values.append(row["close"] / (row["revenue"] / shares_raw))

        avg_pbv = median_multiple(pbv_values)
        avg_per = median_multiple(per_values)
        avg_psr = median_multiple(psr_values)

        # fallback: kelipatan saat ini, tetap dalam mata uang yang sama
        if not np.isfinite(avg_pbv) and latest_equity > 0:
            avg_pbv = median_multiple([price_value / (latest_equity / shares_raw)])
        if not np.isfinite(avg_per) and eps_current > 0:
            avg_per = median_multiple([price_value / eps_current])
        if not np.isfinite(avg_psr) and latest_revenue > 0:
            avg_psr = median_multiple([price_value / (latest_revenue / shares_raw)])

        current_pbv = price_value / (latest_equity / shares_raw) if latest_equity > 0 else np.nan
        current_per = price_value / eps_current if np.isfinite(eps_current) and eps_current > 0 else np.nan
        current_psr = (
            price_value / (latest_revenue / shares_raw)
            if np.isfinite(latest_revenue) and latest_revenue > 0
            else np.nan
        )

        if not any(np.isfinite(v) for v in (avg_pbv, avg_per, avg_psr)):
            notes.append("Tidak ada kelipatan PBV/PER/PSR yang valid.")

        return {
            "Quote Currency": quote_currency,
            "Valuation Currency": quote_major,
            "Sector": info.get("sector"),
            "Financial Currency": financial_currency,
            "FX Rate": rate_now,
            "Shares (in Million)": shares_raw / 1_000_000,
            "Last Price": price_value,
            "Pendapatan": latest_revenue,
            "Profit": latest_income,
            "Equity": latest_equity,
            "ROE (Annual)": roe_annual,
            "ROE (in 5 Years)": roe_5y,
            "EPS Growth (in 5 Years)": eps_growth * 100 if np.isfinite(eps_growth) else np.nan,
            "SPS Growth (Annual)": sps_growth_annual * 100 if np.isfinite(sps_growth_annual) else np.nan,
            "SPS Growth (in 5 Years)": sps_growth * 100 if np.isfinite(sps_growth) else np.nan,
            "Dividend Payout Ratio (DPR%)": payout_ratio,
            "Average PBV": avg_pbv,
            "Average PER": avg_per,
            "Average PSR": avg_psr,
            "Current PBV": current_pbv,
            "Current PER": current_per,
            "Current PSR": current_psr,
            "Notes": notes,
        }
    except Exception as exc:  # noqa: BLE001
        st.warning(f"⚠️ Gagal mengambil data dari Yahoo Finance untuk `{ticker}`: {exc}")
        return None


# --------------------------------------------------------------- proyeksi ---
def build_projection(data, horizon=5, start_year=None,
                     multiple_source="average", bv_model="retention"):
    """Hitung proyeksi. Semua nilai sudah dalam mata uang kuotasi (mayor).

    bv_model:
      "retention" -> BV tumbuh sebesar laba ditahan: ROE x (1 - DPR)
      "roe"       -> BV tumbuh sebesar ROE penuh (metode lama)
    """
    notes = []
    shares = to_number(data["Shares (in Million)"]) * 1_000_000
    price = to_number(data["Last Price"])

    if not np.isfinite(shares) or shares <= 0 or not np.isfinite(price) or price <= 0:
        return None, None, ["Data harga/jumlah saham tidak valid."]

    bvps = data["Equity"] / shares if np.isfinite(to_number(data["Equity"])) else np.nan
    eps = data["Profit"] / shares if np.isfinite(to_number(data["Profit"])) else np.nan
    sps = data["Pendapatan"] / shares if np.isfinite(to_number(data["Pendapatan"])) else np.nan

    roe = to_number(data["ROE (in 5 Years)"]) / 100
    eps_growth = to_number(data["EPS Growth (in 5 Years)"]) / 100
    sps_growth = to_number(data["SPS Growth (in 5 Years)"]) / 100

    # Batasi juga di sini supaya input manual tidak menghasilkan proyeksi ekstrem.
    eps_growth = clip_growth(eps_growth, "Pertumbuhan EPS", notes)
    sps_growth = clip_growth(sps_growth, "Pertumbuhan SPS", notes)

    # Pertumbuhan book value.
    payout = to_number(data.get("Dividend Payout Ratio (DPR%)"))
    if bv_model == "retention" and np.isfinite(roe):
        if np.isfinite(payout):
            payout_fraction = min(max(payout / 100, 0.0), 1.0)
            raw_growth_bv = roe * (1 - payout_fraction)
            if payout > 100:
                notes.append(
                    f"DPR {payout:,.1f}% > 100%; untuk proyeksi BV, DPR dibatasi 100% "
                    "(seluruh laba dibagikan, BV tidak tumbuh dari laba)."
                )
            notes.append(
                f"BV diproyeksikan tumbuh {raw_growth_bv * 100:,.2f}%/tahun "
                f"= ROE {roe * 100:,.2f}% x (1 - DPR {payout_fraction * 100:,.2f}%)."
            )
        else:
            raw_growth_bv = roe
            notes.append(
                "DPR tidak tersedia; BV diproyeksikan memakai ROE penuh "
                f"({roe * 100:,.2f}%/tahun)."
            )
    else:
        raw_growth_bv = roe
        if bv_model == "roe" and np.isfinite(roe):
            notes.append(f"BV diproyeksikan memakai ROE penuh ({roe * 100:,.2f}%/tahun).")

    growth_bv = float(np.clip(raw_growth_bv, -0.5, 0.5)) if np.isfinite(raw_growth_bv) else np.nan
    if np.isfinite(raw_growth_bv) and growth_bv != raw_growth_bv:
        notes.append(
            f"Pertumbuhan BV {raw_growth_bv * 100:,.2f}% dibatasi menjadi "
            f"{growth_bv * 100:,.2f}%/tahun agar proyeksi tidak ekstrem."
        )

    multiple_keys = ["Average PBV", "Average PER", "Average PSR"]
    if multiple_source == "current":
        multiple_keys = ["Current PBV", "Current PER", "Current PSR"]
    multiples = {
        "PBV": to_number(data.get(multiple_keys[0])),
        "PER": to_number(data.get(multiple_keys[1])),
        "PSR": to_number(data.get(multiple_keys[2])),
    }
    for key, value in list(multiples.items()):
        if not (np.isfinite(value) and 0 < value <= MAX_MULTIPLE):
            notes.append(f"Kelipatan {key} tidak valid ({fmt(value)}); metode {key} dilewati.")
            multiples[key] = np.nan

    start_year = start_year or datetime.now().year
    years = [start_year + i for i in range(horizon)]

    def grow(base, rate):
        if not (np.isfinite(base) and np.isfinite(rate)):
            return [np.nan] * horizon
        return [base * (1 + rate) ** (i + 1) for i in range(horizon)]

    bv_fv = grow(bvps, growth_bv)
    eps_fv = grow(eps, eps_growth)
    sps_fv = grow(sps, sps_growth)

    def price_series(values, multiple):
        if not np.isfinite(multiple):
            return [np.nan] * horizon
        return [v * multiple if np.isfinite(v) else np.nan for v in values]

    price_pbv = price_series(bv_fv, multiples["PBV"])
    price_per = price_series(eps_fv, multiples["PER"])
    price_psr = price_series(sps_fv, multiples["PSR"])

    frame = pd.DataFrame({
        "Tahun": years,
        "BV Future Value": bv_fv,
        "Future Price (PBV)": price_pbv,
        "EPS Future Value": eps_fv,
        "Future Price (PER)": price_per,
        "SPS Future Value": sps_fv,
        "Future Price (PSR)": price_psr,
    })

    # Konsistensi internal: EPS dan BV diproyeksikan dengan laju berbeda,
    # sehingga ROE implisit bisa bergeser jauh dari ROE saat ini.
    if np.isfinite(roe) and np.isfinite(eps_fv[-1]) and np.isfinite(bv_fv[-1]) and bv_fv[-1] > 0:
        implied_roe = eps_fv[-1] / bv_fv[-1] * 100
        current_roe = roe * 100
        if abs(implied_roe - current_roe) > max(5.0, 0.5 * abs(current_roe)):
            notes.append(
                f"ROE implisit {years[-1]} = {implied_roe:,.2f}% vs ROE saat ini "
                f"{current_roe:,.2f}%. Pertumbuhan EPS dan BV tidak konsisten; "
                "estimasi berbasis PER/PBV perlu diverifikasi manual."
            )

    if np.isfinite(multiples["PSR"]) and "financial" in str(data.get("Sector") or "").lower():
        notes.append(
            "Sektor keuangan: PSR (harga/pendapatan) kurang relevan untuk bank; "
            "utamakan PBV/PER."
        )

    final_candidates = [price_pbv[-1], price_per[-1], price_psr[-1]]
    valid = [v for v in final_candidates if np.isfinite(v) and v > 0]
    if not valid:
        return frame, None, notes + ["Semua metode proyeksi tidak menghasilkan harga valid."]
    if len(valid) < 3:
        notes.append(f"Hanya {len(valid)} dari 3 metode yang valid; rata-rata memakai metode tersebut.")
    if any(np.isfinite(v) and v <= 0 for v in final_candidates):
        notes.append("Metode dengan harga negatif/nol tidak diikutkan dalam rata-rata.")
    if len(valid) >= 2 and max(valid) / min(valid) > 2:
        notes.append(
            f"Estimasi antar metode berbeda jauh ({min(valid):,.0f} vs {max(valid):,.0f}); "
            "hasil sebaiknya tidak dipakai tanpa verifikasi manual."
        )

    future_price = float(np.mean(valid))
    gain_loss = (future_price / price - 1) * 100
    cagr_value = ((future_price / price) ** (1 / horizon) - 1) * 100
    margin_of_safety = (future_price - price) / future_price * 100

    if not (MIN_VALID_PRICE_RATIO <= future_price / price <= MAX_VALID_PRICE_RATIO):
        notes.append(
            f"⚠️ Future Price / harga sekarang = {future_price / price:,.2f}x, "
            "di luar rentang wajar. Periksa input, mata uang, dan data Yahoo."
        )

    summary = pd.DataFrame(
        [
            ("Last Price (Saat Ini)", price),
            (f"Future Price ({years[-1]})", future_price),
            ("G&L Potential (%)", gain_loss),
            ("Annual Return (%)", cagr_value),
            ("Margin of Safety (%)", margin_of_safety),
            ("CAGR (%)", cagr_value),
        ],
        columns=["Metric", "Value"],
    )
    return frame, summary, notes


# --------------------------------------------------------------------- UI ---
def input_with_default(label, default_val, manual_mode, format="%.2f"):
    key = label.replace(" ", "_").replace("(", "").replace(")", "")
    default_val = to_number(default_val, 0.0)
    if manual_mode:
        return st.number_input(label, value=float(default_val), format=format, key=f"manual_{key}")
    st.text_input(label, value=f"{default_val:,.2f}", disabled=True)
    return default_val


def show_dataframe(frame):
    try:
        st.dataframe(frame.round(2), width="stretch")
    except TypeError:  # Streamlit < 1.49
        st.dataframe(frame.round(2), use_container_width=True)


def main():
    raw_ticker = st.text_input(
        "Masukkan Kode Saham IDX (contoh: BMRI.JK atau BMRI)", value="BMRI.JK"
    ).strip().upper()

    ticker = raw_ticker
    if ":" in ticker:  # menerima format "IDX:BMRI"
        ticker = ticker.split(":")[-1]
    if ticker and "." not in ticker and ticker.isalpha():
        ticker = ticker + ".JK"
        st.caption(f"Kode disesuaikan menjadi **{ticker}** (saham IDX).")
    if ticker and not ticker.endswith(".JK"):
        st.warning(
            "⚠️ Aplikasi ini dioptimalkan untuk saham IDX (akhiran `.JK`). "
            "Kode di luar IDX tetap dijalankan, tetapi bukan cakupan resmi dan "
            "hasilnya bisa menyesatkan (mis. satuan pence/ADR)."
        )

    data = fetch_fundamental_data(ticker)

    st.subheader("🔍 Data Fundamental")
    manual_mode = st.checkbox("Gunakan input manual jika data otomatis tidak lengkap")

    if data is None:
        st.error("Tidak ada data otomatis. Silakan isi secara manual.")
        manual_mode = True
        defaults = {key: 0.0 for key in [
            "Shares (in Million)", "Last Price", "Pendapatan", "Profit", "Equity",
            "ROE (Annual)", "ROE (in 5 Years)", "EPS Growth (in 5 Years)",
            "SPS Growth (Annual)", "SPS Growth (in 5 Years)",
            "Dividend Payout Ratio (DPR%)", "Average PBV", "Average PER", "Average PSR",
        ]}
        currency_label = "-"
    else:
        defaults = data
        currency_label = data["Valuation Currency"]
        for note in data.get("Notes", []):
            st.info(note)

    if currency_label != "-":
        st.caption(
            f"Semua nilai moneter di bawah ini dalam **{currency_label}** "
            f"(laporan: {defaults.get('Financial Currency', currency_label)})."
        )
        if defaults.get("Quote Currency") not in (None, currency_label):
            st.caption(
                f"Catatan: harga kuotasi asli dalam {defaults['Quote Currency']}, "
                f"dinormalisasi ke {currency_label}."
            )

    col1, col2 = st.columns(2)
    with col1:
        shares = input_with_default("Shares (in Million)", defaults["Shares (in Million)"], manual_mode)
        last_price = input_with_default("Last Price", defaults["Last Price"], manual_mode)
        pendapatan = input_with_default("Pendapatan", defaults["Pendapatan"], manual_mode)
        profit = input_with_default("Profit", defaults["Profit"], manual_mode)
        equity = input_with_default("Equity", defaults["Equity"], manual_mode)
        roe_annual = input_with_default("ROE (Annual) (%)", defaults["ROE (Annual)"], manual_mode)
        roe_5y = input_with_default("ROE (in 5 Years) (%)", defaults["ROE (in 5 Years)"], manual_mode)
        eps_growth_5y = input_with_default("EPS Growth (in 5 Years) (%)", defaults["EPS Growth (in 5 Years)"], manual_mode)
    with col2:
        sps_growth_annual = input_with_default("SPS Growth (Annual) (%)", defaults["SPS Growth (Annual)"], manual_mode)
        sps_growth_5y = input_with_default("SPS Growth (in 5 Years) (%)", defaults["SPS Growth (in 5 Years)"], manual_mode)
        dpr = input_with_default("Dividend Payout Ratio (DPR%)", defaults["Dividend Payout Ratio (DPR%)"], manual_mode)
        avg_pbv = input_with_default("Average PBV", defaults["Average PBV"], manual_mode)
        avg_per = input_with_default("Average PER", defaults["Average PER"], manual_mode)
        avg_psr = input_with_default("Average PSR", defaults["Average PSR"], manual_mode)

    multiple_source = "average"
    bv_model = "retention"
    if currency_label != "-":
        choice = st.radio(
            "Dasar kelipatan valuasi",
            ["Rata-rata historis (median)", "Kelipatan saat ini"],
            horizontal=True,
        )
        if choice == "Kelipatan saat ini":
            multiple_source = "current"

        bv_choice = st.radio(
            "Model pertumbuhan Book Value",
            ["Laba ditahan: ROE × (1 − DPR)", "ROE penuh (metode lama)"],
            horizontal=True,
        )
        if bv_choice == "ROE penuh (metode lama)":
            bv_model = "roe"

    manual_data = {
        "Shares (in Million)": shares,
        "Last Price": last_price,
        "Pendapatan": pendapatan,
        "Profit": profit,
        "Equity": equity,
        "ROE (in 5 Years)": roe_5y,
        "EPS Growth (in 5 Years)": eps_growth_5y,
        "SPS Growth (in 5 Years)": sps_growth_5y,
        "Dividend Payout Ratio (DPR%)": dpr,
        "Average PBV": avg_pbv,
        "Average PER": avg_per,
        "Average PSR": avg_psr,
        "Current PBV": to_number(defaults.get("Current PBV"), avg_pbv) if not manual_mode else avg_pbv,
        "Current PER": to_number(defaults.get("Current PER"), avg_per) if not manual_mode else avg_per,
        "Current PSR": to_number(defaults.get("Current PSR"), avg_psr) if not manual_mode else avg_psr,
        "Sector": defaults.get("Sector"),
    }

    st.markdown("---")
    st.markdown(
        f"📈 Proyeksi Future Price 5 Tahun – "
        f"<span style='color: green; font-weight: bold;'>{ticker}</span>",
        unsafe_allow_html=True,
    )

    frame, summary, notes = build_projection(
        manual_data, multiple_source=multiple_source, bv_model=bv_model
    )
    if frame is None:
        st.error("Tidak dapat menghitung proyeksi: " + " ".join(notes))
        return

    show_dataframe(frame)

    st.markdown("---")
    st.subheader("🎯 Kesimpulan & Potensi Investasi")
    show_dataframe(summary)

    for note in notes:
        st.warning(note)

    st.caption(
        "Estimasi ini adalah skenario berbasis asumsi dan data Yahoo Finance, "
        "bukan rekomendasi investasi. Selalu verifikasi laporan keuangan emiten."
    )

    with st.expander("💡 Metodologi & asumsi"):
        st.write(
            """
- **Mata uang:** laporan keuangan dikonversi ke mata uang harga saham (kurs live dan kurs historis per tahun buku), sehingga PBV/PER/PSR konsisten. Ini penting untuk emiten IDX yang melaporkan dalam USD.
- **Book value:** tumbuh dari laba ditahan, `ROE × (1 − DPR)`; opsi ROE penuh tersedia (keduanya dibatasi ±50%/tahun).
- **EPS & SPS:** memakai CAGR historis dari laporan (bukan growth 1 periode), dibatasi ±30%/tahun.
- **Kelipatan:** median PBV/PER/PSR historis 5 tahun, atau kelipatan saat ini; nilai 0/negatif/tidak wajar tidak diikutkan.
- **Future Price:** rata-rata estimasi PBV, PER, dan PSR yang valid.
- **Keterbatasan:** DPR dan jumlah saham dianggap konstan; rights issue/buyback dapat mendistorsi CAGR; PSR kurang relevan untuk sektor keuangan; data Yahoo bisa keliru.
- Jika data otomatis tidak tersedia, centang **input manual**.
"""
        )


if __name__ == "__main__":
    main()
