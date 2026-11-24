# Fundamental Price Prediction (IDX)

Aplikasi Streamlit untuk memperkirakan harga saham **IDX** 5 tahun ke depan
berdasarkan data fundamental (PBV, PER, PSR) dari Yahoo Finance.

> Estimasi ini adalah skenario berbasis asumsi, bukan rekomendasi investasi.

## Menjalankan

```bash
pip install -r requirements.txt
streamlit run FPP.py
```

Masukkan kode saham dengan atau tanpa akhiran `.JK` (mis. `BMRI` atau `BMRI.JK`).
Format `IDX:BMRI` juga diterima.

## Metodologi

1. **Mata uang disamakan.** Laporan keuangan (`financialCurrency`) dikonversi ke
   mata uang harga saham (`currency`) memakai kurs live dan kurs historis per
   tahun buku. Ini penting karena sebagian emiten IDX (mis. PGAS, MEDC, ENRG,
   GIAA, RAJA) melaporkan dalam USD.
2. **Book value** tumbuh dari laba ditahan: `ROE × (1 − DPR)`, dengan opsi
   ROE penuh. Pertumbuhan dibatasi ±50%/tahun.
3. **EPS dan SPS** diproyeksikan memakai CAGR historis dari laporan (bukan
   growth satu periode), dibatasi ±30%/tahun.
4. **Kelipatan** PBV/PER/PSR memakai median historis 5 tahun atau kelipatan
   saat ini. Nilai 0, negatif, atau tidak wajar tidak diikutkan.
5. **Future Price** = rata-rata estimasi PBV, PER, dan PSR yang valid.
   Aplikasi memberi peringatan bila estimasi antar metode berbeda jauh atau
   di luar rentang wajar.

## Keterbatasan

- DPR dan jumlah saham dianggap konstan 5 tahun.
- Rights issue/buyback dapat mendistorsi CAGR dan kelipatan historis.
- PSR kurang relevan untuk sektor keuangan (aplikasi memberi peringatan).
- Data Yahoo Finance dapat keliru atau tertunda; verifikasi selalu diperlukan.
- Aplikasi dioptimalkan untuk IDX. Kode non-`.JK` tetap dijalankan tetapi diberi
  peringatan dan di luar cakupan resmi.
