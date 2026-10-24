"""
gui_adapter.py
Lapisan adaptor GUI (GUI Integration Adapter) untuk PeatFR Kalimantan.
Menghubungkan kalkulasi analitis dengan antarmuka pengguna grafis (PyQt6 Desktop / Streamlit / Web Dashboard).
Menyediakan hooks, ringkasan metrik instan, dan ekspor data terstruktur.
"""

import json
import os
from typing import Dict, Any, List, Optional, Callable
from .kalimantan_telemetry import KalimantanPeatlandTelemetry
from .forecasting_precision import PeatFRForecastingPrecision
from .pfvi_calibrated import CalibratedPFVI30Min

class PeatFRGuiAdapter:
    """
    Service Controller untuk GUI yang mengonsumsi hasil telemetri & peramalan.
    """

    def __init__(self, station_id: str = "pulang_pisau", interval_min: int = 30):
        self.station_id = station_id
        self.interval_min = interval_min
        self.telemetry_handler = KalimantanPeatlandTelemetry(station_id=station_id)
        self.forecasting_engine = PeatFRForecastingPrecision(interval_min=interval_min)
        self.pfvi_calc = CalibratedPFVI30Min()

    def run_full_pipeline(
        self,
        raw_records: Optional[List[Dict[str, Any]]] = None,
        forecast_horizon_hours: int = 24,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Menjalankan pipeline komputasi terpadu untuk konsumsi GUI.
        """
        def update_progress(pct: int, msg: str):
            if progress_callback:
                progress_callback(pct, msg)

        # 1. Pemuatan / Penyiapan Data
        update_progress(10, "Menyiapkan dataset telemetri lahan gambut...")
        if not raw_records:
            # Gunakan data telemetri acuan Kalimantan
            raw_records = self.telemetry_handler.generate_synthetic_30min_kalimantan(days=14)

        # 2. Perhitungan PFVI Terkalibrasi Siklus 30 Menit
        update_progress(35, "Menghitung Peat Fire Vulnerability Index (PFVI) terkalibrasi...")
        pfvi_results = self.pfvi_calc.simulate_telemetry_series(raw_records, interval_min=self.interval_min)

        # 3. Peramalan Presisi Multivariat (Bi-LSTM + GRU + ARIMA)
        steps_horizon = int(forecast_horizon_hours * 60 / self.interval_min)
        update_progress(70, f"Melatih & meramalkan {steps_horizon} langkah ({forecast_horizon_hours} jam ke depan)...")
        forecast_results = self.forecasting_engine.forecast_multivariate_ensemble(
            raw_records,
            horizon_steps=steps_horizon
        )

        # 4. Ringkasan Status & Indikator Dashboard GUI
        update_progress(90, "Menyusun metrik dan indikator dasbor GUI...")
        latest = pfvi_results[-1]
        latest_wtd = latest["wtd_m"]
        latest_pfvi = latest["pfvi"]
        risk_info = self.pfvi_calc.classify_fire_risk(latest_pfvi, latest_wtd)

        # Rekomendasi mitigasi lapangan untuk petugas/analis
        if risk_info["code"] == "CRITICAL":
            rekomendasi = "SIAGA PENUH: Aktifkan pompa air pembasahan, tutup pintu sekat kanal (canal blocking), dan kirim tim patroli darat."
        elif risk_info["code"] == "ADVISORY":
            rekomendasi = "WASPADA: Lakukan pemantauan tinggi muka air berkala, atur elevasi sekat kanal agar genangan terjaga."
        else:
            rekomendasi = "NORMAL: Kondisi hidrologis gambut dalam batas aman basah (> -0.4 m). Pertahankan kelembapan alami."

        dashboard_payload = {
            "metadata": {
                "station_id": self.station_id,
                "station_name": self.telemetry_handler.meta["name"],
                "coordinates": [self.telemetry_handler.meta["lat"], self.telemetry_handler.meta["lon"]],
                "interval_min": self.interval_min,
                "total_samples": len(raw_records),
                "generated_time": raw_records[-1]["timestamp"]
            },
            "current_status": {
                "wtd_m": latest_wtd,
                "soil_moisture_pct": latest["soil_moisture_pct"],
                "pfvi": latest_pfvi,
                "risk_status": risk_info["status"],
                "risk_code": risk_info["code"],
                "risk_color": risk_info["color"],
                "description": risk_info["description"],
                "action_recommendation": rekomendasi
            },
            "forecast_summary": {
                "horizon_hours": forecast_horizon_hours,
                "predicted_wtd_end_m": forecast_results["WTD_pred"][-1],
                "predicted_sm_end_pct": forecast_results["SM_pred"][-1],
                "wtd_trajectory": forecast_results["WTD_pred"],
                "wtd_ci95_upper": forecast_results["WTD_ci95_upper"],
                "wtd_ci95_lower": forecast_results["WTD_ci95_lower"],
                "sm_trajectory": forecast_results["SM_pred"],
                "models_used": forecast_results["models_evaluated"]
            },
            "historical_timeseries": pfvi_results[-96:]  # 48 jam terakhir untuk grafik GUI
        }

        update_progress(100, "Pipeline selesai. Data siap dikonsumsi GUI.")
        return dashboard_payload

    def export_for_gui(self, payload: Dict[str, Any], output_path: str):
        """Mengekspor payload ke file JSON terstruktur untuk dibaca modul UI."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
