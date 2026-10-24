"""
pfvi_calibrated.py
Modul perhitungan Peat Fire Vulnerability Index (PFVI) terkalibrasi untuk siklus 30 menit.
Memperbaiki skala waktu dt, intersepsi kanopi sub-harian, dan klasifikasi bahaya kebakaran gambut (KLHK/BRGM).
"""

import math
from typing import Dict, Any, List, Tuple

class CalibratedPFVI30Min:
    """
    Kalkulator PFVI terspesialisasi untuk siklus sub-harian (30 menit).
    """

    # Parameter default hasil optimasi Nelder-Mead pada tanah gambut tropis (Mahdiyasa et al., 2025)
    DEFAULT_PARAMS = {
        "a_h": 285.4,
        "b_h": 0.88,
        "n": 1.45,
        "alpha": 32.1,
        "r0": 3000.0,
        "is_peat": True
    }

    def __init__(self, params: Dict[str, Any] = None):
        self.params = self.DEFAULT_PARAMS.copy()
        if params:
            self.params.update(params)

    @staticmethod
    def calculate_dt(interval_min: int = 30) -> float:
        """Menghitung skala waktu pecahan hari (misal 30 menit = 1/48 hari = 0.020833)."""
        return interval_min / 1440.0

    def calculate_subdaily_df(self, pfvi_prev: float, temp_c: float, interval_min: int = 30) -> float:
        """
        Evapotranspirasi faktor kekeringan DF_t untuk interval sub-harian.
        """
        dt = self.calculate_dt(interval_min)
        r0 = self.params["r0"]
        
        # Suku evapotranspirasi potensial harian
        et_daily = 0.4982 * math.exp(0.0905 * temp_c + 1.6096) - 4.268
        et_daily = max(0.0, et_daily)

        numerator = (300.0 - pfvi_prev) * et_daily * dt * 1e-3
        denominator = 1.0 + 10.88 * math.exp(-0.00173582677165354 * r0)
        return float(numerator / denominator)

    @staticmethod
    def calculate_subdaily_rf(rf_current: float, rolling_rain_24h: float) -> float:
        """
        Faktor curah hujan efektif RF_t dengan pelacakan intersepsi kanopi gambut harian.
        """
        if rf_current <= 0:
            return 0.0
        
        # Jika akumulasi hujan harian belum melewati batas intersepsi 5.1 mm/hari
        if rolling_rain_24h < 5.1:
            excess = max(0.0, rolling_rain_24h - 5.1)
            return min(float(rf_current), excess)
        else:
            return float(rf_current)

    def calculate_wtf(self, wt_depth_cm: float) -> Tuple[float, float]:
        """
        Faktor Muka Air Tanah (Water Table Factor) van Genuchten.
        wt_depth_cm: kedalaman muka air tanah positif dalam cm.
        """
        a_h = self.params["a_h"]
        b_h = self.params["b_h"]
        n = self.params["n"]
        alpha = self.params["alpha"]

        if n <= 1.0 or alpha <= 0:
            return 0.0, 0.0

        m = 1.0 - (1.0 / n)
        v = max(0.0, wt_depth_cm)
        theta = (1.0 + (v / alpha) ** n) ** (-m)
        wtf = a_h - b_h * ((1.0 - theta) * 300.0)
        return float(wtf), float(theta)

    def calculate_drought_index_obs(self, sm: float) -> float:
        """
        Menghitung Indeks Kekeringan Teramati (DI_obs) dari kadar air tanah volumetrik.
        """
        if self.params["is_peat"]:
            fc = 32.0   # Field capacity gambut (%)
            sat = 75.0  # Porositas jenuh gambut (%)
        else:
            fc = 20.0
            sat = 55.0

        di = 300.0 * (1.0 - ((sm - fc) / (sat - fc)))
        return max(0.0, min(300.0, di))

    @staticmethod
    def classify_fire_risk(pfvi_value: float, wtd_m: float) -> Dict[str, str]:
        """
        Klasifikasi status kerentanan kebakaran gambut berdasarkan regulasi BRG/KLHK.
        Standar: WTD > -0.4m adalah batas aman hidrologis gambut.
        """
        if pfvi_value >= 200.0 or wtd_m <= -0.80:
            status = "BAHAYA"
            code = "CRITICAL"
            color = "#EF4444"
            desc = "Lahan gambut sangat kering, muka air tanah di bawah ambang kritis, potensi kebakaran tinggi."
        elif pfvi_value >= 100.0 or wtd_m <= -0.40:
            status = "WASPADA"
            code = "ADVISORY"
            color = "#F59E0B"
            desc = "Muka air tanah melewati batas -0.4 m, lahan mulai rentan terbakar, siagakan sekat kanal."
        else:
            status = "AMAN"
            code = "SAFE"
            color = "#10B981"
            desc = "Lahan gambut tergenang/basah memadai, risiko kebakaran minimal."

        return {
            "status": status,
            "code": code,
            "color": color,
            "description": desc
        }

    def simulate_telemetry_series(self, dataset: List[Dict[str, Any]], interval_min: int = 30) -> List[Dict[str, Any]]:
        """
        Menjalankan kalkulasi PFVI berurutan dari rentetan telemetri siklus 30 menit.
        """
        results = []
        pfvi_curr = 50.0  # Nilai awal netral
        rolling_rain_window = []
        window_size = int(1440 / interval_min)  # 48 langkah untuk 24 jam

        for row in dataset:
            wtd_m = row["water_table_depth_m"]
            sm = row["soil_moisture_pct"]
            rf = row["rainfall_mm"]
            temp = row["air_temperature_c"]

            # Konversi WTD meter ke kedalaman cm positif
            wtd_depth_cm = abs(min(0.0, wtd_m)) * 100.0

            # Lacak akumulasi hujan 24 jam
            rolling_rain_window.append(rf)
            if len(rolling_rain_window) > window_size:
                rolling_rain_window.pop(0)
            rolling_24h = sum(rolling_rain_window)

            # Hitung faktor-faktor pembentuk
            df = self.calculate_subdaily_df(pfvi_curr, temp, interval_min)
            rf_factor = self.calculate_subdaily_rf(rf, rolling_24h)
            wtf, theta = self.calculate_wtf(wtd_depth_cm)
            di_obs = self.calculate_drought_index_obs(sm)

            # Persamaan neraca PFVI t = PFVI t-1 + DF_t - RF_t + WTF_t
            pfvi_next = pfvi_curr + df - (rf_factor * 0.8) + (wtf * 0.005)
            pfvi_curr = max(0.0, min(300.0, pfvi_next))

            risk = self.classify_fire_risk(pfvi_curr, wtd_m)

            results.append({
                "timestamp": row["timestamp"],
                "wtd_m": wtd_m,
                "soil_moisture_pct": sm,
                "rainfall_mm": rf,
                "air_temp_c": temp,
                "pfvi": round(pfvi_curr, 2),
                "di_obs": round(di_obs, 2),
                "risk_status": risk["status"],
                "risk_code": risk["code"],
                "risk_color": risk["color"]
            })

        return results
