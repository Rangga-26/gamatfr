"""
kalimantan_telemetry.py
Modul penanganan, standardisasi, dan pengumpulan dataset telemetri lahan gambut Kalimantan.
Mendukung siklus 30 menit (48 sampel/hari) dari sensor stasiun lapangan dan telemetri satelit/reanalisis.
"""

import os
import json
import math
import urllib.request
import datetime
from typing import Dict, Any, List, Optional, Tuple

class KalimantanPeatlandTelemetry:
    """
    Handler dataset telemetri lahan gambut Kalimantan.
    Mendukung ingest 30 menit (BRGM, AWS, Open-Meteo ERA5) dan pembersihan data.
    """
    
    # Koordinat stasiun acuan gambut Kalimantan
    STATIONS = {
        "sebangau_kalteng": {"lat": -2.32, "lon": 113.90, "name": "Sebangau National Park, Central Kalimantan"},
        "pulang_pisau": {"lat": -2.75, "lon": 114.25, "name": "Pulang Pisau Peatland Dome, Central Kalimantan"},
        "palangka_raya": {"lat": -2.21, "lon": 113.92, "name": "Palangka Raya Peat Observatory, Central Kalimantan"},
        "sambas_kalbar": {"lat": 1.36, "lon": 109.30, "name": "Sambas Coastal Peatland, West Kalimantan"}
    }

    def __init__(self, station_id: str = "pulang_pisau"):
        self.station_id = station_id
        self.meta = self.STATIONS.get(station_id, self.STATIONS["pulang_pisau"])

    @staticmethod
    def generate_synthetic_30min_kalimantan(
        days: int = 30,
        start_date: str = "2026-09-01 00:00:00",
        seed: int = 42
    ) -> List[Dict[str, Any]]:
        """
        Menghasilkan dataset telemetri siklus 30 menit realistis (48 data per hari)
        mengikuti karakteristik hidrologi gambut tropis Kalimantan (musim kemarau/transisi).
        """
        import random
        random.seed(seed)
        
        n_steps = days * 48  # 48 langkah per hari
        base_time = datetime.datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
        
        records = []
        wtd_curr = -0.32  # Kedalaman awal Muka Air Tanah (meter)
        sm_curr = 68.5    # Kelembapan tanah awal (%)
        
        # Pola hujan periodik tipikal Kalimantan (hujan badai singkat sore/malam)
        for i in range(n_steps):
            t = base_time + datetime.timedelta(minutes=30 * i)
            hour = t.hour + t.minute / 60.0
            day_idx = i // 48
            
            # Fluktuasi suhu diurnal (harian): puncak jam 13-14 siang (~33-35 C), terendah subuh (~23-24 C)
            solar_phase = math.sin((hour - 8.0) * math.pi / 12.0)
            temp_ambient = 28.5 + 5.5 * solar_phase + random.gauss(0, 0.4)
            temp_ambient = max(22.0, min(36.5, temp_ambient))
            
            # Pola curah hujan (kejadian hujan lokal sporadis)
            # Hari ke 5-18 kemarau kering, hari ke 1-4 dan 19-30 ada hujan sesekali
            is_dry_spell = (5 <= day_idx <= 18)
            rain_prob = 0.02 if is_dry_spell else 0.08
            
            if random.random() < rain_prob and (14 <= hour <= 20):
                rainfall = round(random.uniform(5.0, 32.0), 2)
            elif random.random() < 0.01:
                rainfall = round(random.uniform(0.5, 3.0), 2)
            else:
                rainfall = 0.0
                
            # Dinamika Muka Air Tanah (WTD):
            # Jika hujan: WTD naik (mendekati 0) karena infiltrasi
            # Jika kering: WTD turun akibat evapotranspirasi (0.5 - 1.5 cm/hari)
            if rainfall > 0:
                # Infiltrasi gambut: 10 mm hujan menaikkan WTD ~ 4-7 cm
                wtd_curr += (rainfall / 1000.0) * 4.5
                sm_curr += rainfall * 0.8
            else:
                # Penurunan WTD bertahap per 30 menit
                et_loss = (0.012 / 48.0) * (temp_ambient / 28.0)
                wtd_curr -= et_loss + random.gauss(0, 0.0003)
                sm_curr -= (0.4 / 48.0) * (temp_ambient / 28.0)
                
            # Batas fisik
            wtd_curr = max(-1.60, min(0.15, wtd_curr))
            sm_curr = max(24.0, min(85.0, sm_curr))
            
            # Sinusoidal encoding jam (fitur temporal untuk ML)
            hour_sin = math.sin(2.0 * math.pi * hour / 24.0)
            hour_cos = math.cos(2.0 * math.pi * hour / 24.0)
            
            records.append({
                "timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),
                "step_idx": i,
                "interval_min": 30,
                "station": "Pulang_Pisau_Well_01",
                "province": "Kalimantan Tengah",
                "water_table_depth_m": round(wtd_curr, 4),
                "soil_moisture_pct": round(sm_curr, 2),
                "rainfall_mm": rainfall,
                "air_temperature_c": round(temp_ambient, 2),
                "hour_sin": round(hour_sin, 4),
                "hour_cos": round(hour_cos, 4)
            })
            
        return records

    def fetch_open_meteo_kalimantan(
        self,
        start_date: str,
        end_date: str,
        hourly: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Mengambil dataset reanalisis historis/real-time Open-Meteo ERA5
        untuk titik koordinat stasiun gambut Kalimantan terpilih.
        """
        lat = self.meta["lat"]
        lon = self.meta["lon"]
        
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&"
            f"hourly=temperature_2m,relative_humidity_2m,precipitation,surface_pressure&"
            f"timezone=Asia%2FJakarta"
        )
        
        req = urllib.request.Request(url, headers={"User-Agent": "PeatFR-Telemetry/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                return data
        except Exception as e:
            print(f"[Warning] Open-Meteo fetch error: {e}")
            return None

    @staticmethod
    def save_to_csv(records: List[Dict[str, Any]], filepath: str):
        """Menyimpan list records ke CSV tanpa dependensi eksternal."""
        if not records:
            return
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        keys = list(records[0].keys())
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            for r in records:
                line = [str(r.get(k, "")) for k in keys]
                f.write(",".join(line) + "\n")
