import csv
import math
import os
from typing import Dict, List, Optional, Tuple

import requests


# =========================
# 📍 LOCATION
# =========================
def get_ip_location():
    try:
        r = requests.get("http://ip-api.com/json/")
        data = r.json()

        return {
            "city": data.get("city", "Jaipur"),
            "lat": data.get("lat"),
            "lon": data.get("lon"),
        }

    except:
        return {
            "city": "Jaipur",
            "lat": 26.9124,
            "lon": 75.7873
        }


# =========================
# 📏 DISTANCE
# =========================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2 +
        math.cos(math.radians(lat1)) *
        math.cos(math.radians(lat2)) *
        math.sin(dlon / 2) ** 2
    )

    return 2 * R * math.asin(math.sqrt(a))


# =========================
# 🌍 FETCH HOSPITALS
# =========================
def fetch_hospitals(lat, lon, radius_km=100, limit=5):
    try:
        query = f"""
        [out:json];
        node["amenity"="hospital"](around:{radius_km*1000},{lat},{lon});
        out;
        """

        res = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": query},
            timeout=60
        )

        data = res.json()

        hospitals = []

        for el in data.get("elements", []):
            tags = el.get("tags", {})

            name = tags.get("name", "Hospital")

            # 🔥 FIX ADDRESS
            address = tags.get("addr:full")

            if not address:
                street = tags.get("addr:street", "")
                city = tags.get("addr:city", "")
                address = f"{street}, {city}" if street else "Address not available"

            h_lat = el.get("lat")
            h_lon = el.get("lon")

            if h_lat and h_lon:
                dist = haversine(lat, lon, h_lat, h_lon)

                # 🔥 FIX DISTANCE FORMAT
                distance = f"{int(dist*1000)} meters" if dist < 1 else f"{round(dist,2)} km"

                hospitals.append({
                    "name": name,
                    "address": address,
                    "distance": distance
                })

        hospitals = sorted(hospitals, key=lambda x: x["distance"])

        print(f"✅ Found {len(hospitals)} hospitals")

        return hospitals[:limit]

    except Exception as e:
        print("API Error:", e)
        return []


# =========================
# 🏥 MAIN FUNCTION
# =========================
def find_hospitals(
    disease=None,
    location_city=None,
    lat=None,
    lon=None,
    limit=5
) -> Tuple[List[Dict], str]:

    if lat is None or lon is None:
        loc = get_ip_location()
        lat = loc["lat"]
        lon = loc["lon"]
        location_city = loc["city"]

    print(f"📍 Location: {location_city}")

    hospitals = fetch_hospitals(lat, lon, limit=limit)

    return hospitals, "API"