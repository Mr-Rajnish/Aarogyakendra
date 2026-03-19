import requests
import os
import math
from typing import List, Dict, Tuple


# =========================
# 📍 GET LOCATION
# =========================
def get_ip_location():
    try:
        res = requests.get("http://ip-api.com/json/", timeout=10)
        data = res.json()

        return {
            "city": data.get("city", "Jaipur"),
            "lat": data.get("lat"),
            "lon": data.get("lon"),
        }

    except:
        return {"city": "Jaipur", "lat": 26.9124, "lon": 75.7873}


# =========================
# 📏 DISTANCE
# =========================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )

    return 2 * R * math.asin(math.sqrt(a))


# =========================
# 🌍 GEOAPIFY API
# =========================
def fetch_hospitals_geoapify(lat, lon, radius_km=100, limit=5):
    API_KEY = os.getenv("GEOAPIFY_API_KEY")

    if not API_KEY:
        print("❌ No Geoapify API key")
        return []

    try:
        url = (
            f"https://api.geoapify.com/v2/places?"
            f"categories=healthcare.hospital"
            f"&filter=circle:{lon},{lat},{radius_km * 1000}"
            f"&limit={limit}"
            f"&apiKey={API_KEY}"
        )

        res = requests.get(url, timeout=15)
        data = res.json()

        hospitals = []

        for item in data.get("features", []):
            props = item.get("properties", {})

            h_lat = props.get("lat")
            h_lon = props.get("lon")

            if h_lat and h_lon:
                dist = haversine(lat, lon, h_lat, h_lon)

                hospitals.append({
                    "name": props.get("name", "Hospital"),
                    "address": props.get("formatted", "Address not available"),
                    "distance_km": round(dist, 2)
                })

        return sorted(hospitals, key=lambda x: x["distance_km"])

    except Exception as e:
        print("Geoapify error:", e)
        return []


# =========================
# 🏥 MAIN FUNCTION
# =========================
def find_hospitals(
    disease,
    location_city,
    lat=None,
    lon=None,
    limit=5,
    radius_km=100,
    use_dataset_fallback=True,
):

    if lat is None or lon is None:
        loc = get_ip_location()
        lat = loc["lat"]
        lon = loc["lon"]
        location_city = loc["city"]

    print("📍 Location:", location_city, lat, lon)

    hospitals = fetch_hospitals_geoapify(lat, lon, radius_km, limit)

    if hospitals:
        print("✅ Geoapify hospitals:", len(hospitals))
        return hospitals[:limit], "Geoapify"

    print("⚠️ API failed")

    return [], "no-data"