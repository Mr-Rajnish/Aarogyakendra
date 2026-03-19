import requests
import os
import math
from typing import List, Dict, Tuple, Optional


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
        print("❌ Missing GEOAPIFY_API_KEY")
        return []

    try:
        url = (
            f"https://api.geoapify.com/v2/places?"
            f"categories=healthcare.hospital,healthcare.clinic"
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
            else :
                 dist=None 
            hospitals.append({
                    "name": props.get("name", "Hospital"),
                    "address": props.get("formatted", "Address not available"),
                    "distance_km": round(dist, 2)
            })

        hospitals = sorted(hospitals, key=lambda x: x["distance_km"])

        print("✅ Hospitals found:", len(hospitals))
        return hospitals[:limit]

    except Exception as e:
        print("❌ Geoapify error:", e)
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
    import os
    import requests

    # ✅ fallback location
    if lat is None or lon is None:
        lat, lon = 26.9124, 75.7873  # Jaipur fallback

    print("📍 Location:", lat, lon)

    hospitals = []

    # =========================
    # 🔥 1. GEOAPIFY API
    # =========================
    try:
        API_KEY = os.getenv("GEOAPIFY_API_KEY")

        if API_KEY:
            url = (
                f"https://api.geoapify.com/v2/places?"
                f"categories=healthcare.hospital"
                f"&filter=circle:{lon},{lat},{radius_km * 1000}"
                f"&limit={limit}"
                f"&apiKey={API_KEY}"
            )

            res = requests.get(url, timeout=10)
            data = res.json()

            for item in data.get("features", []):
                props = item.get("properties", {})

                hospitals.append({
                    "name": props.get("name", "Hospital"),
                    "address": props.get("formatted", "Address not available"),
                    "distance_km": round(props.get("distance", 0) / 1000, 2)
                })

            if hospitals:
                print("✅ Geoapify success")
                return hospitals[:limit], "Geoapify"

    except Exception as e:
        print("Geoapify failed:", e)

    # =========================
    # 🔥 2. OVERPASS API (backup)
    # =========================
    try:
        query = f"""
        [out:json];
        node["amenity"="hospital"](around:{radius_km*1000},{lat},{lon});
        out;
        """

        res = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": query},
            timeout=20
        )

        data = res.json()

        for el in data.get("elements", []):
            name = el.get("tags", {}).get("name", "Hospital")

            hospitals.append({
                "name": name,
                "address": "Nearby hospital",
                "distance_km": 0.0
            })

        if hospitals:
            print("✅ Overpass success")
            return hospitals[:limit], "Overpass"

    except Exception as e:
        print("Overpass failed:", e)

    # =========================
    # 🔥 3. FINAL FALLBACK (STATIC)
    # =========================
    print("⚠️ Using fallback hospitals")

    fallback = [
        {"name": "AIIMS Hospital", "address": "New Delhi", "distance_km": 5.0},
        {"name": "Fortis Hospital", "address": "Jaipur", "distance_km": 8.0},
        {"name": "Apollo Hospital", "address": "Delhi", "distance_km": 10.0},
        {"name": "Manipal Hospital", "address": "Jaipur", "distance_km": 12.0},
        {"name": "Sawai Man Singh Hospital", "address": "Jaipur", "distance_km": 3.0},
    ]

    return fallback[:limit], "fallback"