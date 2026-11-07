"""Shared geographic reporting regions (first matching box wins)."""

# name, lon_min, lat_min, lon_max, lat_max (first match wins)
REGIONS = [
    ("Persian Gulf", 47.0, 23.0, 57.5, 30.6),
    ("Caspian Sea", 46.0, 36.5, 55.0, 47.5),
    ("Red Sea & Suez", 32.0, 12.0, 44.0, 31.0),
    ("Gulf of Mexico", -98.5, 17.0, -80.0, 31.0),
    ("North Sea", -4.5, 50.5, 10.0, 62.5),
    ("Norwegian & Barents Sea", 0.0, 62.5, 45.0, 75.0),
    ("Gulf of Guinea", -4.0, -7.0, 14.0, 7.5),
    ("Angola & Congo", 8.0, -18.0, 14.5, -7.0),
    ("Brazil", -50.0, -30.0, -34.0, 0.0),
    ("Trinidad & Venezuela", -72.0, 8.0, -58.0, 13.5),
    ("Mediterranean", -6.0, 30.0, 36.0, 46.0),
    ("Bohai & Yellow Sea", 117.0, 33.0, 126.0, 41.5),
    ("South China Sea", 105.0, 5.0, 122.0, 23.0),
    ("Gulf of Thailand", 99.0, 5.0, 105.0, 14.0),
    ("Malaysia & Indonesia", 95.0, -9.0, 125.0, 7.5),
    ("India & Bay of Bengal", 66.0, 5.0, 95.0, 25.0),
    ("Australia & Timor Sea", 110.0, -40.0, 155.0, -8.0),
    ("Alaska & Cook Inlet", -170.0, 55.0, -140.0, 72.0),
    ("Sakhalin", 140.0, 45.0, 146.0, 55.0),
]


def region_of(lat: float, lon: float) -> str:
    for name, x0, y0, x1, y1 in REGIONS:
        if x0 <= lon <= x1 and y0 <= lat <= y1:
            return name
    return "Other"
