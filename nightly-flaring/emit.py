# %%
"""Public NASA CMR overpasses and optional Earth Engine review chips (D59)."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

HERE = Path(__file__).resolve().parent
CACHE = HERE / "outputs" / "emit_cache"
CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
PRODUCTS = ("EMITL2BCH4ENH", "EMITL1BRAD")
COLLECTION = "NASA/EMIT/L1B/RAD"
CHIP_RECIPE = "hitl-v7"
NO_DATA_MESSAGE = (
    "No EMIT data at the source in this scene (it falls between image strips)"
)
CAPTION = (
    "Quick-look relative CH4 absorption index, not a calibrated methane "
    "concentration or emission rate. Surface, cloud and water effects can "
    "produce signal; no plume has been confirmed by this screen. "
    "Grey marks missing EMIT data; dark blue (#00131f) marks index values <=1 sigma."
)


def _listing_path(product, lon, lat, start, end, cache_dir):
    query = [product, round(lon, 6), round(lat, 6), start, end]
    digest = hashlib.sha256(json.dumps(query).encode()).hexdigest()
    return cache_dir / "cmr" / (digest + ".json"), query


def calendar_listing(product, lon, lat, months, cache_dir, today, session):
    """Reuse final months; refresh all other months with one product query.

    Successful combined results also populate stable monthly snapshots. On
    failure, monthly snapshots remain reviewable with incomplete status.
    """
    entries, mutable, snapshots = [], [], []
    for start, end in months:
        path, query = _listing_path(product, lon, lat, start, end, cache_dir)
        try:
            cached = json.loads(path.read_text())
            if cached["query"] != query:
                raise ValueError("Different query")
            final = date.fromisoformat(cached.get("fetched_on", "1900-01-01"))
            if final >= date.fromisoformat(end) + timedelta(days=30):
                entries.extend(cached["entries"])
                continue
            snapshots.extend(cached["entries"])
        except (OSError, ValueError, KeyError):
            pass
        mutable.append((start, end))
    if not mutable:
        return entries
    try:
        fresh = cmr_granules(
            product,
            lon,
            lat,
            mutable[0][0],
            mutable[-1][1],
            cache_dir=cache_dir,
            today=today,
            session=session,
        )
    except CachedListing as cached:
        raise CachedListing(entries + snapshots + cached.entries) from None
    except Exception:
        raise CachedListing(entries + snapshots) from None
    for start, end in mutable:
        first, last = map(
            datetime.fromisoformat,
            utc_window(start, end, lon).replace("Z", "+00:00").split(","),
        )
        selected = [
            e
            for e in fresh
            if e.get("time_start")
            and first
            <= datetime.fromisoformat(e["time_start"].replace("Z", "+00:00"))
            < last
        ]
        path, query = _listing_path(product, lon, lat, start, end, cache_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        temp.write_text(
            json.dumps(
                dict(query=query, entries=selected, fetched_on=today.isoformat())
            )
        )
        temp.replace(path)
    return entries + fresh


def pending_message(collection_latest_utc: str) -> str:
    """Describe the measured collection horizon without assuming ingestion."""
    latest = datetime.fromisoformat(collection_latest_utc.replace("Z", "+00:00"))
    return (
        "not in Earth Engine yet (its EMIT collection ends "
        f"{latest.day} {latest:%b %Y})"
    )


def cmr_window(candidate: dict, as_of: str) -> tuple[str, str]:
    """Fourteen days before/after a closed spell; ongoing windows end as-of."""
    start = date.fromisoformat(candidate["start"]) - timedelta(days=14)
    end = date.fromisoformat(candidate["end"]) + timedelta(days=14)
    return start.isoformat(), min(end, date.fromisoformat(as_of)).isoformat()


def utc_window(start: str, end: str, lon: float) -> str:
    """Inclusive local evening nights are noon-to-noon, converted to UTC."""
    first = datetime.fromisoformat(start).replace(tzinfo=timezone.utc) + timedelta(
        hours=12 - lon / 15
    )
    last = datetime.fromisoformat(end).replace(tzinfo=timezone.utc) + timedelta(
        days=1, hours=12 - lon / 15
    )
    return (
        first.isoformat().replace("+00:00", "Z")
        + ","
        + last.isoformat().replace("+00:00", "Z")
    )


def scene_phase(t: datetime, candidate: dict) -> str:
    """Bracket transitions by measured VIIRS passes; never date a stop exactly."""
    times = candidate.get("viirs_times", {})

    def instant(key):
        value = times.get(key)
        if not value or not isinstance(value, str):
            return None
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(
            tzinfo=timezone.utc
        )

    last_lit, first_dark = instant("last_lit"), instant("first_dark")
    last_dark, recovery = instant("last_dark"), instant("recovery")
    if last_lit and t <= last_lit:
        return "before"
    if recovery and t >= recovery:
        return "after"
    if first_dark and last_dark and first_dark <= t <= last_dark:
        return "during"
    if last_lit or first_dark:
        return "boundary"
    # Legacy callers without passes: the noon-shift convention still applies,
    # and boundary nights remain explicitly uncertain.
    night = (t + timedelta(hours=candidate["unit"]["lon"] / 15 - 12)).date().isoformat()
    if night < candidate["start"]:
        return "before"
    if night > candidate["end"]:
        return "after"
    return "boundary" if night in (candidate["start"], candidate["end"]) else "during"


class CachedListing(RuntimeError):
    """An offline snapshot preserves review material but cannot prove coverage."""

    def __init__(self, entries):
        self.entries = entries


def cmr_granules(
    short_name: str,
    lon: float,
    lat: float,
    start: str,
    end: str,
    cache_dir: Path = CACHE,
    today: date | None = None,
    session=requests,
) -> list[dict]:
    """Page through public CMR; only windows ≥30 days old are final."""
    today = today or datetime.now(timezone.utc).date()
    key = json.dumps([short_name, round(lon, 6), round(lat, 6), start, end])
    path = cache_dir / "cmr" / (hashlib.sha256(key.encode()).hexdigest() + ".json")
    closed = date.fromisoformat(end) <= today - timedelta(days=30)
    if closed and path.exists():
        try:
            cached = json.loads(path.read_text())
            if cached["query"] == json.loads(key) and date.fromisoformat(
                cached.get("fetched_on", "1900-01-01")
            ) >= date.fromisoformat(end) + timedelta(days=30):
                return cached["entries"]
        except (ValueError, KeyError, OSError):
            pass
    entries, page = [], 1
    while True:
        try:
            response = session.get(
                CMR_URL,
                params=dict(
                    short_name=short_name,
                    point=f"{lon},{lat}",
                    temporal=utc_window(start, end, lon),
                    page_size=2000,
                    page_num=page,
                ),
                timeout=45,
            )
            response.raise_for_status()
            batch = response.json()["feed"]["entry"]
        except Exception:
            if path.exists():
                try:
                    cached = json.loads(path.read_text())
                    if cached["query"] == json.loads(key):
                        raise CachedListing(cached["entries"]) from None
                except (OSError, ValueError, KeyError):
                    pass
            raise
        entries.extend(batch)
        if len(batch) < 2000:
            break
        page += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(
            dict(query=json.loads(key), entries=entries, fetched_on=today.isoformat())
        )
    )
    tmp.replace(path)
    return entries


def public_browse(entry: dict) -> str | None:
    """Only expose public PNGs, never signed or protected download URLs."""
    for link in entry.get("links", []):
        url = link.get("href", "")
        parsed = urlparse(url)
        if (
            parsed.scheme == "https"
            and not parsed.query
            and "/lp-prod-public/" in parsed.path
            and parsed.path.lower().endswith(".png")
        ):
            return url
    return None


def scene_records(entries: list[tuple[str, dict]], candidate: dict) -> list[dict]:
    """Deduplicate products by acquisition scene, preserving product metadata."""
    records = {}
    for product, e in entries:
        producer = e.get("producer_granule_id", e.get("title", ""))
        match = re.search(r"\d{8}T\d{6}(?:_\d{7}_\d{3})?", producer)
        if not match or not e.get("time_start"):
            continue
        scene = match.group()
        t = datetime.fromisoformat(e["time_start"].replace("Z", "+00:00"))
        local_day = (
            (t + timedelta(hours=candidate["unit"]["lon"] / 15)).date().isoformat()
        )
        phase = scene_phase(t, candidate)
        record = records.setdefault(
            scene[:15],
            dict(
                scene_id=scene,
                time_utc=e["time_start"],
                local_date=local_day,
                local_night=(t + timedelta(hours=candidate["unit"]["lon"] / 15 - 12))
                .date()
                .isoformat(),
                phase=phase,
                cloud_cover=None,
                browse_url=None,
                products=[],
                caption=CAPTION,
            ),
        )
        record["products"].append(product)
        if e.get("cloud_cover") is not None:
            record["cloud_cover"] = float(e["cloud_cover"])
        record["browse_url"] = record["browse_url"] or public_browse(e)
    return sorted(records.values(), key=lambda r: r["time_utc"])


def _ee_client():
    import ee

    ee.Initialize(project="cerulean-338116")
    ee.data.setDeadline(60000)
    return ee


def _stamp_png(content: bytes, destination: Path, title: str) -> None:
    """Mark the source and a 5 km scale on a 30 km square thumbnail."""
    import io
    from PIL import Image, ImageDraw

    # Pixels outside the scene come back transparent: show them as grey, the
    # same no-data colour as masked pixels inside the CH4 index.
    raw = Image.open(io.BytesIO(content)).convert("RGBA")
    image = Image.new("RGBA", raw.size, (128, 128, 128, 255))
    image.alpha_composite(raw)
    image = image.convert("RGB")
    w, h = image.size
    draw = ImageDraw.Draw(image)
    cx, cy = w / 2, h / 2
    draw.ellipse((cx - 8, cy - 8, cx + 8, cy + 8), outline="black", width=4)
    draw.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), outline="white", width=2)
    draw.rectangle((10, h - 38, 30 + w / 6, h - 8), fill="black")
    draw.line((20, h - 18, 20 + w / 6, h - 18), fill="white", width=3)
    draw.text((20, h - 34), "5 km", fill="white")
    draw.rectangle((0, 0, w, 24), fill="black")
    draw.text((8, 5), title, fill="white")
    image.save(destination, format="PNG")


def chip_folder(scene: dict, unit: dict, cache_dir: Path, recipe=CHIP_RECIPE) -> Path:
    digest = hashlib.sha256(
        json.dumps([scene["scene_id"], unit["lat"], unit["lon"], recipe]).encode()
    ).hexdigest()[:20]
    return cache_dir / "chips" / digest


def _coverage(pixel, count, method):
    return dict(
        source_coverage=(
            "no_data"
            if pixel == 0 or count == 0
            else "covered"
            if pixel == 1 and count is not None and count > 0
            else "unknown"
        ),
        source_pixel_mask=pixel,
        source_valid_pixels=count,
        source_coverage_radius_m=300,
        source_coverage_method=method,
    )


def png_coverage(path: Path) -> dict:
    """Migrate old black-gap thumbnails without counting the stamped ring.

    At 512 pixels across 30 km, 300 m is 5.12 pixels, inside the ring.
    This is a thumbnail-pixel count, not a native-raster count.
    """
    from PIL import Image

    try:
        with Image.open(path) as original:
            im = original.convert("RGB")
            w, h = im.size
            if min(w, h) < 100:
                raise ValueError("Not a review chip")
            cx, cy = w // 2, h // 2
            radius = min(5.12, 300 * min(w, h) / 30000)
            colours = [
                im.getpixel((cx + x, cy + y))
                for x in range(-math.ceil(radius), math.ceil(radius) + 1)
                for y in range(-math.ceil(radius), math.ceil(radius) + 1)
                if x * x + y * y <= radius * radius
            ]

            def valid(colour):
                return colour not in ((0, 0, 0), (128, 128, 128))

            return _coverage(
                int(valid(im.getpixel((cx, cy)))),
                sum(map(valid, colours)),
                "cached_thumbnail_pixels",
            )
    except (OSError, ValueError):
        return _coverage(None, None, "cached_thumbnail_pixels")


def usable_chips(imagery: dict) -> bool:
    return (
        imagery.get("status") == "available"
        and imagery.get("source_coverage") == "covered"
        and bool(imagery.get("ch4") and imagery.get("rgb"))
    )


def cached_chips(scene: dict, unit: dict, cache_dir: Path) -> dict | None:
    from PIL import Image

    for recipe in (CHIP_RECIPE, "hitl-v6", "hitl-v5", "hitl-v4", "hitl-v3"):
        folder = chip_folder(scene, unit, cache_dir, recipe)
        try:
            path = folder / "metadata.json"
            result = json.loads(path.read_text())
            if result["status"] not in ("available", "no_data"):
                continue
            if "source_coverage" not in result:
                result.update(png_coverage(folder / "ch4.png"))
                if result["source_coverage"] == "no_data":
                    result.update(status="no_data", message=NO_DATA_MESSAGE)
                path.write_text(json.dumps(result, indent=2))
            # No-data is final, including a legacy tile: it has no chips to show.
            if result["source_coverage"] == "no_data":
                return {**result, "status": "no_data", "message": NO_DATA_MESSAGE}
            try:
                for name in ("ch4.png", "rgb.png"):
                    with Image.open(folder / name) as im:
                        im.verify()
            except (OSError, ValueError):
                result.update(_coverage(None, None, "unreadable_chip"))
            return {
                **result,
                "cache_folder": str(folder),
                # Older recipes may show missing data as black, not grey.
                **({"legacy_render": True} if recipe != CHIP_RECIPE else {}),
                **(
                    {"needs_render": True}
                    if recipe != CHIP_RECIPE or result["source_coverage"] == "unknown"
                    else {}
                ),
            }
        except (OSError, ValueError, KeyError):
            pass
    return None


def fold_acquisitions(scenes: list[dict]) -> tuple[list[dict], dict[str, str]]:
    """Fold same-orbit adjacent tiles within 30 s, preferring source coverage."""
    groups = []
    for scene in sorted(scenes, key=lambda s: s["time_utc"]):
        orbit = re.search(r"T\d{6}_(\d{7})_\d{3}$", scene["scene_id"])
        t = datetime.fromisoformat(scene["time_utc"].replace("Z", "+00:00"))
        for group in reversed(groups):
            first, group_orbit, stamp = group
            if orbit and orbit[1] == group_orbit and (t - stamp).total_seconds() <= 30:
                first.append(scene)
                break
        else:
            groups.append(([scene], orbit[1] if orbit else None, t))
    kept, aliases = [], {}
    order = {"covered": 0, "unknown": 1, "no_data": 2}
    for group, _, _ in groups:
        chosen = min(
            group,
            key=lambda s: (
                order.get(s.get("imagery", {}).get("source_coverage"), 1),
                s["scene_id"],
            ),
        )
        adjacent = []
        for scene in group:
            aliases[scene["scene_id"]] = chosen["scene_id"]
            if scene is not chosen:
                adjacent.append(
                    {
                        k: scene[k]
                        for k in ("scene_id", "time_utc", "browse_url")
                        if k in scene
                    }
                )
                adjacent[-1]["source_coverage"] = scene.get("imagery", {}).get(
                    "source_coverage", "unknown"
                )
        if adjacent:
            chosen["adjacent_tiles"] = adjacent
        kept.append(chosen)
    return kept, aliases


def emit_truecolor(granule: str, pt, ee) -> tuple:
    """hitl_app._emit_truecolor with its validity guards, plus a mask for EMIT edge fill."""
    attempts = (
        ("NASA/EMIT/L2A/RFL/EMIT_L2A_RFL_001", "reflectance", "L2A reflectance"),
        ("NASA/EMIT/L1B/RAD/EMIT_L1B_RAD_001", "radiance", "L1B radiance"),
    )
    for prefix, band_type, source in attempts:
        try:
            bands = [f"{band_type}_{i}" for i in (35, 23, 11)]
            asset = f"{prefix}_{granule}"
            image = ee.Image(asset).select(bands)
            # EMIT strip edges carry fill: -9999, or exact zeros in every band
            # (measured on 7533, 30 Aug 2026). Either would stretch to black; mask
            # them so they render as no-data grey and stay out of the stretch.
            image = image.updateMask(
                image.reduce(ee.Reducer.min())
                .gt(-1)
                .And(image.reduce(ee.Reducer.max()).gt(0))
            )
            stats = image.reduceRegion(
                ee.Reducer.percentile([2, 50, 98]),
                geometry=pt.buffer(20000),
                scale=120,
                bestEffort=True,
            ).getInfo()
            lows = [stats.get(f"{b}_p2") for b in bands]
            highs = [stats.get(f"{b}_p98") for b in bands]
            medians = [stats.get(f"{b}_p50") for b in bands]
            if any(v is None for v in lows + highs + medians):
                continue
            if band_type == "reflectance" and max(highs) > 1.5:
                continue
            if max(medians) / max(min(medians), 1e-9) > 5.0:
                continue
            low, high = min(lows), max(highs)
            if high - low < 1e-6:
                continue
            note = f"true colour ({source})"
            if band_type == "radiance":
                note += " — L2A unphysical on this granule"
            return (
                image,
                dict(min=low, max=high, gamma=1.8),
                dict(rgb_bands=bands, rgb_asset=asset, rgb_note=note),
            )
        except Exception:
            continue
    raise ValueError("No usable EMIT true-colour source")


def fetch_chips(
    scene: dict, unit: dict, cache_dir: Path = CACHE, ee=None, session=requests
) -> dict:
    """Render versioned review chips or cache a final no-data source result.

    Keep legacy covered chips reviewable while refreshing their recipe. Failures
    remain retryable; no signed thumbnail URL or credentials are published.
    """
    folder = chip_folder(scene, unit, cache_dir)
    metadata = folder / "metadata.json"
    cached = cached_chips(scene, unit, cache_dir)
    if cached and not cached.get("needs_render"):
        return cached
    try:
        ee = ee or _ee_client()
        if len(scene["scene_id"]) == 15:
            # Current CMR v2/v3 names omit the old orbit/scene suffix used by
            # Earth Engine v1. Match the acquisition, never invent the suffix.
            t = datetime.fromisoformat(scene["time_utc"].replace("Z", "+00:00"))
            ids = (
                ee.ImageCollection(COLLECTION)
                .filterDate(
                    (t - timedelta(seconds=1)).isoformat(),
                    (t + timedelta(seconds=1)).isoformat(),
                )
                .filterBounds(ee.Geometry.Point([unit["lon"], unit["lat"]]))
                .aggregate_array("system:index")
                .getInfo()
            )
            if not ids:
                raise ValueError("Scene not ingested")
            asset = f"{COLLECTION}/{ids[0]}"
        else:
            asset = f"{COLLECTION}/EMIT_L1B_RAD_001_{scene['scene_id']}"
        available = ee.data.getAsset(asset)
        if not available:
            raise ValueError("Scene not ingested")
    except Exception:
        # A missing asset is pending only if the collection's current ingest
        # horizon precedes it. Authentication/network errors stay unavailable.
        try:
            latest = (
                ee.ImageCollection(COLLECTION)
                .aggregate_max("system:time_start")
                .getInfo()
                if ee
                else None
            )
            acquired = datetime.fromisoformat(scene["time_utc"].replace("Z", "+00:00"))
            if latest and acquired.timestamp() * 1000 > latest:
                cutoff = datetime.fromtimestamp(latest / 1000, timezone.utc).isoformat()
                return dict(
                    status="pending",
                    message=pending_message(cutoff),
                    caption=CAPTION,
                    collection_latest_utc=cutoff,
                )
        except Exception:
            pass
        return cached or dict(
            status="unavailable",
            message=(
                "Earth Engine imagery unavailable; scene availability is unknown. "
                "Retried on the next build."
            ),
            caption=CAPTION,
            collection_latest_utc=None,
        )
    try:
        pt = ee.Geometry.Point([unit["lon"], unit["lat"]])
        # Square in a local metric projection: stamp positions/scale are fixed.
        zone = min(60, int((unit["lon"] + 180) / 6) + 1)
        crs = f"EPSG:{32600 + zone if unit['lat'] >= 0 else 32700 + zone}"
        xy = pt.transform(crs, 1).coordinates().getInfo()
        bounds = [xy[0] - 15000, xy[1] - 15000, xy[0] + 15000, xy[1] + 15000]
        region = ee.Geometry.Rectangle(bounds, crs, False)
        rad = ee.Image(asset)
        cont = rad.select([f"radiance_{i}" for i in range(231, 245)]).reduce(
            ee.Reducer.mean()
        )
        absorption = rad.select([f"radiance_{i}" for i in range(256, 269)]).reduce(
            ee.Reducer.mean()
        )
        raw = (
            cont.subtract(absorption)
            .divide(cont)
            .updateMask(cont.gt(0))
            .rename("ch4_depth")
        )
        # Normalizing the index never changes this mask. Check it first so a
        # fully empty source does not depend on normalization or RGB success.
        coverage = index_coverage(raw, pt, ee)
        if coverage["source_coverage"] == "no_data":
            result = dict(
                status="no_data",
                message=NO_DATA_MESSAGE,
                **coverage,
                recipe_version=CHIP_RECIPE,
            )
            folder.mkdir(parents=True, exist_ok=True)
            metadata.write_text(json.dumps(result, indent=2))
            return result
        stats = raw.reduceRegion(
            ee.Reducer.percentile([2, 16, 50, 84, 98]),
            geometry=pt.buffer(15000),
            scale=60,
            bestEffort=True,
        ).getInfo()
        med = stats["ch4_depth_p50"]
        sigma = (stats["ch4_depth_p84"] - stats["ch4_depth_p16"]) / 2
        if sigma < 1e-5:
            sigma = (stats["ch4_depth_p98"] - stats["ch4_depth_p2"]) / 4
        if not math.isfinite(sigma) or sigma < 1e-5:
            raise ValueError("No finite normalization")
        index = raw.subtract(med).divide(sigma)
        display = index.subtract(1).divide(5).clamp(0, 1).pow(0.65)
        granule = asset.rsplit("EMIT_L1B_RAD_001_", 1)[1]
        rgb, rgb_vis, rgb_recipe = emit_truecolor(granule, pt, ee)
        views = [
            (
                "ch4.png",
                display,
                dict(min=0, max=1, palette=["00131f", "7e3d01", "fd8e28", "fff4d0"]),
                "CH4 index: 1-6 sigma",
            ),
            (
                "rgb.png",
                rgb,
                rgb_vis,
                rgb_recipe["rgb_note"],
            ),
        ]
        folder.mkdir(parents=True, exist_ok=True)
        for name, im, vis, label in views:
            if name == "ch4.png":
                im = im.visualize(**vis).unmask(ee.Image.constant([128, 128, 128]))
                vis = {}
            url = im.getThumbURL(
                dict(region=region, crs=crs, dimensions=512, format="png", **vis)
            )
            response = session.get(url, timeout=90)
            response.raise_for_status()
            _stamp_png(response.content, folder / name, label)
        result = dict(
            status="available",
            **coverage,
            recipe_version=CHIP_RECIPE,
            caption=CAPTION,
            normalization=dict(median=med, robust_sigma=sigma, radius_m=15000),
            asset=asset,
            recipe=dict(
                **rgb_recipe,
                rgb_percentiles=[2, 98],
                rgb_radius_m=20000,
                index_gamma=0.65,
                rgb_scale_m=120,
                rgb_gamma=1.8,
                rgb_min=rgb_vis["min"],
                rgb_max=rgb_vis["max"],
            ),
        )
        metadata.write_text(json.dumps(result, indent=2))
        return {**result, "cache_folder": str(folder)}
    except Exception:
        return cached or dict(
            status="unavailable",
            message=(
                "Scene listed; chip rendering unavailable. Retried on the next build."
            ),
            caption=CAPTION,
        )


def index_coverage(index, pt, ee):
    """Check the actual rendered index mask and valid native pixels nearby."""
    try:
        pixel = (
            index.mask()
            .unmask(0)
            .reduceRegion(
                ee.Reducer.first(),
                geometry=pt,
                scale=60,
            )
            .getInfo()
            .get("ch4_depth")
        )
        count = (
            index.reduceRegion(
                ee.Reducer.count(),
                geometry=pt.buffer(300),
                scale=60,
            )
            .getInfo()
            .get("ch4_depth")
        )
        pixel = int(pixel > 0) if isinstance(pixel, (int, float)) else None
        count = int(count) if isinstance(count, (int, float)) else None
        return _coverage(pixel, count, "rendered_index_mask")
    except Exception:
        return _coverage(None, None, "rendered_index_mask")


def attach_chips(scene, chips, site_dir, prefix):
    """Publish usable chips only; keep one shared caption outside each scene."""
    chips = dict(chips)
    folder = chips.pop("cache_folder", None)
    chips.pop("caption", None)
    legacy = chips.pop("needs_render", False)
    chips.setdefault("source_coverage", "unknown")
    if folder and site_dir and chips.get("source_coverage") == "covered":
        for name in ("ch4.png", "rgb.png"):
            path = Path(folder) / name
            if not path.exists():
                continue
            dest = site_dir / "img" / "emit" / f"{prefix}_{scene['scene_id']}_{name}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            if legacy and name == "ch4.png":
                _copy_legacy_index(path, dest)
            else:
                shutil.copyfile(path, dest)
            chips[name.removesuffix(".png")] = str(dest.relative_to(site_dir))
    scene["imagery"] = chips


def _copy_legacy_index(source, destination):
    """Show legacy black gaps in grey while awaiting the new EE recipe.

    Pure black is outside the index palette. Preserve the title, scale bar
    and source ring; change data pixels only, leaving the cached PNG intact.
    """
    import numpy as np
    from PIL import Image

    with Image.open(source) as im:
        pixels = np.array(im.convert("RGB"))
    h, w, _ = pixels.shape
    gap = (pixels == 0).all(axis=2)
    gap[:24] = False
    gap[h - 38 : h - 7, 10 : math.ceil(31 + w / 6)] = False
    cy, cx = h // 2, w // 2
    gap[cy - 10 : cy + 11, cx - 10 : cx + 11] = False
    pixels[gap] = [128, 128, 128]
    Image.fromarray(pixels).save(destination)


def enrich(
    payload: dict,
    cache_dir: Path = CACHE,
    site_dir: Path | None = None,
    session=requests,
    imagery: bool = True,
) -> dict:
    """Attach public listing and imagery statuses without failing a site build."""
    client = None
    tried_client = False
    for candidate in payload["candidates"]:  # every tier, Watch included (D61)
        start, end = cmr_window(candidate, payload["as_of"])
        entries, failures = [], []
        u = candidate["unit"]
        for product in PRODUCTS:
            try:
                entries.extend(
                    (product, e)
                    for e in cmr_granules(
                        product,
                        u["lon"],
                        u["lat"],
                        start,
                        end,
                        cache_dir=cache_dir,
                        session=session,
                    )
                )
            except CachedListing as cached:
                entries.extend((product, e) for e in cached.entries)
                failures.append(product)
            except Exception:
                failures.append(product)  # do not leak request URLs or credentials
        scenes = scene_records(entries, candidate)
        for scene in scenes:
            if imagery:
                cached = cached_chips(scene, u, cache_dir)
                if (not cached or cached.get("needs_render")) and not tried_client:
                    tried_client = True
                    try:
                        client = _ee_client()
                    except Exception:
                        pass
                chips = (
                    cached if cached and not cached.get("needs_render") else None
                ) or (
                    fetch_chips(scene, u, cache_dir, ee=client, session=session)
                    if client is not None
                    else cached
                    or dict(
                        status="unavailable",
                        message=(
                            "Earth Engine connection unavailable; scene availability "
                            "is unknown. Retried on the next build."
                        ),
                        collection_latest_utc=None,
                        caption=CAPTION,
                    )
                )
                if cached and chips["status"] not in ("available", "no_data"):
                    chips = cached
            else:
                chips = dict(
                    status="unavailable",
                    message="Imagery disabled for this build.",
                    caption=CAPTION,
                )
            scene.pop("caption", None)
            attach_chips(scene, chips, site_dir, candidate["id"])
        scenes, _ = fold_acquisitions(scenes)
        candidate["emit"] = dict(
            status="unavailable"
            if len(failures) == 2
            else "partial"
            if failures
            else "available",
            window_start=start,
            window_end=end,
            failed_products=failures,
            scenes=scenes,
            with_chips=sum(usable_chips(s["imagery"]) for s in scenes),
            during_count=sum(s["phase"] == "during" for s in scenes),
            nearest_scene=min(
                scenes,
                key=lambda s: abs(
                    (
                        date.fromisoformat(s["local_date"])
                        - date.fromisoformat(candidate["start"])
                    ).days
                ),
            )["scene_id"]
            if scenes
            else None,
            phase_basis=(
                "Local night = UTC + longitude/15 h −12 h. Before/after are "
                "bracketed by last lit/recovery VIIRS passes; during lies between "
                "first and last dark passes; transitions are boundary."
            ),
            caption=CAPTION,
        )
    return payload


def main(payload=None, site_dir=None):
    if payload is None:
        import flareouts

        payload = flareouts.main()
    return enrich(payload, site_dir=site_dir)


# %%
if __name__ == "__main__":
    main()
