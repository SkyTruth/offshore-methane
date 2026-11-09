"""Round-4 regressions: neighbour-detected dark nights, the neighbour-only strip
state, grey no-data in chips, and explicit unknown coverage on every scene."""

import io
from unittest.mock import Mock

import pandas as pd
import pytest
from PIL import Image

import darknights as d
import emit
from test_darknights import at, frames, listing, night_rows, screen


@pytest.fixture(autouse=True)
def no_live_services(monkeypatch):
    monkeypatch.setattr(
        d.requests, "get", Mock(side_effect=AssertionError("offline test"))
    )
    monkeypatch.setattr(
        emit, "_ee_client", Mock(side_effect=AssertionError("offline test"))
    )


def neighbour_pair(neighbour_lit_on_dark_night: bool):
    """Source 1 at lon 0, dark on night 20. Source 2 sits 1.34 km east; its
    night-20 detection lands 1.61 km from source 1 (outside the darkness
    aperture) but nearest source 2, as at 3780 / 3779-3781 on 1 Oct 2026."""
    a = frames(dark=(20,))
    ns, ps, cs = frames(dark=() if neighbour_lit_on_dark_night else (20,))
    ns["site_id"] = 2
    ns["lon"] = 0.012
    ps["site_id"] = 2
    cs["assigned_site"] = 2
    cs["group_id"] = 2
    cs["centroid_lon"] = 0.012
    night20 = pd.to_datetime(cs.time_utc).dt.strftime("%Y-%m-%d") == "2026-08-22"
    cs.loc[night20, "centroid_lon"] = 0.0145
    return tuple(pd.concat([x, y], ignore_index=True) for x, y in zip(a, (ns, ps, cs)))


def source_one_nights(payload):
    return [n["night"] for n in night_rows(payload) if n["unit"]["id"] == "1"]


def test_dark_night_is_not_listed_when_a_neighbour_within_1500_m_was_detected():
    result = screen(*neighbour_pair(neighbour_lit_on_dark_night=True))
    assert "2026-08-21" not in source_one_nights(result)
    assert result["counts"]["skipped_neighbour_detected"] >= 1


def test_dark_night_is_listed_when_the_neighbour_was_also_dark():
    result = screen(*neighbour_pair(neighbour_lit_on_dark_night=False))
    assert "2026-08-21" in source_one_nights(result)


def test_neighbour_only_nights_have_their_own_strip_state():
    days = pd.date_range("2026-08-01", periods=10).strftime("%Y-%m-%d")
    h = pd.DataFrame(
        {
            "lit": [True] * 10,
            "qualified": [True] * 10,
            "heat_mw": [2.0] * 10,
            "neighbour_only": [False] * 10,
        },
        index=days,
    )
    h.loc[days[4], ["lit", "heat_mw"]] = [False, float("nan")]
    h.loc[days[6], ["heat_mw", "neighbour_only"]] = [float("nan"), True]
    h.loc[days[7], "heat_mw"] = float("nan")
    blocks = d._series(h, [{"start": days[4], "end": days[4]}])
    state = "".join(b["state"] for b in blocks)
    assert state[4] == "D"
    assert state[6] == "N"
    assert state[7] == "H"
    assert "N" in screen(*frames())["series_states"]


def test_stamp_fills_transparent_pixels_grey(tmp_path):
    raw = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    for x in range(0, 30):
        for y in range(40, 60):
            raw.putpixel((x, y), (200, 10, 10, 255))
    buf = io.BytesIO()
    raw.save(buf, format="PNG")
    out = tmp_path / "chip.png"
    emit._stamp_png(buf.getvalue(), out, "test")
    stamped = Image.open(out).convert("RGB")
    assert stamped.getpixel((80, 40)) == (128, 128, 128)
    assert stamped.getpixel((5, 50)) == (200, 10, 10)


def test_every_published_scene_states_its_coverage(monkeypatch, tmp_path):
    listing(monkeypatch, "2026-08-21T15:00:00Z", "2026-08-10T15:00:00Z")
    client = Mock()
    collection = client.ImageCollection.return_value
    collection.aggregate_max.return_value.getInfo.return_value = (
        pd.Timestamp("2026-08-15", tz="UTC").timestamp() * 1000
    )
    monkeypatch.setattr(emit, "_ee_client", lambda: client)
    monkeypatch.setattr(
        emit,
        "fetch_chips",
        Mock(return_value={"status": "unavailable", "message": "fixture render"}),
    )
    result = d.enrich(screen(*frames()), cache_dir=tmp_path, max_renders=1)
    assert at(result, 20)["emit"]["scenes"]
    for source in result["sources"].values():
        for scene in source["scenes"].values():
            assert scene["imagery"]["source_coverage"] in ("covered", "no_data", "unknown")


def test_far_detections_do_not_count_as_a_neighbours_own():
    """A lone copy 2 km from every unit is owned by none: no skip from it."""
    units = [
        {"id": "1", "lat": 0.0, "lon": 0.0},
        {"id": "2", "lat": 0.0, "lon": 0.012},
    ]
    copies = pd.DataFrame(
        {
            "night": ["2026-08-21", "2026-08-22"],
            "centroid_lat": [0.0, 0.0],
            "centroid_lon": [0.032, 0.0125],
        }
    )
    busy = d.neighbour_detection_nights(copies, units)
    assert "2026-08-21" not in busy["1"]
    assert "2026-08-22" in busy["1"]
