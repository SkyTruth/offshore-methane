"""Round-three source attribution, coverage and acquisition checks, offline."""

import json
from unittest.mock import Mock

import darknights as d
import emit
import pandas as pd
import pytest
from PIL import Image
from test_darknights import frames, listing, screen
from test_flareouts import candidate


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(
        emit.requests.sessions.Session,
        "request",
        Mock(side_effect=AssertionError("offline")),
    )
    monkeypatch.setattr(
        emit.requests, "get", Mock(side_effect=AssertionError("offline"))
    )
    monkeypatch.setattr(emit, "_ee_client", Mock(side_effect=AssertionError("offline")))


def chip(tmp_path, scene, unit, colour, recipe="hitl-v3"):
    folder = emit.chip_folder(scene, unit, tmp_path, recipe)
    folder.mkdir(parents=True)
    (folder / "metadata.json").write_text(
        json.dumps({"status": "available", "caption": emit.CAPTION})
    )
    for name in ("ch4.png", "rgb.png"):
        Image.new("RGB", (512, 512), colour).save(folder / name)
    return folder


@pytest.mark.parametrize(
    "pixel,count,coverage",
    [
        (1, 85, "covered"),
        (0, 0, "no_data"),
        (0, 80, "no_data"),
        (None, 80, "unknown"),
        (1, None, "unknown"),
    ],
)
def test_index_mask_and_nearby_count(pixel, count, coverage):
    index, ee = Mock(), Mock()
    mask = index.mask.return_value.unmask.return_value
    mask.reduceRegion.return_value.getInfo.return_value = {"ch4_depth": pixel}
    index.reduceRegion.return_value.getInfo.return_value = {"ch4_depth": count}
    result = emit.index_coverage(index, Mock(), ee)
    assert result["source_coverage"] == coverage
    assert result["source_pixel_mask"] == pixel
    assert result["source_valid_pixels"] == count
    assert result["source_coverage_radius_m"] == 300
    index.mask.return_value.unmask.assert_called_once_with(0)


@pytest.mark.parametrize(
    "colour,coverage", [("#00131f", "covered"), ("black", "no_data")]
)
def test_legacy_cache_migration_and_final_no_data(tmp_path, colour, coverage):
    scene, unit = {"scene_id": "20260822T014348_2623401_008"}, candidate()["unit"]
    folder = chip(tmp_path, scene, unit, colour)
    result = emit.cached_chips(scene, unit, tmp_path)
    assert result["source_coverage"] == coverage
    assert result["source_pixel_mask"] == (coverage == "covered")
    assert result["source_valid_pixels"] == (89 if coverage == "covered" else 0)
    assert (
        json.loads((folder / "metadata.json").read_text())["source_coverage"]
        == coverage
    )
    assert emit.chip_folder(scene, unit, tmp_path) != folder
    if coverage == "no_data":
        assert result["status"] == "no_data"
        assert emit.fetch_chips(scene, unit, tmp_path, ee=Mock()) == result
    else:
        assert result["needs_render"] is True


@pytest.mark.parametrize("missing", [True, False])
def test_unreadable_or_missing_png_stays_unknown(tmp_path, missing):
    scene, unit = {"scene_id": "test"}, candidate()["unit"]
    folder = chip(tmp_path, scene, unit, "#00131f")
    if missing:
        (folder / "ch4.png").unlink()
    else:
        (folder / "ch4.png").write_bytes(b"not a PNG")
    assert emit.cached_chips(scene, unit, tmp_path)["source_coverage"] == "unknown"


def test_no_data_fetch_is_final_without_rgb_or_normalization(tmp_path, monkeypatch):
    scene = {
        "scene_id": "20260822T014348_2623401_008",
        "time_utc": "2026-08-22T01:43:48Z",
    }
    ee, session = Mock(), Mock()
    point = ee.Geometry.Point.return_value
    point.transform.return_value.coordinates.return_value.getInfo.return_value = [0, 0]
    monkeypatch.setattr(
        emit, "index_coverage", lambda *a: emit._coverage(0, 0, "rendered_index_mask")
    )
    result = emit.fetch_chips(
        scene, candidate()["unit"], tmp_path, ee=ee, session=session
    )
    assert result["status"] == "no_data"
    assert not session.get.called
    ee.reset_mock()
    assert emit.fetch_chips(scene, candidate()["unit"], tmp_path, ee=ee) == result
    assert not ee.data.getAsset.called


def test_same_orbit_fold_prefers_covered_and_keeps_browse():
    scenes = [
        {
            "scene_id": f"20260822T0143{second}_2623401_{tile}",
            "time_utc": f"2026-08-22T01:43:{second}Z",
            "browse_url": f"https://example.org/{tile}.png",
            "imagery": {"source_coverage": coverage},
        }
        for second, tile, coverage in [
            ("36", "007", "covered"),
            ("48", "008", "no_data"),
        ]
    ]
    folded, aliases = emit.fold_acquisitions(scenes)
    assert len(folded) == 1
    assert aliases[scenes[1]["scene_id"]] == scenes[0]["scene_id"]
    assert folded[0]["adjacent_tiles"][0]["source_coverage"] == "no_data"
    assert folded[0]["adjacent_tiles"][0]["browse_url"].endswith("008.png")
    scenes[1]["scene_id"] = scenes[1]["scene_id"].replace("2623401", "2623402")
    assert len(emit.fold_acquisitions(scenes)[0]) == 2


@pytest.mark.parametrize(
    "coverage,expected", [("covered", 1), ("no_data", 0), ("unknown", 0)]
)
def test_both_modes_count_only_covered_chips(tmp_path, monkeypatch, coverage, expected):
    listing(monkeypatch, "2026-08-21T15:00:00Z")
    monkeypatch.setattr(emit, "_ee_client", lambda: Mock())
    monkeypatch.setattr(emit, "cached_chips", lambda *a: None)
    monkeypatch.setattr(
        emit,
        "fetch_chips",
        lambda *a, **kw: {
            "status": "no_data" if coverage == "no_data" else "available",
            "source_coverage": coverage,
            "ch4": "index.png",
            "rgb": "rgb.png",
            "caption": emit.CAPTION,
        },
    )
    result = d.enrich(screen(*frames()), cache_dir=tmp_path)
    assert result["counts"]["with_chips"] == expected
    c = candidate()
    session = Mock()
    session.get.return_value.json.return_value = {
        "feed": {
            "entry": [
                {
                    "producer_granule_id": (
                        "EMIT_L1B_RAD_001_20261025T154451_2629810_001"
                    ),
                    "time_start": "2026-10-25T15:44:51Z",
                }
            ]
        }
    }
    emit.enrich(
        {"as_of": "2026-11-04", "candidates": [c]}, cache_dir=tmp_path, session=session
    )
    assert c["emit"]["with_chips"] == expected
    assert "caption" not in c["emit"]["scenes"][0]["imagery"]


def test_comparisons_drop_unknown_and_empty_source_scenes(tmp_path, monkeypatch):
    listing(
        monkeypatch,
        "2026-08-10T15:00:00Z",
        "2026-08-11T15:00:00Z",
        "2026-08-12T15:00:00Z",
    )
    coverage = {"10": "no_data", "11": "unknown", "12": "covered"}
    monkeypatch.setattr(
        emit,
        "cached_chips",
        lambda scene, *a: {
            "status": "available",
            "source_coverage": coverage[scene["time_utc"][8:10]],
        },
    )
    result = d.enrich(screen(*frames()), cache_dir=tmp_path)
    source = result["sources"]["1"]
    assert len(source["comparison_scenes"]) == 1
    assert source["scenes"][source["comparison_scenes"][0]]["time_utc"].startswith(
        "2026-08-12"
    )


def test_rate_source_rule_and_neighbour_denominator():
    days = ["2026-08-01", "2026-08-02", "2026-08-03", "2026-08-04"]
    h = pd.DataFrame({"lit": [True] * 4}, index=days)
    cp = pd.DataFrame(
        {
            "night": days,
            "centroid_lat": [0.0] * 4,
            "centroid_lon": [0.008, -0.011, 0.012, 0.0091],
        }
    )
    unit = {"id": "1", "lat": 0.0, "lon": 0.0}
    neighbour = {"id": "2", "lat": 0.0, "lon": 0.014}
    result, attributed = d.rate_history(h, cp, unit, [unit, neighbour])
    # Within 1 km even nearer another unit; >1 km but nearest this unit;
    # >1 km and nearest another unit (two boundary cases).
    assert attributed.rate_source.tolist() == [True, True, False, False]
    assert result.neighbour_only.tolist() == [False, False, True, True]
    assert result.lit.all()  # all still veto a dark night


def test_neighbour_only_nights_are_excluded_not_dark(tmp_path, monkeypatch):
    n, p, c = frames()
    units = [
        {"id": "1", "site_ids": [1], "lat": 0.0, "lon": 0.0, "region": "test"},
        {"id": "2", "site_ids": [2], "lat": 0.0, "lon": 0.014, "region": "test"},
    ]
    monkeypatch.setattr(d.f, "source_units", lambda *a: units)
    times = pd.to_datetime(c.time_utc)
    # 16 neighbour-only qualified nights in the selected night's window.
    neighbour = times.dt.day.between(9, 25)
    c.loc[neighbour, "centroid_lon"] = 0.012
    result = screen(n, p, c)
    run = next(r for r in result["runs"] if r["source_id"] == "1")
    rate = run["nights"][0]["burning_rate"]
    assert rate == {"lit": 12, "qualified": 12, "neighbour_only": 16}
    assert run["clear_dark_nights"] == 1
    # Leaving fewer than ten own/dark nights removes it, without creating dark nights.
    c.loc[times.dt.day.between(6, 26), "centroid_lon"] = 0.012
    assert not screen(n, p, c)["runs"]


def test_neighbour_cannot_supply_lit_before_run(monkeypatch):
    n, p, c = frames(days=30, dark=(14,))
    units = [
        {"id": "1", "site_ids": [1], "lat": 0.0, "lon": 0.0, "region": "test"},
        {"id": "2", "site_ids": [2], "lat": 0.0, "lon": 0.014, "region": "test"},
    ]
    monkeypatch.setattr(d.f, "source_units", lambda *a: units)
    c.loc[pd.to_datetime(c.time_utc) < pd.Timestamp("2026-08-16"), "centroid_lon"] = (
        0.012
    )
    assert not screen(n, p, c)["runs"]


def test_both_modes_fold_adjacent_tiles_before_counting(tmp_path, monkeypatch):
    rows = [
        {
            "producer_granule_id": (
                f"EMIT_L1B_RAD_001_20260821T1500{sec}_2623301_{tile}"
            ),
            "time_start": f"2026-08-21T15:00:{sec}Z",
        }
        for sec, tile in [("00", "007"), ("12", "008")]
    ]
    monkeypatch.setattr(emit, "cmr_granules", lambda *a, **kw: rows)
    monkeypatch.setattr(
        emit,
        "cached_chips",
        lambda scene, *a: {
            "status": "available" if scene["scene_id"].endswith("007") else "no_data",
            "source_coverage": "covered"
            if scene["scene_id"].endswith("007")
            else "no_data",
            "ch4": "index.png",
            "rgb": "rgb.png",
        },
    )
    result = d.enrich(screen(*frames()), cache_dir=tmp_path)
    run = result["runs"][0]
    assert len(run["emit_scenes"]) == 1
    scene = result["sources"]["1"]["scenes"][run["emit_scenes"][0][0]]
    assert scene["scene_id"].endswith("007")
    assert len(scene["adjacent_tiles"]) == 1
    assert result["counts"]["with_chips"] == 1
    c = candidate()
    emit.enrich({"as_of": "2026-11-04", "candidates": [c]}, cache_dir=tmp_path)
    assert len(c["emit"]["scenes"]) == 1
    assert c["emit"]["with_chips"] == 1


def test_covered_metadata_cannot_hide_corrupt_cache(tmp_path):
    scene, unit = {"scene_id": "test"}, candidate()["unit"]
    folder = chip(tmp_path, scene, unit, "#00131f", recipe=emit.CHIP_RECIPE)
    metadata = folder / "metadata.json"
    metadata.write_text(
        json.dumps({"status": "available", "source_coverage": "covered"})
    )
    (folder / "ch4.png").write_bytes(b"corrupt")
    result = emit.cached_chips(scene, unit, tmp_path)
    assert result["source_coverage"] == "unknown"
    assert result["needs_render"] is True


def test_legacy_display_uses_grey_preserves_palette_and_stamps(tmp_path):
    source, destination = tmp_path / "old.png", tmp_path / "new.png"
    image = Image.new("RGB", (512, 512), "black")
    image.putpixel((200, 200), (0, 19, 31))
    image.save(source)
    emit._copy_legacy_index(source, destination)
    with Image.open(destination) as output:
        assert output.getpixel((100, 100)) == (128, 128, 128)
        assert output.getpixel((200, 200)) == (0, 19, 31)
        assert output.getpixel((20, 5)) == (0, 0, 0)  # title
        assert output.getpixel((20, 490)) == (0, 0, 0)  # scale
        assert output.getpixel((256, 256)) == (0, 0, 0)  # source ring
    with Image.open(source) as original:
        assert original.getpixel((100, 100)) == (0, 0, 0)
