"""Offline thermal-screen, CMR/EE and operational-runner regressions."""

import json
from datetime import date
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

import build_viewer
import emit
import flareouts as f
import update


def history_fixture(
    days=80, outage=55, neighbour=False, turndown=False, reassigned=False, cloudy=False
):
    """Three satellites, heat variation and routine isolated baseline misses."""
    nights, passes, copies = [], [], []
    rng = np.random.default_rng(19)
    for sid in [1, 2] if neighbour else [1]:
        lat, lon = 24.0, 53.0 + (sid - 1) * 0.07
        for i, dt in enumerate(pd.date_range("2026-08-01", periods=days)):
            nt = dt.strftime("%Y-%m-%d")
            lit = i < outage
            reduced = turndown and i >= 35
            if turndown:
                lit = True
            if reassigned:
                lit = True
            nights.append(dict(site_id=sid, night=nt, lat=lat, lon=lon))
            for sat_i, sat in enumerate(["SNPP", "NOAA-20", "NOAA-21"]):
                # local evening date at 53E: UTC next day 00-02h
                t = (dt + pd.Timedelta(days=1, hours=sat_i)).isoformat()
                detected = lit and i % 17 != 4
                rh = (0.6 if reduced else 12.0) * np.exp(rng.normal(0, 0.12))
                assigned = 9 if reassigned and i >= outage else sid
                passes.append(
                    dict(
                        site_id=sid,
                        satellite=sat,
                        granule=f"g{i}_{sat}",
                        time_utc=t,
                        sza=140,
                        cloud_mask=3 if cloudy and i >= outage else 0,
                        cloud_clear_neighbors=8,
                        dist_m=30.0,
                        detected=detected and assigned == sid,
                        rh_mw=rh if detected and assigned == sid else 0,
                    )
                )
                if detected:
                    copies.append(
                        dict(
                            satellite=sat,
                            granule=f"g{i}_{sat}",
                            group_id=sid,
                            time_utc=t,
                            assigned_site=assigned,
                            centroid_lat=lat + rng.normal(0, 0.0001),
                            centroid_lon=lon + rng.normal(0, 0.0001),
                            group_rh_mw=rh,
                            cloud_mask_peak=0,
                        )
                    )
    return pd.DataFrame(nights), pd.DataFrame(passes), pd.DataFrame(copies)


def fixture_states(n, p, c):
    days = sorted(n.night.unique())
    p = f.prepare_passes(n, p)
    c = f.prepare_copies(c, days[0], days[-1])
    unit = next(u for u in f.source_units(c, days[29]) if 1 in u["site_ids"])
    h, _, _ = f.unit_history(unit, p, c, days)
    return f.heat_states(h)


def test_injected_outage_flagged_and_json_serializable():
    data = f.screen(*history_fixture())
    assert len(data["candidates"]) == 1
    c = data["candidates"][0]
    assert c["tier"] == "Mid"
    assert c["unit"]["site_ids"] == [1]
    assert c["dark_qualified_nights"] == 25
    assert c["status"] == "ongoing"
    assert c["satellites"] == ["NOAA-20", "NOAA-21", "SNPP"]
    assert c["context"]["label"] == "no comparison flare within 10 km"
    assert data["calibration"]["miss_persistence_used"] >= 0.59
    assert c["pass_list"][0]["status"] == "clear miss"
    json.dumps(data, allow_nan=False)


def test_turndown_is_reduced_heat_not_outage():
    data = f.screen(*history_fixture(turndown=True))
    assert not data["candidates"]
    states = fixture_states(*history_fixture(turndown=True))
    reduced = [s for s in states if s["state"] == "reduced heat"]
    assert reduced
    assert reduced[0]["start"] == "2026-09-05"
    assert reduced[0]["median_mw"] < 1


def test_prior_turndown_not_redated_at_later_darkness():
    n, p, c = history_fixture(turndown=True, days=100)
    cutoff = "2026-10-01"
    p.loc[p.time_utc >= cutoff, "detected"] = False
    c = c[c.time_utc < cutoff]
    data = f.screen(n, p, c)
    assert data["candidates"][0]["tier"] == "Watch"
    assert data["candidates"][0]["pre_spell_state"]["state"] == "reduced heat"


def test_reassignment_and_unassigned_source_reset_absence():
    for assigned in (9, np.nan):
        n, p, c = history_fixture(reassigned=True)
        c.loc[c.assigned_site == 9, "assigned_site"] = assigned
        assert not f.screen(n, p, c)["candidates"]


def test_cloudy_nondetections_do_not_count_as_dark_opportunities():
    assert not f.screen(*history_fixture(cloudy=True))["candidates"]


def test_shared_drop_in_independent_neighbour_is_mid():
    data = f.screen(*history_fixture(neighbour=True))
    assert len(data["candidates"]) == 2
    for c in data["candidates"]:
        assert c["tier"] == "Mid"
        assert c["context"]["label"] == "shared heat loss"
        assert c["context"]["scales_km"]["10"]["shared"] == 1
        assert c["context"]["neighbours"][0]["unit"] != c["unit"]


def test_neighbour_that_stays_lit_but_loses_heat_is_shared():
    n, p, c = history_fixture(neighbour=True)
    dt = pd.to_datetime("2026-08-01") + pd.Timedelta(days=55)
    for i in range(55, 80):
        for sat_i, sat in enumerate(["SNPP", "NOAA-20", "NOAA-21"]):
            c.loc[len(c)] = dict(
                satellite=sat,
                granule=f"g{i}_{sat}",
                group_id=2,
                time_utc=(dt + pd.Timedelta(days=i - 55 + 1, hours=sat_i)).isoformat(),
                assigned_site=2,
                centroid_lat=24.0,
                centroid_lon=53.07,
                group_rh_mw=0.5,
                cloud_mask_peak=0,
            )
    result = f.screen(n, p, c)
    target = next(x for x in result["candidates"] if x["unit"]["site_ids"] == [1])
    donor = target["context"]["neighbours"][0]
    assert donor["shared_heat_loss"]
    assert donor["spell_median_mw"] == 0.5
    assert all(r["lit"] for r in donor["series"][-25:])


@pytest.mark.parametrize(
    "bright,shared,score,k,resolved,tier",
    [
        (True, False, 0.01, 14, True, "High"),
        (True, True, 0.01, 14, True, "Mid"),
        (True, False, 4.0, 6, True, "Mid"),
        (True, False, 0.01, 14, False, "Mid"),
        (False, False, 0.001, 40, True, "Watch"),
    ],
)
def test_exact_tier_rules(bright, shared, score, k, resolved, tier):
    assert f.classify_tier(bright, shared, score, k, resolved) == tier


def test_heat_bound_reports_conditional_limit_and_small_flame_ambiguity():
    curve = [
        dict(lower_mw=0, upper_mw=0.2, probability=0.19, trials=934),
        dict(lower_mw=0.2, upper_mw=0.3, probability=0.341, trials=3071),
        dict(lower_mw=0.3, upper_mw=0.5, probability=0.532, trials=7227),
    ]
    assert f.heat_bound(14, curve)["excluded_from_mw"] == 0.2
    assert f.heat_bound(6, curve)["excluded_from_mw"] == 0.3
    assert f.heat_bound(44, curve)["all_detected_classes_excluded"]
    assert "untested" in f.heat_bound(44, curve)["assumption"]


def candidate():
    return dict(
        id="1_2026-10-30",
        tier="Mid",
        start="2026-10-30",
        end="2026-11-04",
        status="ongoing",
        unit=dict(site_ids=[1], lat=19.6, lon=-92.3),
    )


def test_cmr_window_ongoing_and_ended_cap():
    c = candidate()
    assert emit.cmr_window(c, "2026-11-04") == ("2026-10-16", "2026-11-04")
    c.update(status="ended", end="2026-10-31")
    assert emit.cmr_window(c, "2026-11-30") == ("2026-10-16", "2026-11-14")


def fake_response(entries):
    return SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: {"feed": {"entry": entries}}
    )


def test_cmr_only_closed_windows_cached_and_open_retried(tmp_path):
    session = Mock()
    session.get.return_value = fake_response([])

    def fetch(end):
        return emit.cmr_granules(
            "EMITL1BRAD",
            1,
            2,
            "2026-10-01",
            end,
            cache_dir=tmp_path,
            today=date(2026, 11, 9),
            session=session,
        )

    fetch("2026-10-09")
    fetch("2026-10-09")
    assert session.get.call_count == 1
    fetch("2026-11-09")
    fetch("2026-11-09")
    assert session.get.call_count == 3
    assert session.get.call_args.kwargs["params"]["point"] == "1,2"


def test_cmr_pagination_and_failed_request_not_cached(tmp_path):
    session = Mock()
    session.get.side_effect = [fake_response([{}] * 2000), fake_response([{}])]
    result = emit.cmr_granules(
        "EMITL1BRAD",
        1,
        2,
        "2026-10-01",
        "2026-10-02",
        cache_dir=tmp_path,
        session=session,
    )
    assert len(result) == 2001
    assert session.get.call_args.kwargs["params"]["page_num"] == 2
    session.get.side_effect = RuntimeError("network")
    with pytest.raises(RuntimeError):
        emit.cmr_granules(
            "EMITL2BCH4ENH",
            1,
            2,
            "2026-10-01",
            "2026-10-02",
            cache_dir=tmp_path,
            session=session,
        )
    assert len(list(tmp_path.rglob("*.json"))) == 1


def test_cmr_current_product_names_dedup_and_public_browse():
    entries = []
    for product in emit.PRODUCTS:
        entries.append(
            (
                product,
                dict(
                    producer_granule_id="EMIT_L1B_RAD_002_20261025T154451",
                    time_start="2026-10-25T15:44:51Z",
                    cloud_cover="76",
                    links=[
                        {
                            "href": "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-public/a.png"
                        }
                    ],
                ),
            )
        )
    result = emit.scene_records(entries, candidate())
    assert len(result) == 1
    assert result[0]["scene_id"] == "20261025T154451"
    assert result[0]["phase"] == "before"
    assert result[0]["cloud_cover"] == 76
    assert len(result[0]["products"]) == 2
    assert (
        emit.public_browse(
            dict(links=[{"href": "https://x/lp-prod-public/a.png?token=hidden"}])
        )
        is None
    )


def test_emit_network_failure_is_graceful(tmp_path):
    session = Mock()
    session.get.side_effect = RuntimeError("network")
    p = dict(as_of="2026-11-04", candidates=[candidate()])
    result = emit.enrich(p, cache_dir=tmp_path, session=session)
    assert result["candidates"][0]["emit"]["status"] == "unavailable"
    assert not list(tmp_path.rglob("*.json"))


def test_emit_lists_overpasses_for_every_tier(tmp_path):
    entry = dict(
        producer_granule_id="EMIT_L1B_RAD_002_20261101T154451",
        time_start="2026-11-01T15:44:51Z",
    )
    session = Mock()
    session.get.return_value = fake_response([entry])
    cards = []
    for tier in ("High", "Mid", "Watch"):
        c = candidate()
        c.update(id=f"{tier}_2026-10-30", tier=tier)
        cards.append(c)
    p = dict(as_of="2026-11-04", candidates=cards)
    emit.enrich(p, cache_dir=tmp_path, session=session, imagery=False)
    for c in p["candidates"]:
        assert c["emit"]["status"] == "available"
        assert [s["scene_id"] for s in c["emit"]["scenes"]] == ["20261101T154451"]


def test_ee_missing_recent_scene_is_pending_and_retried(tmp_path):
    ee = Mock()
    ee.ImageCollection.return_value.filterDate.return_value.filterBounds.return_value.aggregate_array.return_value.getInfo.return_value = []
    ee.ImageCollection.return_value.aggregate_max.return_value.getInfo.return_value = (
        pd.Timestamp("2026-08-31", tz="UTC").timestamp() * 1000
    )
    scene = dict(scene_id="20261025T154451", time_utc="2026-10-25T15:44:51Z")
    for _ in range(2):
        result = emit.fetch_chips(scene, candidate()["unit"], tmp_path, ee=ee)
        assert result["status"] == "pending"
        assert result["message"] == (
            "not in Earth Engine yet (its EMIT collection ends 31 Aug 2026)"
        )
    assert ee.ImageCollection.return_value.filterDate.call_count == 2
    assert not list(tmp_path.rglob("*.json"))


def test_ee_failure_sanitizes_error(tmp_path):
    ee = Mock()
    ee.data.getAsset.side_effect = RuntimeError("private credential material")
    ee.ImageCollection.side_effect = RuntimeError("network")
    scene = dict(
        scene_id="20261025T154451_2629810_001", time_utc="2026-10-25T15:44:51Z"
    )
    result = emit.fetch_chips(scene, candidate()["unit"], tmp_path, ee=ee)
    assert result["status"] == "unavailable"
    assert "credential material" not in json.dumps(result)


def test_update_processes_next_nights_without_publish(tmp_path, monkeypatch):
    folder = tmp_path / "v5_prior"
    folder.mkdir()
    tag = "2026-10-01_2026-11-04"
    pd.DataFrame({"night": ["2026-11-04"]}).to_csv(
        folder / f"site_nights_{tag}.csv", index=False
    )
    (folder / f"overpasses_{tag}.csv").write_text("site_id\n")
    calls = []

    def backfill(**kwargs):
        calls.append(kwargs)
        out = kwargs["out_dir"]
        out.mkdir()
        t = f"{kwargs['start']}_{kwargs['end']}"
        pd.DataFrame({"night": [kwargs["end"]]}).to_csv(
            out / f"site_nights_{t}.csv", index=False
        )
        (out / f"overpasses_{t}.csv").write_text("site_id\n")
        (out / f"run_summary_{t}.json").write_text(
            json.dumps(dict(expected=1, complete=1))
        )

    monkeypatch.setattr(update.backfill, "main", backfill)
    build = Mock(return_value=tmp_path / "site")
    publish = Mock()
    monkeypatch.setattr(build_viewer, "main", build)
    monkeypatch.setattr(build_viewer, "publish_pages", publish)
    update.main(base=tmp_path, today=date(2026, 11, 9))
    assert calls[0]["start"] == "2026-11-05"
    assert calls[0]["end"] == "2026-11-06"
    assert not publish.called
    update.main(base=tmp_path, today=date(2026, 11, 9), publish=True)
    assert len(calls) == 1
    assert publish.call_count == 1


def test_update_retention_warning_and_expired_stop(tmp_path, monkeypatch):
    folder = tmp_path / "v5_prior"
    folder.mkdir()
    tag = "2026-08-01_2026-08-01"
    pd.DataFrame({"night": ["2026-08-01"]}).to_csv(
        folder / f"site_nights_{tag}.csv", index=False
    )
    (folder / f"overpasses_{tag}.csv").write_text("site_id\n")
    with pytest.warns(UserWarning, match="91 days"):
        with pytest.raises(ValueError, match="retention"):
            update.main(base=tmp_path, today=date(2026, 11, 9))


def test_small_source_long_absence_is_watch():
    n, p, c = history_fixture(days=100, outage=55)
    p["rh_mw"] /= 20
    c["group_rh_mw"] /= 20
    result = f.screen(n, p, c)
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["tier"] == "Watch"
    assert result["candidates"][0]["dark_qualified_nights"] == 45


def test_detection_under_cloud_ends_a_spell():
    n, p, c = history_fixture()
    original = c.iloc[0].copy()
    original["time_utc"] = "2026-10-15T00:00:00"
    original["granule"] = "recovery"
    original["cloud_mask_peak"] = 3
    c.loc[len(c)] = original
    result = f.screen(n, p, c)
    assert result["candidates"][0]["status"] == "ended"
    assert result["candidates"][0]["recovery"] == "2026-10-14"


def test_source_complex_merges_swapped_assignments_without_duplicate_card():
    n, p, c = history_fixture()
    c.loc[c.index % 2 == 0, "assigned_site"] = 3
    units = f.source_units(
        f.prepare_copies(c, "2026-08-01", "2026-10-19"), "2026-08-30"
    )
    assert len(units) == 1
    assert units[0]["site_ids"] == [1, 3]
    result = f.screen(n, p, c)
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["unit"]["site_ids"] == [1, 3]


def test_load_copies_keeps_distinct_groups_and_later_copy_wins(tmp_path):
    paths = []
    for index in (1, 2):
        folder = tmp_path / str(index)
        folder.mkdir()
        tag = f"2026-08-0{index}_2026-08-0{index}"
        rows = pd.DataFrame(
            dict(
                satellite=["SNPP", "SNPP"],
                granule=["g", "g"],
                group_id=[0, 1],
                peak_row=[4, 4],
                peak_col=[5, 6],
                group_rh_mw=[index, index + 10],
            )
        )
        rows.to_csv(folder / f"copies_{tag}.csv", index=False)
        paths.append((folder, tag))
    result = build_viewer.load_copies(paths)
    assert len(result) == 2
    assert result.group_rh_mw.tolist() == [2, 12]


def test_successful_ee_chips_are_cached_and_copied(tmp_path, monkeypatch):
    import io
    from PIL import Image

    png = io.BytesIO()
    Image.new("RGB", (512, 512)).save(png, format="PNG")
    session = Mock()
    session.get.return_value = SimpleNamespace(
        content=png.getvalue(), raise_for_status=lambda: None
    )
    ee = Mock()
    ee.data.getAsset.return_value = {"id": "scene"}
    ee.Geometry.Point.return_value.transform.return_value.coordinates.return_value.getInfo.return_value = [
        500000,
        2000000,
    ]
    rad = ee.Image.return_value
    cont = rad.select.return_value.reduce.return_value
    raw = cont.subtract.return_value.divide.return_value.updateMask.return_value.rename.return_value
    monkeypatch.setattr(
        emit, "index_coverage", lambda *a: emit._coverage(1, 85, "rendered_index_mask")
    )
    raw.reduceRegion.return_value.getInfo.return_value = dict(
        ch4_depth_p2=0.0,
        ch4_depth_p16=0.1,
        ch4_depth_p50=0.2,
        ch4_depth_p84=0.3,
        ch4_depth_p98=0.4,
    )
    # True colour masks EMIT fill values (-9999) before its stretch statistics.
    rgb = rad.select.return_value.updateMask.return_value
    rgb.reduceRegion.return_value.getInfo.return_value = {
        f"radiance_{band}_p{pct}": pct / 10
        for band in (35, 23, 11)
        for pct in (2, 50, 98)
    }
    scene = dict(
        scene_id="20260825T154451_2623710_001", time_utc="2026-08-25T15:44:51Z"
    )
    first = emit.fetch_chips(
        scene, candidate()["unit"], tmp_path, ee=ee, session=session
    )
    assert first["status"] == "available"
    assert first["recipe_version"] == emit.CHIP_RECIPE
    assert first["source_coverage"] == "covered"
    ee.Image.constant.assert_called_once_with([128, 128, 128])
    assert first["normalization"]["robust_sigma"] == pytest.approx(0.1)
    assert first["recipe"]["rgb_bands"] == ["radiance_35", "radiance_23", "radiance_11"]
    assert first["recipe"]["rgb_percentiles"] == [2, 98]
    assert first["recipe"]["rgb_radius_m"] == 20000
    assert first["recipe"]["rgb_scale_m"] == 120
    assert first["recipe"]["rgb_min"] == 0.2
    assert first["recipe"]["rgb_max"] == 9.8
    assert first["recipe"]["rgb_gamma"] == 1.8
    assert first["recipe"]["index_gamma"] == 0.65
    ee.Geometry.Point.return_value.buffer.assert_any_call(20000)
    raw.subtract.return_value.divide.return_value.subtract.return_value.divide.return_value.clamp.return_value.pow.assert_called_once_with(
        0.65
    )
    assert session.get.call_count == 2
    again = emit.fetch_chips(
        scene, candidate()["unit"], tmp_path, ee=ee, session=session
    )
    assert again == first
    assert session.get.call_count == 2
    from pathlib import Path

    assert Image.open(Path(first["cache_folder"]) / "ch4.png").size == (512, 512)
    stored = json.dumps(
        json.loads((Path(first["cache_folder"]) / "metadata.json").read_text())
    )
    assert "getThumbURL" not in stored


def test_update_resumes_done_but_incomplete_manifest_before_rebuild(
    tmp_path, monkeypatch
):
    folder = tmp_path / "v5_update_2026-11-01_2026-11-06"
    folder.mkdir()
    tag = "2026-11-01_2026-11-06"
    pd.DataFrame({"night": ["2026-11-06"]}).to_csv(
        folder / f"site_nights_{tag}.csv", index=False
    )
    (folder / f"overpasses_{tag}.csv").write_text("site_id\n")
    (folder / "status.json").write_text(json.dumps(dict(phase="done")))
    (folder / f"run_summary_{tag}.json").write_text(
        json.dumps(dict(expected=2, complete=1))
    )
    backfill = Mock()
    monkeypatch.setattr(update.backfill, "main", backfill)
    build = Mock()
    monkeypatch.setattr(build_viewer, "main", build)
    with pytest.raises(RuntimeError, match="incomplete"):
        update.main(base=tmp_path, today=date(2026, 11, 9))
    assert backfill.call_count == 1
    assert not build.called


def test_heat_aperture_excludes_neighbour_but_wider_source_veto_remains():
    n, p, c = history_fixture(turndown=True)
    other = c.copy()
    other["assigned_site"] = 3
    other["group_id"] = 3
    other["centroid_lat"] += 0.012  # 1.33 km: veto radius, outside heat radius
    other["group_rh_mw"] = 30
    states = fixture_states(n, p, pd.concat([c, other], ignore_index=True))
    assert any(s["state"] == "reduced heat" and s["median_mw"] < 1 for s in states)


def test_load_copies_replaces_whole_granule_snapshot(tmp_path):
    segments = []
    for i in (1, 2):
        folder = tmp_path / str(i)
        folder.mkdir()
        tag = str(i)
        pd.DataFrame(
            dict(
                satellite=["SNPP"],
                granule=["g"],
                group_id=[i],
                peak_row=[i],
                peak_col=[i],
            )
        ).to_csv(folder / f"copies_{tag}.csv", index=False)
        segments.append((folder, tag))
    result = build_viewer.load_copies(segments)
    assert result.group_id.tolist() == [2]
