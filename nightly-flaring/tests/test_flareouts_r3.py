"""Round-two review regressions: snapshots, fixed scans and HITL fidelity."""

import json
from datetime import date, datetime
from unittest.mock import Mock

import pandas as pd
import pytest

import build_viewer as b
import emit
import flareouts as f
import pipeline
import update
from test_flareouts import candidate, fake_response, history_fixture
from test_flareouts_r2 import write_segment_stub


def test_incomplete_adjacent_snapshot_preserves_earlier_night_recovery(tmp_path):
    n, p, c = history_fixture(days=88, outage=43)
    p["temp_k"] = 1500.0
    recovery_time = "2026-09-30T00:00:00"
    key = ("SNPP", "g59_SNPP")
    recovery = c.iloc[0].copy()
    recovery["time_utc"] = recovery_time
    recovery["granule"] = key[1]
    recovery["group_rh_mw"] = 12.0
    c.loc[len(c)] = recovery
    c["peak_row"] = c["peak_col"] = 1
    p.loc[p.time_utc == recovery_time, ["detected", "rh_mw"]] = [True, 12.0]
    sites = n.drop_duplicates("site_id")[["site_id", "lat", "lon"]]
    segments = []
    for i, (start, end) in enumerate(
        [("2026-08-01", "2026-09-29"), ("2026-09-30", "2026-10-27")]
    ):
        tag = f"{start}_{end}"
        folder = tmp_path / f"v5_{i}"
        folder.mkdir()
        night = (pd.to_datetime(p.time_utc) - pd.Timedelta(hours=9)).dt.strftime(
            "%Y-%m-%d"
        )
        passes = p[night.between(start, end)]
        night = (pd.to_datetime(c.time_utc) - pd.Timedelta(hours=9)).dt.strftime(
            "%Y-%m-%d"
        )
        copies = c[night.between(start, end)]
        manifest = passes[["satellite", "granule"]].drop_duplicates().copy()
        manifest["complete"] = True
        if i == 1:
            # Accepted missing UTC boundary job: its earlier local night has
            # no unknown-exposure row in this newer segment.
            manifest.loc[len(manifest)] = [*key, False]
        pipeline.site_nights(
            passes, sites, date.fromisoformat(start), date.fromisoformat(end)
        ).to_csv(folder / f"site_nights_{tag}.csv", index=False)
        passes.to_csv(folder / f"overpasses_{tag}.csv", index=False)
        copies.to_csv(folder / f"copies_{tag}.csv", index=False)
        manifest.to_csv(folder / f"jobs_{tag}.csv", index=False)
        segments.append((folder, tag))
    assert b.granule_owners(segments)[key] == 0
    nights, passes = b.load_segments(segments)
    copies = b.load_copies(segments)
    assert passes.loc[passes.time_utc == recovery_time, "detected"].tolist() == [True]
    assert recovery_time in set(copies.time_utc)
    cards = f.screen(nights, passes, copies)["candidates"]
    assert cards and all(card["status"] == "ended" for card in cards)
    assert cards[0]["recovery"] == "2026-09-29"


def test_incomplete_only_snapshot_supplies_no_exposure_or_positive(tmp_path):
    tag = "2026-11-01_2026-11-01"
    pd.DataFrame(dict(site_id=[1], night=["2026-11-01"])).to_csv(
        tmp_path / f"site_nights_{tag}.csv", index=False
    )
    pd.DataFrame(
        dict(satellite=["SNPP"], granule=["g"], site_id=[1], time_utc=["2026-11-02"])
    ).to_csv(tmp_path / f"overpasses_{tag}.csv", index=False)
    pd.DataFrame(
        dict(
            satellite=["SNPP"], granule=["g"], group_id=[1], peak_row=[1], peak_col=[1]
        )
    ).to_csv(tmp_path / f"copies_{tag}.csv", index=False)
    pd.DataFrame(dict(satellite=["SNPP"], granule=["g"], complete=[False])).to_csv(
        tmp_path / f"jobs_{tag}.csv", index=False
    )
    segments = [(tmp_path, tag)]
    assert b.load_segments(segments)[1].empty
    assert b.load_copies(segments).empty


def test_older_record_length_does_not_enlarge_30_night_family_or_card_score():
    # Two sources provide enough calibration transitions for the same 0.59
    # persistence floor with either archive length. Only scan scope can drift.
    n, p, c = history_fixture(days=180, outage=174, neighbour=True)
    full = f.screen(n, p, c)
    first = sorted(n.night.unique())[-88]
    short = f.screen(n[n.night >= first], p, c)
    for result in (full, short):
        cal = result["calibration"]
        score = result["candidates"][0]["score"]
        assert score["evaluation_nights"] == 30
        assert score["scan_window_start"] == sorted(n.night.unique())[-30]
        assert score["scan_window_end"] == n.night.max()
        total = sum(
            count
            for day, count in cal["eligible_units_by_night"].items()
            if score["scan_window_start"] <= day <= score["scan_window_end"]
        )
        assert score["family_unit_nights"] == max(score["family_floor"], total)
    assert full["calibration"]["miss_persistence_used"] == 0.59
    assert short["calibration"]["miss_persistence_used"] == 0.59
    assert full["candidates"][0]["score"] == short["candidates"][0]["score"]


def test_startup_uses_only_available_evaluation_nights():
    data = f.screen(*history_fixture(days=40, outage=34))
    score = data["candidates"][0]["score"]
    assert score["evaluation_nights"] == 10
    assert score["scan_window_nights"] == 30
    assert score["family_floor"] == 10


def test_end_to_end_isolated_dark_spell_reaches_high():
    n, p, c = history_fixture(days=80, outage=55)
    dn, dp, dc = history_fixture(days=80, outage=80)
    dn["site_id"] = dp["site_id"] = dc["assigned_site"] = 2
    dn["lon"] += 0.07
    dc["centroid_lon"] += 0.07
    dc["group_id"] = 2
    result = f.screen(
        *(pd.concat(pair, ignore_index=True) for pair in ((n, dn), (p, dp), (c, dc)))
    )
    card = result["candidates"][0]
    assert card["unit"]["site_ids"] == [1]
    assert card["dark_qualified_nights"] == 25
    assert card["tier"] == "High"
    assert card["score"]["budget_multiple"] < 1
    assert card["context"]["label"] == "isolated"
    assert card["context"]["neighbours"][0]["measured_nights"] == 25


def test_accept_missing_outside_segment_does_not_block_next_month(
    tmp_path, monkeypatch
):
    tag = "2026-11-01_2026-11-06"
    summary = dict(expected=1, complete=0, missing=[["SNPP", "east"]])
    folder = write_segment_stub(tmp_path, tag, summary)
    pipeline.persistent_jobs(
        [
            dict(
                satellite="SNPP",
                gid="east",
                start=datetime(2026, 11, 7, 6),
                sites=[[1, 0, 150]],
                keys={},
            )
        ],
        folder,
        tag,
    )
    pd.DataFrame(columns=["site_id", "night", "satellite", "granule"]).to_csv(
        folder / f"missing_observations_{tag}.csv", index=False
    )
    pd.DataFrame(dict(satellite=["SNPP"], granule=["east"], complete=[False])).to_csv(
        folder / f"jobs_{tag}.csv", index=False
    )
    calls = []

    def run(**kw):
        calls.append((kw["start"], kw["end"]))
        if kw["out_dir"] != folder:
            write_segment_stub(
                tmp_path, f"{kw['start']}_{kw['end']}", dict(expected=1, complete=1)
            )

    monkeypatch.setattr(update.backfill, "main", run)
    monkeypatch.setattr(b, "main", Mock(return_value=tmp_path / "site"))
    with pytest.warns(UserWarning, match="Accepted 1"):
        update.main(base=tmp_path, today=date(2026, 11, 9), accept_missing=True)
    update.main(base=tmp_path, today=date(2026, 12, 9))
    assert calls == [("2026-11-01", "2026-11-06"), ("2026-11-07", "2026-12-06")]


def test_unrelated_missing_exposure_cannot_hide_an_unrecorded_deficit(tmp_path):
    tag = "2026-11-01_2026-11-06"
    jobs = [
        dict(
            satellite="SNPP",
            gid=gid,
            start=datetime(2026, 11, 2),
            sites=[[1, 24, 53]],
            keys={},
        )
        for gid in ("recorded", "unrecorded")
    ]
    pipeline.persistent_jobs(jobs, tmp_path, tag)
    pd.DataFrame(
        dict(
            site_id=[1], night=["2026-11-01"], satellite=["SNPP"], granule=["recorded"]
        )
    ).to_csv(tmp_path / f"missing_observations_{tag}.csv", index=False)
    with pytest.raises(RuntimeError, match="unknown-exposure"):
        update.check_missing_exposures(
            tmp_path, tag, [["SNPP", "recorded"], ["SNPP", "unrecorded"]]
        )


def test_expected_snapshot_compresses_full_metadata_without_scope_loss(tmp_path):
    tag = "2026-11-01_2026-11-30"
    jobs = [
        dict(
            satellite="SNPP",
            gid=f"g{i}",
            start=datetime(2026, 11, 2),
            keys=dict(M10="public/object"),
            sites=[[sid, 24 + sid / 100, 53] for sid in range(100)],
        )
        for i in range(300)
    ]
    pipeline.persistent_jobs(jobs, tmp_path, tag)
    path = pipeline.expected_jobs_path(tmp_path, tag)
    assert path.suffix == ".gz"
    assert path.stat().st_size < len(json.dumps(jobs, default=str).encode()) / 10
    assert pipeline.read_expected_jobs(tmp_path, tag) == jobs
    assert pipeline.persistent_jobs(jobs[:1], tmp_path, tag) == jobs


@pytest.mark.parametrize("cutoff", ["2026-09-18", "2026-10-01"])
def test_pending_message_tracks_measured_ee_horizon(tmp_path, cutoff):
    ee = Mock()
    ee.ImageCollection.return_value.filterDate.return_value.filterBounds.return_value.aggregate_array.return_value.getInfo.return_value = []
    ee.ImageCollection.return_value.aggregate_max.return_value.getInfo.return_value = (
        pd.Timestamp(cutoff, tz="UTC").timestamp() * 1000
    )
    scene = dict(scene_id="20261025T154451", time_utc="2026-10-25T15:44:51Z")
    result = emit.fetch_chips(scene, candidate()["unit"], tmp_path, ee=ee)
    assert result["status"] == "pending"
    assert result["collection_latest_utc"].startswith(cutoff)
    assert "31 Aug" not in result["message"]
    assert pd.Timestamp(cutoff).strftime("%-d %b %Y") in result["message"]


def test_missing_ee_horizon_keeps_availability_unknown(tmp_path):
    ee = Mock()
    ee.data.getAsset.return_value = None
    ee.ImageCollection.return_value.aggregate_max.return_value.getInfo.return_value = (
        None
    )
    scene = dict(scene_id="20261025T154451_123_001", time_utc="2026-10-25T15:44:51Z")
    result = emit.fetch_chips(scene, candidate()["unit"], tmp_path, ee=ee)
    assert result["status"] == "unavailable"
    assert result["collection_latest_utc"] is None
    assert "unknown" in result["message"]


def test_failed_ee_connection_exports_unknown_availability(tmp_path, monkeypatch):
    session = Mock()
    session.get.return_value = fake_response(
        [
            dict(
                producer_granule_id="EMIT_L1B_RAD_002_20261025T154451",
                time_start="2026-10-25T15:44:51Z",
            )
        ]
    )
    monkeypatch.setattr(emit, "_ee_client", Mock(side_effect=RuntimeError("offline")))
    result = emit.enrich(
        dict(as_of="2026-11-04", candidates=[candidate()]),
        cache_dir=tmp_path,
        session=session,
    )
    imagery = result["candidates"][0]["emit"]["scenes"][0]["imagery"]
    assert imagery["status"] == "unavailable"
    assert imagery["collection_latest_utc"] is None
    assert "availability is unknown" in imagery["message"]


@pytest.mark.parametrize(
    "bad", [None, "physical", "balance", "missing", "flat", "error"]
)
def test_truecolor_shared_stretch_and_hitl_reflectance_fallback(bad):
    ee, pt = Mock(), Mock()
    rfl, rad = Mock(), Mock()

    def stats(prefix):
        return {
            f"{prefix}_{band}_p{pct}": value
            for band, offset in ((35, 0), (23, 0.1), (11, 0.2))
            for pct, value in (
                (2, 0.05 + offset),
                (50, 0.2 + offset),
                (98, 0.6 + offset),
            )
        }

    reflectance = stats("reflectance")
    if bad == "physical":
        reflectance["reflectance_35_p98"] = 1.6
    elif bad == "balance":
        reflectance["reflectance_35_p50"] = 3.0
    elif bad == "missing":
        reflectance.pop("reflectance_35_p50")
    elif bad == "flat":
        reflectance = dict.fromkeys(reflectance, 0.1)
    if bad == "error":
        rfl.reduceRegion.side_effect = RuntimeError("unavailable")
    else:
        rfl.reduceRegion.return_value.getInfo.return_value = reflectance
    rad.reduceRegion.return_value.getInfo.return_value = stats("radiance")
    # Fill values (-9999) are masked first; the masked image is what is returned.
    rfl_raw, rad_raw = Mock(), Mock()
    rfl_raw.updateMask.return_value = rfl
    rad_raw.updateMask.return_value = rad
    ee.Image.side_effect = lambda asset: Mock(
        select=Mock(return_value=rfl_raw if "/L2A/" in asset else rad_raw)
    )
    image, vis, recipe = emit.emit_truecolor("acquisition", pt, ee)
    assert image is (rad if bad else rfl)
    (rad_raw if bad else rfl_raw).updateMask.assert_called_once()
    ee.Reducer.min.assert_called()
    ee.Reducer.max.assert_called()
    assert vis == dict(min=0.05, max=0.8, gamma=1.8)
    assert recipe["rgb_bands"] == [
        f"{'radiance' if bad else 'reflectance'}_{band}" for band in (35, 23, 11)
    ]
    ee.Reducer.percentile.assert_called_with([2, 50, 98])
    pt.buffer.assert_called_with(20000)
    assert image.reduceRegion.call_args.kwargs["scale"] == 120
