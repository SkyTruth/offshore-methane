"""Tests for the website data build (build_viewer.py)."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build_viewer  # noqa: E402


def _nights(rows):
    cols = ["site_id", "night", "status", "rh_mw", "n_clear"]
    return pd.DataFrame(rows, columns=cols)


def test_encode_site_nights_codes_values_and_gaps():
    nights = ["2026-08-01", "2026-08-02", "2026-08-03", "2026-08-04", "2026-08-05"]
    g = _nights(
        [
            (1, "2026-08-01", "flaring", 3.25, 2),
            (1, "2026-08-02", "clear_no_flare", 0.0, 12),
            (1, "2026-08-03", "flaring_cloud", np.nan, 0),
            (1, "2026-08-05", "flaring_cloud", 1.5, 1),
        ]
    )
    st, rv, cp = build_viewer.encode_site_nights(g, nights)
    assert st == "FCDND"  # the missing night is not observed
    assert rv == [3.25, None, 1.5]  # one value per F/D night, in order
    assert cp == "2c001"  # clear passes as base-36 digits


def test_night_series_fills_missing_nights():
    df = _nights(
        [(1, "2026-08-01", "cloudy", np.nan, 0), (1, "2026-08-04", "cloudy", np.nan, 0)]
    )
    assert build_viewer.night_series(df) == [
        "2026-08-01",
        "2026-08-02",
        "2026-08-03",
        "2026-08-04",
    ]


def _write_segment(folder: Path, tag: str, nights: pd.DataFrame, over: pd.DataFrame):
    folder.mkdir()
    nights.to_csv(folder / f"site_nights_{tag}.csv", index=False)
    over.to_csv(folder / f"overpasses_{tag}.csv", index=False)


def test_segments_join_and_later_segment_wins(tmp_path):
    over_cols = ["satellite", "granule", "site_id", "time_utc", "rh_mw"]
    _write_segment(
        tmp_path / "v5_later",
        "2026-08-02_2026-08-03",
        _nights(
            [
                (1, "2026-08-02", "flaring", 9.0, 1),
                (1, "2026-08-03", "cloudy", np.nan, 0),
            ]
        ),
        pd.DataFrame(
            [["NOAA-20", "g2", 1, "2026-08-03T01:00:00", 9.0]], columns=over_cols
        ),
    )
    _write_segment(
        tmp_path / "v5_early",
        "2026-08-01_2026-08-02",
        _nights(
            [
                (1, "2026-08-01", "clear_no_flare", 0.0, 2),
                (1, "2026-08-02", "cloudy", np.nan, 0),
            ]
        ),
        pd.DataFrame(
            [
                ["NOAA-20", "g1", 1, "2026-08-02T01:00:00", 0.0],
                ["NOAA-20", "g2", 1, "2026-08-03T01:00:00", 8.0],
            ],
            columns=over_cols,
        ),
    )
    (tmp_path / "dev_v5").mkdir()  # outside the pattern: ignored
    segments = build_viewer.find_segments(tmp_path)
    assert [f.name for f, _ in segments] == ["v5_early", "v5_later"]
    nights, over = build_viewer.load_segments(segments)
    by_night = nights.set_index("night")["status"].to_dict()
    assert by_night == {
        "2026-08-01": "clear_no_flare",
        "2026-08-02": "flaring",
        "2026-08-03": "cloudy",
    }
    # granule g2 was read by both segments; one row is kept, from the later one
    assert sorted(over["granule"]) == ["g1", "g2"]
    assert over.loc[over["granule"] == "g2", "rh_mw"].item() == 9.0


def _page_fixture():
    nights = pd.date_range("2025-12-28", "2026-01-05").strftime("%Y-%m-%d").tolist()
    rows = []
    for sid, st, values, cp, region in [
        (1, "FCDNKCFCC", [9.0, None, 3.0], "210002100", "A"),
        (2, "CCFDCCCCN", [6.0, 0.0], "001100000", "A"),
        (3, "NNCCCCFCC", [2.0], "000000100", "B"),
    ]:
        rows.append(
            [sid, 0, 0, 1, st, values, None, 1, region, None, None, None, None, [], cp]
        )
    return {"meta": {}, "nights": nights, "sites": rows}


def test_chunk_roundtrip_year_boundaries_and_stable_hashes(tmp_path):
    data = _page_fixture()
    page = build_viewer.write_night_chunks(data, tmp_path)
    assert len(page["meta"]["chunks"]) == 2
    joined = {r[0]: ["", [], ""] for r in data["sites"]}
    for name in page["meta"]["chunks"]:
        chunk = json.loads((tmp_path / name).read_text())
        for sid, st, rv in chunk["sites"]:
            joined[sid][0] += st
            joined[sid][1].extend(rv)
    for name in page["meta"]["clearpasses"]:
        chunk = json.loads((tmp_path / name).read_text())
        for sid, cp in chunk["sites"]:
            joined[sid][2] += cp
    for row in data["sites"]:
        assert joined[row[0]] == [row[4], row[5], row[14]]
        heat, _ = build_viewer.decode_nights(row)
        codes = {v: k for k, v in build_viewer.STATUS_CODE.items()}
        frame = _nights(
            [
                (row[0], night, codes[ch], value, int(cp, 36))
                for night, ch, value, cp in zip(data["nights"], row[4], heat, row[14])
            ]
        )
        assert tuple(joined[row[0]]) == build_viewer.encode_site_nights(
            frame, data["nights"]
        )
    unchanged = page["meta"]["chunks"][0]
    data["sites"][0][4] = data["sites"][0][4][:-1] + "K"
    changed = build_viewer.write_night_chunks(data, tmp_path)["meta"]["chunks"]
    assert changed[0] == unchanged
    assert changed[1] != page["meta"]["chunks"][1]
    assert page["sites"][0][4:6] == [None, None]
    earlier = pd.date_range("2024-12-28", "2025-12-27").strftime("%Y-%m-%d").tolist()
    data["nights"] = earlier + data["nights"]
    for row in data["sites"]:
        row[4] = "N" * len(earlier) + row[4]
        row[14] = "0" * len(earlier) + row[14]
    extended_page = build_viewer.write_night_chunks(data, tmp_path)
    extended = extended_page["meta"]["chunks"]
    assert extended_page["meta"]["clearpasses"][-1] == page["meta"]["clearpasses"][-1]
    assert extended[-1] == changed[-1]  # an archive backfill keeps 2026's file


def test_bin_totals_equal_direct_pandas_site_means():
    # Expected values come from raw frames, independently of decoding/bin helpers.
    dates = pd.date_range("2023-12-28", "2025-03-05")
    rng = np.random.default_rng(66)
    records, rows = [], []
    statuses = ["flaring", "flaring_cloud", "clear_no_flare", "cloudy", "not_observed"]
    for sid in range(1, 5):
        region = "A" if sid < 3 else "B"
        raw = []
        for date in dates:
            status = rng.choice(statuses)
            value = (
                round(float(rng.uniform(0, 10)), 3)
                if status == "flaring"
                else 0.0
                if status == "clear_no_flare"
                else rng.choice([0.0, np.nan])
                if status == "flaring_cloud"
                else np.nan
            )
            raw.append((sid, date.strftime("%Y-%m-%d"), status, value, 1))
            records.append((sid, date, region, status, value))
        st, rv, cp = build_viewer.encode_site_nights(
            _nights(raw), dates.strftime("%Y-%m-%d").tolist()
        )
        rows.append(
            [sid, 0, 0, 1, st, rv, None, 1, region, None, None, None, None, [], cp]
        )
    data = {"nights": dates.strftime("%Y-%m-%d").tolist(), "sites": rows}
    frame = pd.DataFrame(records, columns=["site", "date", "region", "status", "heat"])
    frame["lit"] = frame.status.eq("flaring")
    frame["observed"] = frame.status.ne("not_observed")
    frame["clear"] = frame.heat.notna()
    frame["clear_night"] = frame.status.isin(["flaring", "clear_no_flare"])
    totals = build_viewer.bin_totals(data)
    for unit, freq in [
        ("night", "D"),
        ("week", "W-SUN"),
        ("month", "M"),
        ("quarter", "Q"),
        ("year", "Y"),
    ]:
        independent = frame.assign(period=frame.date.dt.to_period(freq))
        expected, regions = [], {"A": [], "B": []}
        for _, group in independent.groupby("period", sort=True):
            site_means = group.groupby("site").heat.mean().fillna(0)
            expected.append(
                [
                    site_means.sum(),
                    group.groupby("site").lit.any().sum(),
                    group.groupby("date").clear.sum().mean(),
                    group.groupby("date")["observed"].sum().mean(),
                    group.groupby("site").clear_night.any().sum(),
                ]
            )
            for region in regions:
                regions[region].append(
                    group[group.region.eq(region)]
                    .groupby("site")
                    .heat.mean()
                    .fillna(0)
                    .sum()
                )
        assert np.allclose(totals[unit]["network"], expected, atol=5.1e-7, rtol=0)
        assert (
            totals[unit]["regions"] == {}
            if unit == "night"
            else all(
                np.allclose(totals[unit]["regions"][r], v, atol=5.1e-7, rtol=0)
                for r, v in regions.items()
            )
        )
    assert len(totals["night"]["network"]) == len(dates)
    assert pd.Timestamp("2024-02-29") in dates


def test_chunks_ignore_first_lit_elsewhere_and_archive_only_sites(tmp_path):
    dates = ["2024-12-31", "2025-01-01", "2026-01-01"]
    data = {
        "meta": {},
        "nights": dates,
        "sites": [
            [1, 0, 0, 1, "CCC", [], None, 0, "A", None, None, None, None, [], "111"]
        ],
    }
    first = build_viewer.write_night_chunks(data, tmp_path)["meta"]["chunks"]
    data["sites"][0][4:6] = ["CCF", [2.0]]
    second = build_viewer.write_night_chunks(data, tmp_path)["meta"]["chunks"]
    assert first[:2] == second[:2]  # first-ever flare in the newest year
    assert first[2] != second[2]
    data["sites"].append(
        [2, 0, 0, 1, "FNN", [3.0], None, 1, "A", None, None, None, None, [], "100"]
    )
    third = build_viewer.write_night_chunks(data, tmp_path)["meta"]["chunks"]
    assert third[0] != second[0]
    assert third[1:] == second[1:]  # new site observed only in archive year
    page = build_viewer.write_night_chunks(data, tmp_path)
    assert "totals" in page and "totals_gzip" not in page
    assert all(
        unit in page["totals"] for unit in ["night", "week", "month", "quarter", "year"]
    )


def test_publish_retains_both_chunk_kinds_one_build_in_real_git_repo(
    monkeypatch, tmp_path
):
    """Exercise ls-remote/fetch/FETCH_HEAD with a temporary bare remote only."""
    repo, remote, site = [tmp_path / name for name in ("repo", "remote.git", "site")]
    repo.mkdir()
    site.mkdir()

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(repo), *args], text=True
        ).strip()

    git("init", "-q")
    git("config", "user.name", "Retention test")
    git("config", "user.email", "retention@example.invalid")
    git("commit", "--allow-empty", "-qm", "fixture")
    git("init", "--bare", "-q", str(remote))
    git("remote", "add", "fixture", str(remote))
    monkeypatch.setattr(build_viewer, "HERE", repo)
    previous = set()
    for version in (1, 2, 3):
        chunks = {
            f"data/{kind}-2026.{version:08x}.json" for kind in ("nights", "clearpasses")
        }
        for file in site.rglob("*.json"):
            file.unlink()
        for name in chunks:
            file = site / name
            file.parent.mkdir(exist_ok=True)
            file.write_text(json.dumps({"version": version, "file": name}))
        meta = {
            "chunks": sorted(n for n in chunks if "/nights-" in n),
            "clearpasses": sorted(n for n in chunks if "/clearpasses-" in n),
        }
        html = (
            '<script type="application/json" id="flaring-data">'
            + json.dumps({"meta": meta})
            + "</script>"
        )
        (site / "index.html").write_text(html)
        commit = build_viewer.publish_pages(site, branch="pages", remote="fixture")
        actual = set(git("ls-tree", "-r", "--name-only", commit).splitlines())
        assert actual == chunks | previous | {"index.html", ".nojekyll"}
        assert git("show", f"{commit}:index.html") == html
        assert git("rev-list", "--count", commit) == "1"  # parentless publish
        assert git("branch", "--show-current") != "pages"
        assert git("status", "--porcelain") == ""
        previous = chunks


def test_deferred_clearpasses_hashes_are_year_local_and_separate(tmp_path):
    data = _page_fixture()
    first = build_viewer.write_night_chunks(data, tmp_path)
    data["sites"][0][14] = data["sites"][0][14][:-1] + "3"
    second = build_viewer.write_night_chunks(data, tmp_path)
    assert first["meta"]["chunks"] == second["meta"]["chunks"]
    assert first["meta"]["clearpasses"][0] == second["meta"]["clearpasses"][0]
    assert first["meta"]["clearpasses"][1] != second["meta"]["clearpasses"][1]
    import dev_long_record

    (tmp_path / "index.html").write_text(
        '<script type="application/json" id="flaring-data">'
        + json.dumps(second)
        + "</script>"
    )
    assert dev_long_record.read_page(tmp_path / "index.html")["sites"] == data["sites"]


def test_flareout_page_window_and_record_monthly_keep_original():
    series = [
        {
            "night": d.strftime("%Y-%m-%d"),
            "heat_mw": 4.0,
            "qualified": True,
            "lit": True,
        }
        for d in pd.date_range("2025-01-01", "2026-11-04")
    ]
    source = {
        "candidates": [
            {
                "start": "2026-10-01",
                "end": "2026-10-10",
                "nightly_series": series,
                "context": {"neighbours": [{"series": series}]},
            }
        ]
    }
    card = build_viewer.window_flareouts(source)["candidates"][0]
    assert card["nightly_series"][0]["night"] == "2026-07-03"
    assert card["nightly_series"][-1]["night"] == "2026-11-04"
    assert card["context"]["neighbours"][0]["series"] == card["nightly_series"]
    assert card["record_monthly"][0] == ["2025-01", 4.0, 31, 31]
    assert len(source["candidates"][0]["nightly_series"]) == len(series)


def test_synthetic_calendar_gaps_and_real_year_unchanged():
    import dev_long_record

    source = _page_fixture()
    source["meta"] = {"start": source["nights"][0]}
    data = dev_long_record.synthetic_data(source, 750)
    assert data == dev_long_record.synthetic_data(source, 750)
    assert len(data["nights"]) == 750
    assert data["meta"]["synthetic"]
    for old, row in zip(source["sites"], data["sites"]):
        for i, night in enumerate(source["nights"]):
            if night.startswith("2026"):
                assert row[4][data["nights"].index(night)] == old[4][i]
        assert row[4][data["nights"].index("2024-11-10")] == "N"
