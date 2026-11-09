"""Tests for the website data build (build_viewer.py)."""

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
        "2026-08-01", "2026-08-02", "2026-08-03", "2026-08-04"
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
            [["NOAA-20", "g1", 1, "2026-08-02T01:00:00", 0.0],
             ["NOAA-20", "g2", 1, "2026-08-03T01:00:00", 8.0]],
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
