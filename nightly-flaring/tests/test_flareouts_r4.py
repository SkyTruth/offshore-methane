"""Per-spell scan families must retain the tested start and historical scope."""

import pandas as pd
import pytest

import flareouts as f
from test_flareouts import history_fixture


def remove_target_detections(passes, copies, start, end):
    """Leave clear exposures intact while injecting one target-only spell."""
    passes = passes.copy()
    night = (
        pd.to_datetime(passes.time_utc) + pd.to_timedelta(53 / 15 - 12, "h")
    ).dt.strftime("%Y-%m-%d")
    missing = passes.site_id.eq(1) & night.between(start, end)
    passes.loc[missing, ["detected", "rh_mw"]] = [False, 0]
    night = (
        pd.to_datetime(copies.time_utc)
        + pd.to_timedelta(copies.centroid_lon / 15 - 12, "h")
    ).dt.strftime("%Y-%m-%d")
    copies = copies[~(copies.assigned_site.eq(1) & night.between(start, end))]
    return passes, copies


def extend_calendar(nights, end):
    """Extend site-night dates without adding a pass or a source copy."""
    calendar = pd.MultiIndex.from_product(
        [nights.site_id.unique(), pd.date_range(nights.night.min(), end)],
        names=["site_id", "night"],
    ).to_frame(index=False)
    calendar["night"] = calendar.night.dt.strftime("%Y-%m-%d")
    return calendar.merge(
        nights.drop_duplicates("site_id")[["site_id", "lat", "lon"]], on="site_id"
    )


def target(result):
    return next(c for c in result["candidates"] if c["unit"]["site_ids"] == [1])


def test_no_new_observations_cannot_promote_ended_spell_in_empty_later_scan():
    # Sol N1: the routine 25 September miss joins six removed detection nights;
    # the measured recovery is 2 October. Reuse exactly the same pass/copy tables.
    n, p, c = history_fixture(days=70, outage=70, neighbour=True)
    p, c = remove_target_detections(p, c, "2026-09-26", "2026-10-01")
    before = f.screen(n, p, c)
    after = f.screen(extend_calendar(n, "2026-12-18"), p, c)
    old, new = target(before), target(after)
    assert old["tier"] == new["tier"] == "Mid"
    assert old["start"] == new["start"] == "2026-09-25"
    assert old["end"] == new["end"] == "2026-10-01"
    assert old["dark_qualified_nights"] == new["dark_qualified_nights"] == 7
    assert old["baseline"] == new["baseline"]
    assert old["pass_list"] == new["pass_list"]
    assert old["context"]["neighbours"][0]["measured_nights"] == 7
    assert new["context"]["neighbours"][0]["measured_nights"] == 7
    assert before["calibration"]["transitions"] == after["calibration"]["transitions"]
    assert before["calibration"]["miss_persistence_used"] == 0.59
    assert after["calibration"]["miss_persistence_used"] == 0.59
    assert old["score"] == new["score"]
    assert old["score"]["scan_window_start"] == "2026-09-02"
    assert old["score"]["scan_window_end"] == "2026-10-01"
    assert old["score"]["budget_multiple"] > 1
    assert not any(
        count
        for night, count in after["calibration"]["eligible_units_by_night"].items()
        if night >= "2026-11-19"
    )


def test_ended_spell_family_and_score_survive_another_month_of_observations():
    n, p, c = history_fixture(days=100, outage=100, neighbour=True)
    p, c = remove_target_detections(p, c, "2026-09-26", "2026-10-01")
    old = target(f.screen(n[n.night <= "2026-10-09"], p, c))
    new = target(f.screen(n, p, c))
    assert old["status"] == new["status"] == "ended"
    assert old["tier"] == new["tier"] == "Mid"
    assert old["score"] == new["score"]


def test_long_ongoing_spell_keeps_its_first_30_night_family():
    n, p, c = history_fixture(days=110, outage=55)
    scores = []
    for length in (84, 85, 86, 100, 110):
        end = sorted(n.night.unique())[length - 1]
        card = target(f.screen(n[n.night <= end], p, c))
        score = card["score"]
        assert card["status"] == "ongoing"
        assert score["scan_window_start"] <= card["start"] <= score["scan_window_end"]
        assert score["evaluation_nights"] == 30
        assert score["family_unit_nights"] >= score["family_floor"] == 30
        scores.append(score)
    assert scores[0]["scan_window_start"] == "2026-09-24"
    assert scores[1]["scan_window_start"] == "2026-09-25"
    for score in scores[1:]:
        assert score["scan_window_start"] == "2026-09-25"
        assert score["scan_window_end"] == "2026-10-24"
        assert score["family_unit_nights"] == scores[1]["family_unit_nights"]
    ongoing = target(f.screen(extend_calendar(n, "2026-12-18"), p, c))
    assert ongoing["status"] == "ongoing"
    assert ongoing["score"] == scores[-1]
    # A long ended spell uses the same carry-over rule after measured recovery.
    recovery = c.iloc[0].copy()
    recovery["time_utc"] = "2026-11-20T00:00:00"
    recovery["granule"] = "recovery"
    c = c.copy()
    c.loc[len(c)] = recovery
    ended = target(f.screen(extend_calendar(n, "2026-12-18"), p, c))
    assert ended["status"] == "ended"
    assert ended["score"] == scores[-1]


@pytest.mark.parametrize("available", [6, 30])
def test_empty_fleet_family_has_an_explicit_evaluation_night_floor(available):
    nights = pd.date_range("2026-09-01", periods=available).strftime("%Y-%m-%d")
    score = f.spell_scan_family(
        dict(start=nights[0], end=nights[-1]), dict.fromkeys(nights, 0)
    )
    assert score["eligible_unit_nights"] == 0
    assert score["family_unit_nights"] == score["family_floor"] == available
    assert score["family_unit_nights"] > 1


def test_family_counts_the_scored_units_own_eligible_nights():
    n, p, c = history_fixture(days=80, outage=55)
    data = f.screen(n, p, c)
    score = target(data)["score"]
    # This fixture has only the scored unit, so the fleet sum is exactly its
    # own eligible unit-nights, including the spell's tested start.
    counts = data["calibration"]["eligible_units_by_night"]
    own = sum(
        count
        for night, count in counts.items()
        if score["scan_window_start"] <= night <= score["scan_window_end"]
    )
    assert own > 0
    assert score["eligible_unit_nights"] == own
    assert score["family_unit_nights"] >= own


def test_spell_family_rejects_an_unrelated_evaluation_window():
    with pytest.raises(ValueError, match="evaluation nights in its own window"):
        f.spell_scan_family(
            dict(start="2026-09-25", end="2026-10-01"), {"2026-12-18": 0}
        )
