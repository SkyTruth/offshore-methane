# %%
"""Monthly local-night update, rebuild and optional publication (D60)."""

from __future__ import annotations

import json
import warnings
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

import backfill
import build_viewer
import pipeline


def last_processed(segments: list[tuple[Path, str]]) -> date:
    """Read actual local nights rather than a requested filename end."""
    return max(
        date.fromisoformat(
            str(
                pd.read_csv(
                    folder / f"site_nights_{tag}.csv", usecols=["night"]
                ).night.max()
            )
        )
        for folder, tag in segments
    )


def check_retention(start: date, today: date) -> None:
    age = (today - start).days
    if age >= 75:
        warnings.warn(
            f"Oldest uncached night {start} is {age} days old; NOAA NODD retains about 91 days.",
            stacklevel=2,
        )
    if age >= 91:
        raise ValueError(
            "Missing inputs exceed NODD retention; recover archived data before resuming."
        )


def _summary(folder: Path, tag: str) -> dict:
    path = folder / f"run_summary_{tag}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _expected_keys(folder: Path, tag: str) -> set:
    path = folder / f"jobs_{tag}.csv"
    return (
        set(
            pd.read_csv(path)[["satellite", "granule"]].itertuples(
                index=False, name=None
            )
        )
        if path.exists()
        else set()
    )


def _accepted(folder: Path, tag: str, summary: dict) -> bool:
    path = folder / "accepted_missing.json"
    return path.exists() and json.loads(path.read_text()) == dict(
        expected=summary.get("expected"), missing=summary.get("missing", [])
    )


def check_scope(
    folder: Path, tag: str, summary: dict, keys: set, freeze: bool = False
) -> None:
    """Persist original scope before any resumer can overwrite its manifests."""
    path = folder / "update_scope.json"
    if path.exists():
        original = json.loads(path.read_text())
        if summary and (
            summary.get("expected", 0) != original["expected"]
            or (original["keys"] and sorted(map(list, keys)) != original["keys"])
        ):
            raise RuntimeError(
                "Expected inputs vanished or changed from original update scope."
            )
    elif freeze and summary.get("expected"):
        folder.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                dict(expected=summary["expected"], keys=sorted(map(list, keys))),
                indent=2,
            )
        )
        tmp.replace(path)


def check_missing_exposures(folder: Path, tag: str, missing: list) -> None:
    """Verify each deficit, allowing known jobs with no in-range local nights."""
    exposure = folder / f"missing_observations_{tag}.csv"
    if not exposure.exists():
        raise RuntimeError("Missing inputs lack unknown-exposure records.")
    recorded = pd.read_csv(exposure)
    jobs = (
        {
            (j["satellite"], j["gid"]): j
            for j in pipeline.read_expected_jobs(folder, tag)
        }
        if pipeline.expected_jobs_path(folder, tag).exists()
        else {}
    )
    for sat, gid in missing:
        rows = recorded[(recorded.satellite == sat) & (recorded.granule == gid)]
        job = jobs.get((sat, gid))
        if job and job.get("sites"):
            # Validate every affected site-night, not just a nonempty CSV from
            # some other missing granule. Empty expected exposure is legitimate
            # for an east-longitude boundary pass on UTC end+1.
            expected = set()
            for sid, _, lon in job["sites"]:
                night = (
                    (job["start"] + timedelta(hours=lon / 15 - 12)).date().isoformat()
                )
                if tag[:10] <= night <= tag[11:]:
                    expected.add((sid, night))
            if expected.issubset(set(zip(rows.site_id, rows.night))):
                continue
        elif not rows.empty:
            # Legacy manifests may lack coordinates; pipeline abstains across
            # their segment. They cannot use the out-of-range exception.
            continue
        raise RuntimeError(
            "Missing inputs lack unknown-exposure records; archive recovery is required."
        )


def process_segment(
    folder: Path, start: str, end: str, today: date, workers: int, accept_missing: bool
) -> None:
    """Resume an immutable expected set; explicitly acknowledge missing inputs."""
    tag = f"{start}_{end}"
    before = _summary(folder, tag)
    original = _expected_keys(folder, tag)
    check_scope(folder, tag, before, original, freeze=True)
    saved = pipeline.expected_jobs_path(folder, tag)
    if saved.exists():
        jobs = pipeline.read_expected_jobs(folder, tag)
        uncached = [
            j for j in jobs if not pipeline.cache_complete(j, folder / "overpasses_v5")
        ]
        if uncached:
            check_retention(min(j["start"].date() for j in uncached), today)
    elif before.get("expected", 1) != before.get("complete", 0):
        check_retention(date.fromisoformat(start), today)
    backfill.main(start=start, end=end, out_dir=folder, workers=workers)
    summary = _summary(folder, tag)
    after = _expected_keys(folder, tag)
    check_scope(folder, tag, summary, after, freeze=True)
    if (original and after != original) or summary.get("expected", 0) < before.get(
        "expected", 0
    ):
        raise RuntimeError(
            "Expected inputs vanished or changed on resume; refusing diminished completeness."
        )
    if not summary.get("expected"):
        raise RuntimeError("No expected inputs; refusing to certify an empty listing.")
    if summary["complete"] < summary["expected"]:
        missing = summary.get("missing", [])
        if (
            not accept_missing
            or len(missing) != summary["expected"] - summary["complete"]
        ):
            raise RuntimeError(
                "Update has incomplete granules. Retry, or use accept_missing=True with a recorded missing list."
            )
        check_missing_exposures(folder, tag, missing)
        warnings.warn(
            f"Accepted {len(missing)} missing granules; affected site-nights are not observed.",
            stacklevel=2,
        )
        (folder / "accepted_missing.json").write_text(
            json.dumps(dict(expected=summary["expected"], missing=missing), indent=2)
        )


def main(
    publish: bool = False,
    today: date | None = None,
    workers: int = 12,
    base: Path = build_viewer.OUT_DIR,
    site_dir: Path = build_viewer.SITE_DIR,
    accept_missing: bool = False,
) -> Path:
    """Process through today−3; incomplete inputs require explicit acceptance.

    Accepted missing granules are retained in the manifest and affected nights
    are unknown. They do not hold every future month hostage. Expired uncached
    inputs require archive recovery, even with accept_missing=True.
    """
    today = today or date.today()
    pending = []
    for folder in base.glob("v5_update_*"):
        tag = folder.name.removeprefix("v5_update_")
        summary = _summary(folder, tag)
        check_scope(folder, tag, summary, _expected_keys(folder, tag))
        saved = pipeline.expected_jobs_path(folder, tag)
        if saved.exists() and summary:
            expected = {
                (j["satellite"], j["gid"])
                for j in pipeline.read_expected_jobs(folder, tag)
            }
            if (
                summary["expected"] != len(expected)
                or _expected_keys(folder, tag) != expected
            ):
                raise RuntimeError(
                    "Expected inputs vanished or changed; refusing diminished completeness."
                )
        state = (
            json.loads((folder / "status.json").read_text())
            if (folder / "status.json").exists()
            else {}
        )
        complete = (
            summary
            and summary.get("expected", 0) > 0
            and summary["complete"] == summary["expected"]
        )
        if (not complete and not _accepted(folder, tag, summary)) or state.get(
            "phase"
        ) not in (None, "done"):
            pending.append((folder, tag[:10], tag[11:]))
    for folder, start, end in sorted(pending, key=lambda x: x[1]):
        process_segment(folder, start, end, today, workers, accept_missing)
    segments = build_viewer.find_segments(base)
    start = last_processed(segments) + timedelta(days=1)
    end = today - timedelta(days=3)
    if start <= end:
        check_retention(start, today)
        process_segment(
            base / f"v5_update_{start}_{end}",
            start.isoformat(),
            end.isoformat(),
            today,
            workers,
            accept_missing,
        )
    result = build_viewer.main(
        segments=build_viewer.find_segments(base), site_dir=site_dir
    )
    if publish:
        build_viewer.publish_pages(result)
    return result


# %%
if __name__ == "__main__":
    main()
