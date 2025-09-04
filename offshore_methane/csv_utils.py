"""
csv_utils.py

Utilities for treating the CSV files in ../data as a small, joined
"virtual database". The join model is:

  events.csv      (rows keyed by id)
    ⨝  event_granule.csv  (many-to-many via event_id ↔ system_index)
      ⨝  granules.csv     (rows keyed by system_index)

Core capabilities:
- Load events, mappings, and granules with robust typing.
- Produce joined DataFrames filtered by scene/local quality controls.
- Query by event id (eid) or by granule id (gid, aka system:index).
- Simple upsert helpers to create/update rows in each CSV.

Filtering rules (kwargs):
- scene_cloud_pct: defaults to cfg.MASK_PARAMS["cloud"]["scene_cloud_pct"].
  Applied to column "cloudiness"; rows with missing values are dropped when
  this filter is active.
- scene_sga_range: defaults to cfg.MASK_PARAMS["sunglint"]["scene_sga_range"].
  Applied to column "sga_scene" using strict inequality (min < x < max);
  missing values are dropped when active.
- local_sga_range: default None; when provided, applied to column
  "sga_local_median" using strict inequality; rows with missing values are
  dropped when active.
- local_sgi_range: default None; when provided, applied to column
  "sgi_median" using strict inequality; rows with missing values are dropped
  when active.

Notes
-----
- Column synonyms are normalised on load: sga_local_med → sga_local_median,
  sgi_local_med → sgi_median.
- The join preserves all event rows that appear in event_granule.csv. Empty
  system_index entries (used to mark "no granules") are removed in joined
  queries by default because they do not map to granules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

import offshore_methane.config as cfg

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
EVENTS_CSV: Path = cfg.EVENTS_CSV
GRANULES_CSV: Path = cfg.GRANULES_CSV
EVENT_GRANULE_CSV: Path = cfg.EVENT_GRANULE_CSV


# -----------------------------------------------------------------------------
# Loader helpers
# -----------------------------------------------------------------------------
def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_events() -> pd.DataFrame:
    """Load events.csv and coerce common types. Returns an empty DataFrame if
    the file does not exist.
    """
    df = _read_csv(EVENTS_CSV).copy()
    if df.empty:
        return df
    # Ensure canonical column names
    if "id" not in df.columns:
        # Fall back to index as id if absent
        df["id"] = df.index
    # Types
    for col in ("id",):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ("lon", "lat"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Keep dates as strings to avoid timezone parsing surprises
    for col in ("start", "end"):
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


def load_granules() -> pd.DataFrame:
    """Load granules.csv with canonicalised column names and types.

    Normalises synonyms:
      - sga_local_med   → sga_local_median
      - sgi_local_med   → sgi_median
    """
    df = _read_csv(GRANULES_CSV).copy()
    if df.empty:
        return df

    # Normalise synonyms
    if "sga_local_median" not in df.columns and "sga_local_med" in df.columns:
        df = df.rename(columns={"sga_local_med": "sga_local_median"})
    if "sgi_median" not in df.columns and "sgi_local_med" in df.columns:
        df = df.rename(columns={"sgi_local_med": "sgi_median"})

    # Types
    df["system_index"] = df.get("system_index", "").astype(str)
    for col in ("sga_scene", "cloudiness", "sga_local_median", "sgi_median"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].astype(str)
    return df


def load_event_granule() -> pd.DataFrame:
    """Load event_granule.csv with light typing. Rows with empty
    system_index are kept (they mark "no granules found").
    """
    df = _read_csv(EVENT_GRANULE_CSV).copy()
    if df.empty:
        return df
    if "event_id" not in df.columns:
        df["event_id"] = pd.NA
    df["event_id"] = pd.to_numeric(df["event_id"], errors="coerce").astype("Int64")
    if "system_index" not in df.columns:
        df["system_index"] = ""
    df["system_index"] = df["system_index"].astype(str)
    return df


# -----------------------------------------------------------------------------
# Join + filter helpers
# -----------------------------------------------------------------------------
@dataclass
class Filters:
    scene_cloud_pct: Optional[float]
    scene_sga_range: Optional[Tuple[float, float]]
    local_sga_range: Optional[Tuple[float, float]]
    local_sgi_range: Optional[Tuple[float, float]]


def _resolve_filters(**kwargs) -> Filters:
    # Defaults from config
    scene_cloud = kwargs.get(
        "scene_cloud_pct", float(cfg.MASK_PARAMS["cloud"]["scene_cloud_pct"])
    )
    scene_sga = kwargs.get(
        "scene_sga_range", tuple(cfg.MASK_PARAMS["sunglint"]["scene_sga_range"])
    )
    local_sga = kwargs.get("local_sga_range")  # default None
    local_sgi = kwargs.get("local_sgi_range")  # default None

    # Normalise tuple-ish inputs
    def _as_pair(x):
        if x is None:
            return None
        if isinstance(x, (list, tuple)) and len(x) == 2:
            return float(x[0]), float(x[1])
        raise ValueError("Range filters must be two-element (min, max)")

    return Filters(
        scene_cloud_pct=None if scene_cloud is None else float(scene_cloud),
        scene_sga_range=_as_pair(scene_sga) if scene_sga is not None else None,
        local_sga_range=_as_pair(local_sga) if local_sga is not None else None,
        local_sgi_range=_as_pair(local_sgi) if local_sgi is not None else None,
    )


def virtual_db(**kwargs) -> pd.DataFrame:
    """Return the full joined DataFrame (events ⨝ mapping ⨝ granules), then
    apply any provided filters.

    Filters accepted as kwargs (see module docstring). If a filter is not
    provided (or is None), it is not applied. The two scene-level filters
    default to config values and are applied by default.
    """
    events = load_events()
    mapping = load_event_granule()
    grans = load_granules()

    if mapping.empty or events.empty:
        return pd.DataFrame()

    # Rename for join clarity
    ev = events.rename(columns={"id": "event_id"})

    # Remove mapping rows with empty system_index (marker for no granules)
    map_nonempty = mapping[mapping["system_index"].astype(str).str.len() > 0].copy()

    # event ⨝ mapping
    joined = map_nonempty.merge(ev, on="event_id", how="left", suffixes=("", ""))
    # ⨝ granules (metadata columns)
    joined = joined.merge(
        grans,
        on="system_index",
        how="left",
        suffixes=("", ""),
    )

    # Apply filters
    f = _resolve_filters(**kwargs)

    def _between_strict(series: pd.Series, lo: float, hi: float) -> pd.Series:
        return (series > lo) & (series < hi)

    if f.scene_cloud_pct is not None:
        # drop rows with missing cloudiness when filter is active
        joined = joined[joined["cloudiness"].notna()]
        joined = joined[joined["cloudiness"] <= f.scene_cloud_pct]

    if f.scene_sga_range is not None:
        lo, hi = f.scene_sga_range
        joined = joined[joined["sga_scene"].notna()]
        joined = joined[_between_strict(joined["sga_scene"], lo, hi)]

    # Local filters only when provided
    if f.local_sga_range is not None:
        lo, hi = f.local_sga_range
        col = (
            "sga_local_median"
            if "sga_local_median" in joined.columns
            else "sga_local_med"
        )
        if col not in joined.columns:
            # No column available → nothing matches when filter is requested
            return joined.iloc[0:0]
        joined = joined[joined[col].notna()]
        joined = joined[_between_strict(joined[col], lo, hi)]

    if f.local_sgi_range is not None:
        lo, hi = f.local_sgi_range
        col = "sgi_median" if "sgi_median" in joined.columns else "sgi_local_med"
        if col not in joined.columns:
            return joined.iloc[0:0]
        joined = joined[joined[col].notna()]
        joined = joined[_between_strict(joined[col], lo, hi)]

    return joined.reset_index(drop=True)


# -----------------------------------------------------------------------------
# High-level queries
# -----------------------------------------------------------------------------
def df_for_event(eid: int, **kwargs) -> pd.DataFrame:
    """Return joined rows for a single event id, filtered by any provided
    quality kwargs (see virtual_db).
    """
    df = virtual_db(**kwargs)
    if df.empty:
        return df
    return df[df["event_id"] == int(eid)].reset_index(drop=True)


def df_for_granule(gid: str, **kwargs) -> pd.DataFrame:
    """Return joined rows for a single granule id (system:index), filtered by
    any provided quality kwargs (see virtual_db).
    """
    df = virtual_db(**kwargs)
    if df.empty:
        return df
    return df[df["system_index"].astype(str) == str(gid)].reset_index(drop=True)


# -----------------------------------------------------------------------------
# Upsert helpers (create/update rows)
# -----------------------------------------------------------------------------
def upsert_event(row: dict) -> None:
    """Create or update a row in events.csv keyed by id.

    Keys: id (required), lon, lat, start, end, plus any arbitrary columns.
    """
    if "id" not in row:
        raise ValueError("'id' is required to upsert an event row")
    df = load_events()
    if df.empty:
        df = pd.DataFrame(columns=["id", "lon", "lat", "start", "end"])  # base schema
    key = int(row["id"])  # normalize
    # Upsert
    if (df.get("id") == key).any():
        idx = df.index[df["id"] == key][0]
        for k, v in row.items():
            df.loc[idx, k] = v
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(EVENTS_CSV, index=False)


def upsert_granule(row: dict) -> None:
    """Create or update a row in granules.csv keyed by system_index.

    Keys: system_index (required). Accepts quality columns such as sga_scene,
    cloudiness, timestamp, sga_local_median (or sga_local_med), sgi_median
    (or sgi_local_med).
    """
    if "system_index" not in row:
        raise ValueError("'system_index' is required to upsert a granule row")
    df = load_granules()
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "system_index",
                "sga_scene",
                "cloudiness",
                "timestamp",
                "sga_local_median",
                "sgi_median",
            ]
        )

    # Canonicalise synonyms in input
    if "sga_local_med" in row and "sga_local_median" not in row:
        row["sga_local_median"] = row.pop("sga_local_med")
    if "sgi_local_med" in row and "sgi_median" not in row:
        row["sgi_median"] = row.pop("sgi_local_med")

    key = str(row["system_index"])  # normalize
    if (df.get("system_index") == key).any():
        idx = df.index[df["system_index"] == key][0]
        for k, v in row.items():
            df.loc[idx, k] = v
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    # Persist with reduced float precision (4 decimals) to keep files tidy
    df.to_csv(GRANULES_CSV, index=False, float_format="%.4f")


def upsert_event_granule(event_id: int, system_index: str) -> None:
    """Create an event ↔ granule mapping row (idempotent)."""
    df = load_event_granule()
    if df.empty:
        df = pd.DataFrame(columns=["event_id", "system_index"])
    exists = (
        (df.get("event_id") == int(event_id))
        & (df.get("system_index").astype(str) == str(system_index))
    ).any()
    if not exists:
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [{"event_id": int(event_id), "system_index": str(system_index)}]
                ),
            ],
            ignore_index=True,
        )
        df.to_csv(EVENT_GRANULE_CSV, index=False)


# Convenience: update the two local metrics for a granule
def update_local_metrics(
    system_index: str,
    *,
    sga_local_median: Optional[float] = None,
    sgi_median: Optional[float] = None,
) -> None:
    row: dict = {"system_index": system_index}
    if sga_local_median is not None:
        row["sga_local_median"] = sga_local_median
    if sgi_median is not None:
        row["sgi_median"] = sgi_median
    if len(row) > 1:
        upsert_granule(row)
