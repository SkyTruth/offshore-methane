"""Check memo-critical cached histories against the original two CSV segments.

Reads only selected overpass columns in chunks. Does not regenerate or modify
any cache; writes a small provenance check inside scripts_sol/.
"""
from pathlib import Path
import json
import sys

import pandas as pd

sys.path.insert(0,".")
from pipeline import dedupe_overpasses,strict_clear

OUT=Path("outputs/flareout_research/scripts_sol")
CACHE=Path("outputs/flareout_research/scripts_opus/cache")
SEGMENTS=[("v5_backfill","2026-08-09_2026-10-05"),("v5_30n","2026-10-06_2026-11-04")]
IDS=[927,928,1035,3662,3644,6011,3656,3669,4582,4583,4497]


def main():
    cols=["satellite","granule","time_utc","site_id","row","col","dist_m","sza",
          "cloud_mask","cloud_clear_neighbors","detected","rh_mw","n_hot_m10","n_sources"]
    parts=[]
    copies=[]
    primary_rows=0
    for folder,tag in SEGMENTS:
        path=Path("outputs")/folder
        for c in pd.read_csv(path/f"overpasses_{tag}.csv",usecols=cols,chunksize=200000):
            primary_rows+=len(c)
            parts.append(c[c.site_id.isin(IDS)].copy())
        copies.append(pd.read_csv(path/f"copies_{tag}.csv",low_memory=False))
    o=dedupe_overpasses(pd.concat(parts,ignore_index=True))
    s=pd.read_csv("outputs/gfw_oil_sites.csv").set_index("site_id")
    o["night"]=(pd.to_datetime(o.time_utc)+pd.to_timedelta(o.site_id.map(s.lon)/15-12,"h")).dt.strftime("%Y-%m-%d")
    o=o[(o.sza>=97)&o.night.between("2026-08-09","2026-11-04")].copy()
    o["clear"]=strict_clear(o)
    cached=pd.read_pickle(CACHE/"passes.pkl")
    cached=cached[cached.site_id.isin(IDS)]
    compare=["site_id","satellite","granule","time_utc","col","sza","cloud_mask",
             "cloud_clear_neighbors","detected","rh_mw","clear","n_hot_m10","n_sources","night"]
    a=o[compare].sort_values(["site_id","satellite","time_utc"]).reset_index(drop=True)
    b=cached[compare].sort_values(["site_id","satellite","time_utc"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a,b,check_dtype=False)
    c=pd.concat(copies,ignore_index=True).drop_duplicates(["satellite","granule","group_id"])
    cc=pd.read_pickle(CACHE/"copies.pkl").drop_duplicates(["satellite","granule","group_id"])
    ccols=["satellite","granule","group_id","time_utc","centroid_lat","centroid_lon","assigned_site","group_rh_mw"]
    ca=c[ccols].sort_values(["satellite","granule","group_id"]).reset_index(drop=True)
    cb=cc[ccols].sort_values(["satellite","granule","group_id"]).reset_index(drop=True)
    keys=["satellite","granule","group_id"]
    merged=ca.merge(cb,on=keys,how="outer",indicator=True,suffixes=("_primary","_cache"))
    missing=merged[merged._merge=="left_only"]
    extra=merged[merged._merge=="right_only"]
    common=ca.merge(cb[keys],on=keys).sort_values(keys).reset_index(drop=True)
    cb=cb.sort_values(keys).reset_index(drop=True)
    # The retained constituent can change its timestamp while invariant group
    # location/heat stay identical. Compare group values separately from that
    # representative-copy timestamp.
    pd.testing.assert_frame_equal(common.drop(columns="time_utc"),cb.drop(columns="time_utc"),check_dtype=False)
    dt=(pd.to_datetime(common.time_utc)-pd.to_datetime(cb.time_utc)).abs().dt.total_seconds()
    missing.to_csv(OUT/"r2_copy_groups_missing_from_opus_cache.csv",index=False)
    result={"raw_primary_overpass_rows_read":primary_rows,"sites_checked":len(IDS),
            "site_ids":IDS,"deduped_night_passes_compared":len(a),
            "overpass_columns_compared":compare,"all_selected_overpasses_match":True,
            "primary_grouped_copies":len(ca),"opus_cached_grouped_copies":len(cb),
            "copy_columns_compared":[x for x in ccols if x!="time_utc"],"cached_common_group_values_match":True,
            "common_groups_with_different_representative_time":int((dt>0).sum()),
            "maximum_representative_time_difference_seconds":float(dt.max()),
            "primary_groups_missing_from_opus_cache":len(missing),
            "cached_groups_not_in_primary":len(extra),
            "interpretation":"component_id/time deduplication can drop split maxima; Round 2 spatial checks must read original copy CSVs"}
    (OUT/"r2_primary_check.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":main()
