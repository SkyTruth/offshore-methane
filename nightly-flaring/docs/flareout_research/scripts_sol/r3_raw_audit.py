"""Compare the newly inspected local donors with the original overpass CSVs."""

from pathlib import Path
import json
import sys

import pandas as pd

sys.path.insert(0,".")
from pipeline import dedupe_overpasses, strict_clear
from nightfire import solar_zenith_deg

OUT=Path("outputs/flareout_research/scripts_sol")
IDS=[4485,4500,4515,4561,4575]


def main():
    cols=["satellite","granule","time_utc","site_id","row","col","dist_m","sza",
          "cloud_mask","cloud_clear_neighbors","detected","rh_mw","n_hot_m10","n_sources"]
    parts=[]
    total=0
    for folder,tag in [("v5_backfill","2026-08-09_2026-10-05"),
                       ("v5_30n","2026-10-06_2026-11-04")]:
        for block in pd.read_csv(f"outputs/{folder}/overpasses_{tag}.csv",usecols=cols,chunksize=200000):
            total+=len(block)
            parts.append(block[block.site_id.isin(IDS)].copy())
    raw=pd.concat(parts,ignore_index=True)
    o=dedupe_overpasses(raw)
    s=pd.read_csv("outputs/gfw_oil_sites.csv").set_index("site_id")
    o["night"]=(pd.to_datetime(o.time_utc)+pd.to_timedelta(o.site_id.map(s.lon)/15-12,"h")).dt.strftime("%Y-%m-%d")
    o=o[(o.sza>=97)&o.night.between("2026-08-09","2026-11-04")].copy()
    o["clear"]=strict_clear(o)
    cached=pd.read_pickle("outputs/flareout_research/scripts_opus/cache/passes.pkl")
    cached=cached[cached.site_id.isin(IDS)]
    compare=["site_id","satellite","granule","time_utc","col","sza","cloud_mask",
             "cloud_clear_neighbors","detected","rh_mw","clear","n_hot_m10","n_sources","night"]
    a=o[compare].sort_values(["site_id","satellite","time_utc"]).reset_index(drop=True)
    b=cached[compare].sort_values(["site_id","satellite","time_utc"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a,b,check_dtype=False)
    # Copies have SZA>=95 at detection; the night-level analysis requires >=97.
    # Audit the new 4500 source's contributing granules against its site pixel.
    groups=pd.read_csv(OUT/"r3_4500_groups.csv")
    exposure=raw[raw.site_id==4500].groupby(["satellite","granule"]).sza.min()
    test=groups.merge(exposure.rename("site_sza"),on=["satellite","granule"],how="left")
    # An unassigned source may be covered when the nominal platform pixel is
    # not, so it must not be discarded for lacking that site's exposure row.
    # Calculate SZA at every recorded group centroid using local pipeline code.
    test["source_sza"]=[float(solar_zenith_deg(row.centroid_lat,row.centroid_lon,
                                              pd.Timestamp(row.time_utc).to_pydatetime()))
                        for row in test.itertuples(index=False)]
    assert (test.source_sza>=97).all()
    assert (test.loc[test.site_sza.notna(),"site_sza"]>=97).all()
    test[test.site_sza.isna()].to_csv(OUT/"r3_source_groups_without_site_granule.csv",index=False)
    result=dict(primary_overpass_rows_read=total,site_ids=IDS,compared_rows=len(a),
                columns=compare,cache_matches_original=True,
                source4500_groups=len(test),minimum_source4500_site_sza=float(test.site_sza.min()),
                minimum_source4500_centroid_sza=float(test.source_sza.min()),
                source_groups_without_site_granule=int(test.site_sza.isna().sum()),
                all_source4500_groups_have_source_sza97=True)
    (OUT/"r3_raw_audit.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    main()
