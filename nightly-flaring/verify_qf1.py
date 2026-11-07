# %%
"""Inspect one SDR granule's raw radiance and documented QF1 saturation bits."""

from pathlib import Path
from datetime import datetime
import json

import h5py
import numpy as np
import pandas as pd
import requests

import nightfire as nf
import nodd

HERE = Path(__file__).resolve().parent


def main(out_dir: Path = HERE / "outputs" / "dev_v3") -> dict:
    """Download one high-radiance M12 granule and save an empirical QF audit."""
    old = pd.read_csv(HERE / "outputs/dev_v2/overpasses_2026-10-29_2026-10-29.csv")
    row = old.loc[old.rh_mw.idxmax()]
    session = requests.Session()
    index = nodd.granule_index(
        row.satellite, datetime.fromisoformat(row.time_utc[:10]), session
    )
    bucket = nodd.BUCKETS[row.satellite][0]
    key = index[row.granule]["M12"]
    response = session.get(f"{nodd.bucket_url(bucket)}/{key}", timeout=120)
    response.raise_for_status()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "qf1_verification_M12.h5"
    path.write_bytes(response.content)
    with h5py.File(path) as h:
        group = h["All_Data/VIIRS-M12-SDR_All"]
        raw = group["Radiance"][()]
        rad = nf.sdr_radiance(raw, group["RadianceFactors"][()])
        qf = group["QF1_VIIRSMBANDSDR"][()]
        sat = (qf >> 2) & 3
        near = np.isfinite(rad) & (rad >= 0.98 * nf.SATURATION["M12"])
        result = dict(
            key=key,
            attributes={k: str(v) for k, v in group["QF1_VIIRSMBANDSDR"].attrs.items()},
            codes={str(i): int((sat == i).sum()) for i in range(4)},
            near_saturation=int(near.sum()),
            near_flagged=int((near & (sat != 0)).sum()),
            partial_finite=int(((sat == 1) & np.isfinite(rad)).sum()),
            full_fill=int(((sat == 2) & ~np.isfinite(rad)).sum()),
            maximum_radiance=float(np.nanmax(rad)),
        )
    (out_dir / "qf1_verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


# %%
if __name__ == "__main__":
    main()
