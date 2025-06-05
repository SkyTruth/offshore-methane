# %%
#!/usr/bin/env python3
# file: make_k_lookup_s2.py
"""
Pre-compute k_{j,g}(SZA,VZA,P) for Sentinel-2 Bands 9–12.
Requires:
    pip install numpy scipy xarray netCDF4 hapi
    SRF excel from ESA:  'Sentinel-2A MSI Spectral Responses.xlsx'
"""

import numpy as np
import pandas as pd
import xarray as xr
from hapi import absorptionCoefficient_Voigt, fetch

####################  1.  CONFIG  ##########################
INSTRUMENT = "S2B"
RELATIVE_FOLDER = "data"
SRF_FILE = f"{INSTRUMENT}_MSI_SRF.csv"  # change to S2B for that instrument
BANDS = {
    "B9": (0.930, 0.960),
    "B10": (1.355, 1.395),
    "B11": (1.55, 1.70),
    "B12": (2.09, 2.30),
}
GASES = {"H2O": 1, "CO2": 2, "CH4": 6}  # HITRAN ID
SZA_GRID = np.arange(0, 71, 10)
VZA_GRID = np.arange(0, 41, 10)
P_GRID = np.array([600, 800, 1013])  # hPa
TEMP = 296.0
############################################################


def load_srf(band):
    """Return λ (µm), SRF for selected band from excel file."""
    df = pd.read_csv(SRF_FILE)
    # instead of sheet name, choose the column based on teh fact that they look like this: SR_WL	S2A_SR_AV_B1	S2A_SR_AV_B2	S2A_SR_AV_B3	S2A_SR_AV_B4
    return df["SR_WL"].to_numpy() / 1000.0, df[f"{INSTRUMENT}_SR_AV_{band}"].to_numpy()


def band_average_sigma(nu, sigma, lam, srf):
    """Spectrally integrate σ over SRF (λ in µm)."""
    # interpolate σ(λ) on SRF grid
    sigma_interp = np.interp(lam, 1e4 / nu[::-1], sigma[::-1])  # ν(cm⁻¹) ↔ λ(µm)
    return np.trapz(sigma_interp * srf, lam) / np.trapz(srf, lam)


def compute_k():
    k = xr.DataArray(
        np.zeros((len(BANDS), len(GASES), len(SZA_GRID), len(VZA_GRID), len(P_GRID))),
        coords=[list(BANDS), list(GASES), SZA_GRID, VZA_GRID, P_GRID],
        dims=["band", "gas", "sza", "vza", "pres"],
    )
    for j, (bname, (lam_lo, lam_hi)) in enumerate(BANDS.items()):
        lam_srf, srf = load_srf(bname)
        for gname, gid in GASES.items():
            fetch(gname, gid, 1, 1e4 / lam_hi, 1e4 / lam_lo)
            nu, sigma = absorptionCoefficient_Voigt(
                SourceTables=gname,
                Diluent={"air": 1},
                HITRAN_units=False,
                OmegaStep=0.01,
                Environment={"T": TEMP, "p": 1},
            )
            σ̄ = band_average_sigma(nu, sigma, lam_srf, srf)  # m² molec⁻¹
            for si, sza in enumerate(SZA_GRID):
                for vi, vza in enumerate(VZA_GRID):
                    m = 1 / np.cos(np.deg2rad(sza)) + 1 / np.cos(np.deg2rad(vza))
                    for pi, pres in enumerate(P_GRID):
                        k.loc[bname, gname, sza, vza, pres] = m * σ̄
    return k


k = compute_k()
k.name = "k_prime"  # air-mass-scaled σ; leave surface/solar factors to runtime
k.to_netcdf(f"k_{INSTRUMENT}_v1.nc")
print(f"Saved k-LUT to k_{INSTRUMENT}_v1.nc")

# %%
