from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import gmean


def compute_pf(
    mask_pc: pd.Series,
    amb_pc: pd.Series,
    breath_df: pd.DataFrame,
    shift_s: float = 7.0,
) -> pd.DataFrame:
    """Compute per-breath protection factor and rolling geometric mean."""
    mask_pc = mask_pc.copy()
    amb_pc = amb_pc.copy()
    mask_pc.index = mask_pc.index - pd.Timedelta(seconds=shift_s)
    amb_pc.index = amb_pc.index - pd.Timedelta(seconds=shift_s)

    pfs = []
    for _, row in breath_df.iterrows():
        seg_mask = mask_pc[row.start : row.end]
        seg_amb = amb_pc[row.start : row.end]
        max_mask = seg_mask.max()
        mean_amb = seg_amb.mean()
        pf = np.nan
        if pd.notna(max_mask) and pd.notna(mean_amb) and max_mask != 0:
            pf = float(mean_amb) / float(max_mask)
        pfs.append(pf)
    breath_df = breath_df.copy()
    breath_df["PF"] = pfs

    rolling = []
    starts = breath_df["start"]
    for i in range(len(breath_df)):
        t0 = breath_df.iloc[i].end - pd.Timedelta(seconds=60)
        mask = starts >= t0
        vals = breath_df.loc[mask & (breath_df.index <= i), "PF"].dropna()
        if len(vals):
            rolling.append(float(gmean(vals)))
        else:
            rolling.append(np.nan)
    breath_df["PF_roll_geom"] = rolling
    return breath_df
