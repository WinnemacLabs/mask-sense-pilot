from __future__ import annotations

import numpy as np
import pandas as pd


def breath_time_features(
    pressure: pd.Series, cycles: pd.DataFrame, fs: float = 1000.0
) -> pd.DataFrame:
    """Compute time-domain features per breath."""
    records = []
    for i, row in cycles.iterrows():
        start_idx = pressure.index.get_loc(row.start)
        end_idx = pressure.index.get_loc(row.end)
        pip_idx = int(row.pip_idx)
        pep_idx = int(row.pep_idx)

        pip = float(pressure.iloc[pip_idx])
        pep = float(pressure.iloc[pep_idx])

        inhale = pressure.iloc[start_idx : pip_idx + 1]
        exhale = pressure.iloc[pip_idx : end_idx + 1]
        inhale_integral = np.trapz(inhale.values, dx=1 / fs)
        exhale_integral = np.trapz(exhale.values, dx=1 / fs)

        baseline = pep
        peak = pip
        ten = baseline + 0.1 * (peak - baseline)
        ninety = baseline + 0.9 * (peak - baseline)
        idx10 = inhale[inhale >= ten].index[0]
        idx90 = inhale[inhale >= ninety].index[0]
        rise_time = (idx90 - idx10).total_seconds()

        ex_ten = peak - 0.1 * (peak - baseline)
        ex_ninety = peak - 0.9 * (peak - baseline)
        ex_idx10 = exhale[exhale <= ex_ten].index[0]
        ex_idx90 = exhale[exhale <= ex_ninety].index[0]
        fall_time = (ex_idx90 - ex_idx10).total_seconds()

        records.append(
            dict(
                breath=i + 1,
                PIP=pip,
                PEP=pep,
                inhale_integral=inhale_integral,
                exhale_integral=exhale_integral,
                rise_time=rise_time,
                fall_time=fall_time,
            )
        )
    return pd.DataFrame(records).set_index("breath")
