# Mask Sense Pilot

This repository contains scripts for analyzing respirator mask leak data.

## Running Tests

Install the required dependencies from `requirements.txt` and run `pytest`:

```bash
pip install -r requirements.txt
pytest
```

## Calculating Protection Factor

Run `calculate_fit_score.py` to compute a per-breath protection factor using
particle counts stored in `breath_db.sqlite`:

```bash
python calculate_fit_score.py --db breath_db.sqlite
```

The script shifts particle data back seven seconds to compensate for sampling
lag and stores the resulting protection factor in the `protection_factor` column
of the `breath_data` table.
