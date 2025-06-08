# Mask Sense Pilot

This repository contains scripts for capturing and analyzing respirator mask leak data.

## Automated Testing

Use `automated_testing.py` to guide participants through the protocol described in `documentation/Testing Protocol.md`. The script displays live pressure and particle plots while automatically starting and stopping recordings using the Teensy logger and WRPAS devices. The participant advances each stage by pressing **Enter**, including the sensor zeroing step.

Run:

```bash
python automated_testing.py
```

The legacy `pressure-particles-ingestion.py` entry point is still available and now wraps the importable module `pressure_particles_ingestion.py`.
