# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mask Sense Pilot - A Python 3.11 project for capturing and analyzing respirator mask leak data using pressure sensors (Teensy-based 3-axis) and particle counters (TSI WRPAS). Used for scientific/medical research on mask protection factors.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Key Commands

```bash
# Main automated testing protocol (primary entry point)
python collection/automated_testing.py [--start-section {mask1, mask1_leak, mask2, mask2_leak}]

# Lower-level serial data ingestion
python collection/pressure_particles_ingestion.py [--port PORT] [--wrpas WRPAS_PORT] [--baud 921600]

# Data validation
python utils/quick_validate.py [directory]
python utils/data_validation.py <csv_file>

# Fix 32-bit timestamp rollovers
python utils/fix_timestamp_rollovers.py <input_file> [output_file]

# Pressure-to-Protection-Factor analysis
python analysis/pressure_pf_analysis.py  # Main analysis pipeline

# Legacy analysis scripts (archived)
python archive/analysis_v1/segment_breaths.py run01.csv --db breath_db.sqlite [--prominence 1.5] [--plot-file output.png]
python archive/analysis_v1/protection_factor.py data.csv
python archive/analysis_v1/batch_analysis.py
```

## Architecture

### Serial Communication Pattern
- `SerialWorker` thread reads from serial ports into queues (non-blocking)
- Main thread drains queues via `_update_plot()` callback (100ms intervals)
- Teensy: 921600 baud, outputs `t_us,Pa_Global,Pa_Vertical,Pa_Horizontal,raw_Global,raw_Vertical,raw_Horizontal`
- WRPAS: 115200 baud, outputs `Conc1,Conc2` particle concentrations

### Data Flow
```
Teensy Serial → SerialWorker → Queue → _update_plot() → CSV + Live Plot
WRPAS Serial  → SerialWorker → Queue → Regex Parse → CSV
```

### CSV Output Format
```
t_us,Pa_Global,Pa_Vertical,Pa_Horizontal,raw_Global,raw_Vertical,raw_Horizontal,mask_particles,ambient_particles
```

### File Naming Convention
```
P{participant}_{mask_label}_{condition}_{stage}_{timestamp}.csv
Example: P01_AURA_leak_quiet_breathing.csv
```

### Breath Segmentation Algorithm (archive/analysis_v1/segment_breaths.py)
- Low-pass Butterworth filter (4th order, 3Hz cutoff)
- Find negative peaks with prominence thresholding
- Preceding falling zero-crossing marks breath boundary

## Data Organization

```
data/
├── P0/, P1/, P2/, ...     # Participant directories
│   ├── rsc_P*_mask*.csv   # Session recordings
│   └── zeroing/           # Zero calibration data (.npy, .txt, .png)
```

## Key Files

- `collection/automated_testing.py` - Main entry point, interactive CLI protocol
- `collection/pressure_particles_ingestion.py` - Core serial communication & live plotting
- `analysis/pressure_pf_analysis.py` - Pressure-to-Protection-Factor correlation analysis
- `archive/analysis_v1/segment_breaths.py` - Breath segmentation from pressure traces (archived)
- `archive/analysis_v1/protection_factor.py` - Protection factor computation (archived)
- `documentation/Testing Protocol.md` - Detailed experimental procedure

## Directory Structure

```
pressure-fit/
├── collection/          # Data collection tools
├── utils/               # Data utilities (validation, fixes)
├── analysis/            # Active analysis scripts
├── archive/             # Archived analysis (v1) and notebooks
├── data/                # Participant data
├── documentation/       # Protocol docs
└── tests/               # Unit tests
```
