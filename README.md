# RAMAN-analysis

Python tools and notebooks for working with Raman map data from CZTS samples. The repository is currently organized around saved Raman hypercubes in `.npz` format, preprocessing with `ramanspy`, and generating spatial peak maps or interactive dashboards.

## Current File Structure

```text
RAMAN-analysis/
|-- analysis.py
|-- analysis_dashboard.ipynb
|-- archive/
|-- data_folder/
|-- parsed_spectra_551_points/
|-- peak_info.md
|-- plot_peak_map.py
|-- preprocess.py
|-- README.md
|-- requirements.txt
|-- utils.py
`-- CZTS_31_249606_290526_532nm_10%_10s_2a_s_map1700x1700_no_edge.txt
```

## What Each File Does

- `analysis_dashboard.ipynb` - Notebook for interactive Raman data exploration and dashboard-style analysis.
- `plot_peak_map.py` - Main command-line script for plotting Raman peak intensity maps, peak-to-base ratio maps, and Plotly dashboards from `.npz` hypercubes.
- `preprocess.py` - Main command-line preprocessing pipeline for baseline correction, denoising, cosmic-ray removal, and spectral cropping.
- `analysis.py` - Older or alternate preprocessing/analysis script with similar functionality to `preprocess.py`.
- `utils.py` - Shared helper functions for parsing map text files, creating hypercubes, loading `.npz` data, smoothing maps, and normalizing data.
- `peak_info.md` - Notes on Raman peak positions for CZTS and possible secondary phases.
- `requirements.txt` - Python package list used for the current environment.
- `data_folder/` - Raw Raman `.txt` and `.wdf` data files, plus older parsed output folders.
- `parsed_spectra_551_points/` - Current parsed hypercube outputs:
  - `raman_hypercube_manual.npz`
  - `raman_hypercube_manual_processed.npz`
- `archive/` - Older scripts, plots, dashboards, parser versions, and exported map images kept for reference.

## Data Format

The plotting and preprocessing scripts expect a compressed NumPy `.npz` hypercube containing these arrays:

- `x` - X coordinates of the Raman map.
- `y` - Y coordinates of the Raman map.
- `wave` - Raman shift / wavenumber axis.
- `intensity` - 3D intensity cube with shape `(nx, ny, nw)`.

Raw Raman text files are expected to use tab-separated columns:

```text
#X    #Y    #Wave    #Intensity
```

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

The most important packages are `numpy`, `pandas`, `matplotlib`, `plotly`, `ramanspy`, `scipy`, and `scikit-learn`.

## Typical Workflow

### 1. Start From a Parsed Hypercube

The current working hypercube files are in `parsed_spectra_551_points/`.

Use the unprocessed cube when you want to run preprocessing yourself:

```powershell
parsed_spectra_551_points\raman_hypercube_manual.npz
```

Use the processed cube when you want to go directly to plotting:

```powershell
parsed_spectra_551_points\raman_hypercube_manual_processed.npz
```

### 2. Preprocess a Hypercube

Run `preprocess.py` on an input `.npz` file:

```powershell
python preprocess.py parsed_spectra_551_points\raman_hypercube_manual.npz --baseline als --denoise savgol --cosmic-rays --output parsed_spectra_551_points\raman_hypercube_manual_processed.npz
```

Useful options:

- `--baseline als` - apply ALS baseline correction.
- `--baseline none` - skip baseline correction.
- `--denoise savgol` - apply Savitzky-Golay denoising.
- `--denoise gaussian` - apply Gaussian denoising.
- `--denoise none` - skip denoising.
- `--cosmic-rays` - remove spikes using `ramanspy`.
- `--crop-min` and `--crop-max` - keep only a selected wavenumber range.
- `--output` or `-o` - choose the output `.npz` path.

Example with spectral cropping:

```powershell
python preprocess.py parsed_spectra_551_points\raman_hypercube_manual.npz --baseline als --denoise savgol --crop-min 250 --crop-max 500
```

### 3. Plot a Peak Intensity Map

Generate a static Matplotlib peak map:

```powershell
python plot_peak_map.py parsed_spectra_551_points\raman_hypercube_manual_processed.npz --peak 338 --output peak_map_338.png
```

Generate a peak-to-base ratio map:

```powershell
python plot_peak_map.py parsed_spectra_551_points\raman_hypercube_manual_processed.npz --peak 338 --map-type peak-to-base-ratio --output peak_to_base_338.png
```

Apply spatial smoothing:

```powershell
python plot_peak_map.py parsed_spectra_551_points\raman_hypercube_manual_processed.npz --peak 338 --smooth-sigma 1.5 --output peak_map_338_smoothed.png
```

### 4. Open an Interactive Dashboard

Create a Plotly dashboard with a spatial heatmap and a spectrum view:

```powershell
python plot_peak_map.py parsed_spectra_551_points\raman_hypercube_manual_processed.npz --peak 338 --dashboard --output peak_map_338_dashboard.html
```

## Peak Reference Notes

See `peak_info.md` for currently tracked peak positions. Important CZTS-related peaks include:

- CZTS main peaks: `287 cm^-1`, `338 cm^-1`
- ZnS: `272-277 cm^-1`, `349-352 cm^-1`
- CuS: `475 cm^-1`
- CuSnS shoulder: `355 cm^-1`

## Parsing Raw Text Files

The active repository no longer has a top-level parser CLI. Parsing helpers live in `utils.py`, especially:

- `parse_map_file`
- `build_hypercube`
- `save_hypercube`
- `parse_and_save`

Older parser scripts are kept in `archive/`, including:

- `archive/single_data_parser.py`
- `archive/run_parser.py`
- `archive/single_data_parser.ipynb`

Use these archived files as references if you need to regenerate `.npz` hypercubes from raw Raman `.txt` maps.

## Notes

- Keep large raw Raman files in `data_folder/`.
- Keep current parsed `.npz` files in a `parsed_spectra_*` folder.
- Keep older experiments, generated plots, and deprecated scripts in `archive/`.
- `preprocess.py` requires `ramanspy`.
- `plot_peak_map.py` requires `plotly` only for interactive maps and dashboards.
