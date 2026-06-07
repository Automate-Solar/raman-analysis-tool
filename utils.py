"""Utility helpers for parsing Raman map data and generating spatial peak maps."""

"""Utility helpers for parsing Raman map data and generating spatial peak maps."""

import argparse
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_map_file(txt_filename: Path) -> pd.DataFrame:
    """Read a Raman map text file into a pandas DataFrame.

    The input file is expected to contain four TSV columns with header
    labels ["#X", "#Y", "#Wave", "#Intensity"].

    Args:
        txt_filename: Path to the map text file.

    Returns:
        DataFrame with columns ["#X", "#Y", "#Wave", "#Intensity"].
    """
    df = pd.read_csv(txt_filename, sep=r"\t+", engine="python")
    expected = ["#X", "#Y", "#Wave", "#Intensity"]
    if list(df.columns) != expected:
        raise ValueError(f"Unexpected columns: {list(df.columns)}; expected {expected}")
    return df


def create_output_dirs(base_dir: Path) -> Path:
    """Create a timestamped output directory under the base directory.

    Args:
        base_dir: Directory where parsed outputs should be created.

    Returns:
        Path to the created output directory.
    """

    out_dir = base_dir / "parsed_spectra_" + time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def build_hypercube(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a spatial hypercube from the parsed map DataFrame.

    Args:
        df: DataFrame with columns ["#X", "#Y", "#Wave", "#Intensity"].

    Returns:
        x_values: Sorted unique X coordinates.
        y_values: Sorted unique Y coordinates.
        wave_values: Sorted unique wavenumbers.
        intensity_cube: 3D array with shape (nx, ny, nw).
    """
    x_values = np.sort(df["#X"].unique())
    y_values = np.sort(df["#Y"].unique())
    wave_values = np.sort(df["#Wave"].unique())

    nx = x_values.size
    ny = y_values.size
    nw = wave_values.size

    intensity_cube = np.full((nx, ny, nw), np.nan, dtype=float)

    index_map = {x: ix for ix, x in enumerate(x_values)}
    index_may = {y: iy for iy, y in enumerate(y_values)}
    wave_index = {w: iw for iw, w in enumerate(wave_values)}

    for _, row in df.iterrows():
        ix = index_map[row["#X"]]
        iy = index_may[row["#Y"]]
        iw = wave_index[row["#Wave"]]
        intensity_cube[ix, iy, iw] = row["#Intensity"]

    return x_values, y_values, wave_values, intensity_cube


def replace_nan_with_zero(arr: np.ndarray) -> np.ndarray:
    """Replace NaN values with zero in a NumPy array."""
    return np.where(np.isnan(arr), 0.0, arr)


def normalize_intensity_cube(
    intensity_cube: np.ndarray, reference_value: float | None = None
) -> tuple[np.ndarray, float]:
    """Normalize a hypercube so its maximum intensity becomes 1.

    Args:
        intensity_cube: 3D intensity cube with shape (nx, ny, nw).
        reference_value: Optional value to normalize by. If omitted,
            normalize by the cube's global maximum.

    Returns:
        normalized_cube: Intensity cube scaled to a maximum of 1.
        reference_value: The value used for normalization.
    """
    if reference_value is None:
        reference_value = float(np.nanmax(intensity_cube))
    if not np.isfinite(reference_value) or reference_value == 0:
        raise ValueError("Normalization reference value must be a finite non-zero number.")
    return intensity_cube / reference_value, reference_value


def normalize_intensity_cube_by_peak_window(
    intensity_cube: np.ndarray,
    wave: np.ndarray,
    target_peak: float = 338.0,
    window_width: float = 5.0,
) -> tuple[np.ndarray, float]:
    """Normalize the hypercube using the highest intensity in a peak window.

    Args:
        intensity_cube: 3D intensity cube with shape (nx, ny, nw).
        wave: 1D wavenumber axis for the cube.
        target_peak: Center wavenumber of the peak window.
        window_width: Half-width of the window in wavenumbers (cm^-1).

    Returns:
        normalized_cube: Cube scaled so the highest peak in the target window is 1.
        peak_value: Maximum intensity found in the selected window.
    """
    mask = (wave >= target_peak - window_width) & (wave <= target_peak + window_width)
    if not np.any(mask):
        raise ValueError(
            f"No spectral values found within {target_peak} +/- {window_width} cm^-1."
        )
    window = intensity_cube[:, :, mask]
    peak_value = float(np.nanmax(window))
    if not np.isfinite(peak_value) or peak_value == 0:
        raise ValueError(
            "Unable to normalize by peak window: no finite non-zero peak found."
        )
    return intensity_cube / peak_value, peak_value


def save_spectrum_files(df: pd.DataFrame, out_dir: Path) -> None:
    """Save a spectrum file for every unique (X, Y) location.

    Args:
        df: Parsed map DataFrame.
        out_dir: Directory where spectrum files will be written.
    """
    grouped = df.groupby(["#X", "#Y"])
    for (x, y), group in grouped:
        x_name = str(x).replace(".", "p").replace("-", "m")
        y_name = str(y).replace(".", "p").replace("-", "m")
        spectrum_file = out_dir / f"spec_x{x_name}_y{y_name}.txt"
        spectrum_df = group.sort_values("#Wave")[["#Wave", "#Intensity"]]
        spectrum_df.to_csv(
            spectrum_file,
            sep="\t",
            index=False,
            header=["Wave", "Intensity"],
            float_format="%.6f",
        )


def save_hypercube(
    out_dir: Path,
    x_values: np.ndarray,
    y_values: np.ndarray,
    wave_values: np.ndarray,
    intensity_cube: np.ndarray,
) -> None:
    """Save the parsed hypercube arrays to a .npz archive.

    Args:
        out_dir: Directory where the archive will be written.
        x_values: 1D array of X coordinates.
        y_values: 1D array of Y coordinates.
        wave_values: 1D array of wavenumbers.
        intensity_cube: 3D intensity array.
    """
    np.savez_compressed(
        out_dir / f"raman_hypercube_points_{x_values.size * y_values.size}.npz",
        x=x_values,
        y=y_values,
        wave=wave_values,
        intensity=intensity_cube,
    )


def save_summary(
    out_dir: Path,
    df: pd.DataFrame,
    x_values: np.ndarray,
    y_values: np.ndarray,
    wave_values: np.ndarray,
) -> None:
    """Write a plain-text summary of the parsed map.

    Args:
        out_dir: Directory where the summary file is saved.
        df: Parsed map DataFrame.
        x_values: Unique X coordinates.
        y_values: Unique Y coordinates.
        wave_values: Unique wavenumbers.
    """
    summary_file = out_dir / "summary.txt"
    summary_lines = [
        f"Parsed rows: {df.shape[0]}",
        f"Grid points: {len(x_values)} x {len(y_values)} = {len(x_values) * len(y_values)}",
        f"Wave points: {len(wave_values)}",
        f"X values: {', '.join(map(str, x_values))}",
        f"Y values: {', '.join(map(str, y_values))}",
        f"Wave range: {wave_values.min()} to {wave_values.max()}",
    ]
    summary_file.write_text("\n".join(summary_lines), encoding="utf-8")


def parse_and_save(txt_filename: Path, output_dir: Path | None = None) -> Path:
    """Parse a Raman map text file and save structured outputs.

    Args:
        txt_filename: Path to the input text file.
        output_dir: Optional base directory for outputs. If omitted,
            outputs are created next to the input file.

    Returns:
        Path to the created output directory.
    """
    txt_filename = txt_filename.resolve()
    base_dir = output_dir.resolve() if output_dir else txt_filename.parent
    out_dir = create_output_dirs(base_dir)

    df = parse_map_file(txt_filename)
    x_values, y_values, wave_values, intensity_cube = build_hypercube(df)
    #save_spectrum_files(df, out_dir)
    save_hypercube(out_dir, x_values, y_values, wave_values, intensity_cube)
    save_summary(out_dir, df, x_values, y_values, wave_values)

    return out_dir


def load_hypercube(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a saved Raman hypercube archive from a .npz file.

    Args:
        npz_path: Path to the .npz archive created by save_hypercube.

    Returns:
        Tuple of (x, y, wave, intensity).
    """
    with np.load(npz_path) as data:
        x = data["x"]
        y = data["y"]
        wave = data["wave"]
        intensity = data["intensity"]
    return x, y, wave, intensity

# Additional utility functions for peak mapping and plotting

def get_peak_window(wave, intensity_cube, target_peak, search_width):
    """Plotting utilities for Raman hypercube peak maps."""

    peak_idx = int(np.abs(wave - target_peak).argmin())
    lo = max(0, peak_idx - search_width)
    hi = min(wave.size, peak_idx + search_width + 1)
    return peak_idx, lo, hi, intensity_cube[:, :, lo:hi]

def compute_peak_map(
    wave: np.ndarray,
    intensity_cube: np.ndarray,
    target_peak: float,
    search_width: int = 3,
) -> tuple[np.ndarray, float, int, int]:
    """Compute the maximum intensity map around a target wavenumber.

    Args:
        wave: 1D array of wavenumber values.
        intensity_cube: 3D intensity array with shape (nx, ny, nw).
        target_peak: Target peak wavenumber to map.
        search_width: Half-width of the search window in wavenumber indices.

    Returns:
        peak_map: 2D intensity map.
        matched_wave: Wavenumber value closest to the target peak.
        lo: Lower index of the matched window.
        hi: Upper index of the matched window.
    """
    peak_idx = int(np.abs(wave - target_peak).argmin())
    lo = max(0, peak_idx - search_width)
    hi = min(wave.size, peak_idx + search_width + 1)
    window = intensity_cube[:, :, lo:hi]
    masked = np.where(np.isnan(window), -np.inf, window)
    peak_map = np.max(masked, axis=-1)
    empty = np.all(np.isnan(window), axis=-1)
    peak_map = np.where(empty, np.nan, peak_map)

    return peak_map, wave[peak_idx], lo, hi


def _compute_grid_edges(values: np.ndarray) -> np.ndarray:
    """Compute grid edges from center values for pcolormesh plotting."""
    diffs = np.diff(values) / 2.0
    left_edge = values[0] - diffs[0]
    right_edge = values[-1] + diffs[-1]
    edges = np.concatenate(([left_edge], values[:-1] + diffs, [right_edge]))
    return edges


def _gaussian_kernel(sigma: float) -> np.ndarray:
    radius = max(1, int(3 * sigma))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def _convolve_2d(arr: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    result = np.empty_like(arr, dtype=float)
    if axis == 0:
        for j in range(arr.shape[1]):
            result[:, j] = np.convolve(arr[:, j], kernel, mode="same")
    else:
        for i in range(arr.shape[0]):
            result[i, :] = np.convolve(arr[i, :], kernel, mode="same")
    return result


def smooth_map(intensity_map: np.ndarray, sigma: float | None = None) -> np.ndarray:
    """Apply Gaussian smoothing to a 2D intensity map.

    Args:
        intensity_map: 2D map of intensities.
        sigma: Standard deviation of the Gaussian kernel in pixels.

    Returns:
        Smoothed 2D intensity map, preserving NaN pixels.
    """
    if sigma is None or sigma <= 0:
        return intensity_map
    kernel = _gaussian_kernel(sigma)
    valid = ~np.isnan(intensity_map)
    data = np.where(valid, intensity_map, 0.0)
    smoothed = _convolve_2d(data, kernel, axis=0)
    smoothed = _convolve_2d(smoothed, kernel, axis=1)
    weight = _convolve_2d(valid.astype(float), kernel, axis=0)
    weight = _convolve_2d(weight, kernel, axis=1)
    return np.where(weight > 1e-8, smoothed / weight, np.nan)



## Preprocessing function

def map_normalize(peak_map: np.ndarray) -> np.ndarray:
    """Normalize a peak map to the range [0, 1] for better visualization."""
    min_val = np.nanmin(peak_map)
    max_val = np.nanmax(peak_map)
    if max_val > min_val:
        return (peak_map - min_val) / (max_val - min_val)
    else:
        return np.zeros_like(peak_map)
    

def parse_map_file(txt_filename: Path) -> pd.DataFrame:
    """Read a Raman map text file into a DataFrame."""
    df = pd.read_csv(txt_filename, sep=r"\t+", engine="python")
    expected = ["#X", "#Y", "#Wave", "#Intensity"]
    if list(df.columns) != expected:
        raise ValueError(f"Unexpected columns: {list(df.columns)}; expected {expected}")
    return df


def create_output_dirs(base_dir: Path) -> Path:
    """Create the directory for the parsed spectra files."""
    out_dir = base_dir / "parsed_spectra_100_points"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def build_hypercube(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build X/Y grid and intensity hypercube from the DataFrame."""
    x_values = np.sort(df["#X"].unique())
    y_values = np.sort(df["#Y"].unique())
    wave_values = np.sort(df["#Wave"].unique())

    nx = x_values.size
    ny = y_values.size
    nw = wave_values.size

    intensity_cube = np.full((nx, ny, nw), np.nan, dtype=float)

    index_map = {x: ix for ix, x in enumerate(x_values)}
    index_may = {y: iy for iy, y in enumerate(y_values)}
    wave_index = {w: iw for iw, w in enumerate(wave_values)}

    for _, row in df.iterrows():
        ix = index_map[row["#X"]]
        iy = index_may[row["#Y"]]
        iw = wave_index[row["#Wave"]]
        intensity_cube[ix, iy, iw] = row["#Intensity"]

    return x_values, y_values, wave_values, intensity_cube


def save_spectrum_files(df: pd.DataFrame, out_dir: Path) -> None:
    """Save each location's spectrum as a separate text file."""
    grouped = df.groupby(["#X", "#Y"])
    for (x, y), group in grouped:
        x_name = str(x).replace(".", "p").replace("-", "m")
        y_name = str(y).replace(".", "p").replace("-", "m")
        spectrum_file = out_dir / f"spec_x{x_name}_y{y_name}.txt"
        spectrum_df = group.sort_values("#Wave")[["#Wave", "#Intensity"]]
        spectrum_df.to_csv(
            spectrum_file,
            sep="\t",
            index=False,
            header=["Wave", "Intensity"],
            float_format="%.6f",
        )


def save_hypercube(
    out_dir: Path,
    x_values: np.ndarray,
    y_values: np.ndarray,
    wave_values: np.ndarray,
    intensity_cube: np.ndarray,
) -> None:
    """Save the hypercube as a compressed NumPy file with metadata."""
    np.savez_compressed(
        out_dir / f"raman_hypercube_points_{len(x_values) * len(y_values)}.npz",
        x=x_values,
        y=y_values,
        wave=wave_values,
        intensity=intensity_cube,
    )


def save_summary(
    out_dir: Path,
    df: pd.DataFrame,
    x_values: np.ndarray,
    y_values: np.ndarray,
    wave_values: np.ndarray,
) -> None:
    """Save a summary file describing the map and the hypercube."""
    summary_file = out_dir / "summary.txt"
    summary_lines = [
        f"Parsed rows: {df.shape[0]}",
        f"Grid points: {len(x_values)} x {len(y_values)} = {len(x_values) * len(y_values)}",
        f"Wave points: {len(wave_values)}",
        f"X values: {', '.join(map(str, x_values))}",
        f"Y values: {', '.join(map(str, y_values))}",
        f"Wave range: {wave_values.min()} to {wave_values.max()}",
        f"Intensity range: {df['#Intensity'].min()} to {df['#Intensity'].max()}",
    ]
    summary_file.write_text("\n".join(summary_lines), encoding="utf-8")


def parse_and_save(
    txt_filename: Path,
    output_dir: Path | None = None,
    nan_to_zero: bool = False,
    normalize: bool = False,
    normalize_peak: float | None = None,
    normalize_width: float = 5.0,
) -> Path:
    """Parse the input text file and save spectra and hypercube outputs."""
    txt_filename = txt_filename.resolve()
    base_dir = output_dir.resolve() if output_dir else txt_filename.parent
    out_dir = create_output_dirs(base_dir)

    df = parse_map_file(txt_filename)
    x_values, y_values, wave_values, intensity_cube = build_hypercube(df)
    if nan_to_zero:
        intensity_cube = replace_nan_with_zero(intensity_cube)
    if normalize or normalize_peak is not None:
        if normalize_peak is None:
            normalize_peak = 338.0
        intensity_cube, reference_value = normalize_intensity_cube_by_peak_window(
            intensity_cube,
            wave_values,
            normalize_peak,
            normalize_width,
        )
        low = normalize_peak - normalize_width
        high = normalize_peak + normalize_width
        print(
            f"Normalized hypercube using CZTS reference window {low:.1f}-{high:.1f} cm^-1; "
            f"reference peak value = {reference_value:.6g}"
        )
    #save_spectrum_files(df, out_dir)
    save_hypercube(out_dir, x_values, y_values, wave_values, intensity_cube)
    save_summary(out_dir, df, x_values, y_values, wave_values)

    return out_dir

def load_npz_as_dict(npz_path: Path) -> dict[str, Any]:
    """Load an NPZ file and return its contents as a dictionary.

    Args:
        npz_path: Path to the .npz file.

    Returns:
        Dictionary with array names as keys and numpy arrays as values.
    """
    with np.load(npz_path) as data:
        # Convert the NpzFile object to a regular dictionary
        return {key: data[key] for key in data.files}


def load_npz_as_dict_nantozero(npz_path: Path) -> dict[str, Any]:
    """Load an NPZ file and convert all NaN values to zero.

    Args:
        npz_path: Path to the .npz file.

    Returns:
        Dictionary with NaN values replaced by 0.
    """
    result = {}
    with np.load(npz_path) as data:
        for key in data.files:
            arr = data[key].copy()
            # Replace NaN with 0
            arr = np.where(np.isnan(arr), 0, arr)
            result[key] = arr
    return result


def save_npz_dict(output_path: Path, data_dict: dict[str, np.ndarray]) -> None:
    """Save a dictionary of arrays to an NPZ file.

    Args:
        output_path: Path where the .npz file will be saved.
        data_dict: Dictionary with array names and numpy arrays.
    """
    np.savez_compressed(output_path, **data_dict)


def replace_nan_in_npz(input_path: Path, output_path: Path) -> None:
    """Read an NPZ file, replace all NaN with 0, and save to a new file.

    Args:
        input_path: Path to the original .npz file.
        output_path: Path where the modified .npz file will be saved.
    """
    data_dict = load_npz_as_dict_nantozero(input_path)
    save_npz_dict(output_path, data_dict)


def main():
    parser = argparse.ArgumentParser(
        description="Plot a spatial peak intensity map from a Raman hypercube npz file."
    )
    parser.add_argument("input_npz", type=Path, help="Path to the hypercube npz file.")
    parser.add_argument(
        "--peak", type=float, required=True, help="Target peak wavenumber to map."
    )
    parser.add_argument(
        "--search-width",
        type=int,
        default=3,
        help="Half-width of the wave index window used around the target peak.",
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=0.0,
        help="Gaussian smoothing sigma in pixels for the spatial peak map.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output PNG path for the peak map.",
    )
    args = parser.parse_args()
    x, y, wave, intensity = load_hypercube(args.input_npz)
    peak_map, matched_wave, lo, hi = compute_peak_map(
        wave, intensity, args.peak, args.search_width
    )
    if args.smooth_sigma and args.smooth_sigma > 0:
        peak_map = smooth_map(peak_map, args.smooth_sigma)
        title = (
            f"Peak map at {args.peak:.1f} cm^-1 (smoothed σ={args.smooth_sigma:.1f})"
        )
    else:
        title = f"Peak map at {args.peak:.1f} cm^-1 (matched {matched_wave:.1f} cm^-1)"
    if args.output is None:
        output_dir = args.input_npz.parent
        output_path = output_dir / f"peak_map_{int(args.peak)}_cm-1.png"
    else:
        output_path = args.output
    plot_peak_map(x, y, peak_map, title, output_path=output_path)
    print(f"Saved peak intensity map to: {output_path}")
    plt.show()
