"""Analysis pipeline using ramanspy for baseline correction, denoising, and normalization."""

import argparse
from pathlib import Path
import sys
import numpy as np
from typing import Tuple

try:
    import ramanspy
except ImportError:
    ramanspy = None


def check_ramanspy():
    """Verify ramanspy is installed."""
    if ramanspy is None:
        raise ImportError(
            "ramanspy is required for preprocessing. "
            "Install it with: python -m pip install ramanspy"
        )


def load_hypercube(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a hypercube from .npz file.

    Args:
        npz_path: Path to the .npz archive.

    Returns:
        Tuple of (x, y, wave, intensity).
    """
    with np.load(npz_path) as data:
        x = data["x"]
        y = data["y"]
        wave = data["wave"]
        intensity = data["intensity"]
    return x, y, wave, intensity


def save_hypercube(
    output_path: Path,
    x: np.ndarray,
    y: np.ndarray,
    wave: np.ndarray,
    intensity: np.ndarray,
) -> None:
    """Save a hypercube to .npz file.

    Args:
        output_path: Path where the .npz file will be saved.
        x: X coordinates.
        y: Y coordinates.
        wave: Wavenumber axis.
        intensity: 3D intensity cube.
    """
    np.savez_compressed(
        output_path,
        x=x,
        y=y,
        wave=wave,
        intensity=intensity,
    )

def baseline_correct_als(
    intensity_cube: np.ndarray,
    lam: float = 1e4,
    p: float = 0.01,
) -> np.ndarray:
    """Apply Asymmetric Least Squares baseline correction using ramanspy.

    Args:
        intensity_cube: 3D intensity array (nx, ny, nw).
        lam: Smoothness parameter. Higher = smoother baseline.
        p: Asymmetry parameter. Lower = more sensitive to peaks.

    Returns:
        Baseline-corrected intensity cube.
    """
    check_ramanspy()
    print("  Applying ALS baseline correction...")
    nx, ny, nw = intensity_cube.shape
    corrected = np.zeros_like(intensity_cube)

    for i in range(nx):
        for j in range(ny):
            spectrum = ramanspy.Spectrum(intensity_cube[i, j, :], np.arange(nw))
            if np.any(np.isfinite(spectrum.spectral_data)):
                try:
                    pipe = ramanspy.preprocessing.baseline.ASLS(lam=lam, p=p)
                    corrected[i, j, :] = pipe.apply(spectrum).spectral_data
                except Exception as e:
                    print(f"  Warning: baseline correction failed at ({i}, {j}): {e}")
                    corrected[i, j, :] = spectrum.spectral_data
            else:
                corrected[i, j, :] = spectrum.spectral_data
        if (i + 1) % max(1, nx // 10) == 0:
            print(f"    Processed {i + 1}/{nx} rows")

    return corrected


def baseline_correct_ials(
    intensity_cube: np.ndarray,
    niter: int = 20,
) -> np.ndarray:
    """Apply IALS (Improved Asymmetric Least Squares) baseline correction.
    Args:
        intensity_cube: 3D intensity array (nx, ny, nw).
        niter: Number of iterations.

    Returns:
        Baseline-corrected intensity cube.
    """
    check_ramanspy()
    print("  Applying IALS baseline correction...")
    nx, ny, nw = intensity_cube.shape
    corrected = np.zeros_like(intensity_cube)

    for i in range(nx):
        for j in range(ny):
            spectrum = ramanspy.Spectrum(intensity_cube[i, j, :], np.arange(nw))
            if np.any(np.isfinite(spectrum.spectral_data)):
                try:
                    pipe = ramanspy.preprocessing.baseline.IASLS(niter=niter)
                    corrected[i, j, :] = pipe.apply(spectrum).spectral_data
                except Exception as e:
                    print(f"  Warning: baseline correction failed at ({i}, {j}): {e}")
                    corrected[i, j, :] = spectrum.spectral_data
            else:
                corrected[i, j, :] = spectrum.spectral_data
        if (i + 1) % max(1, nx // 10) == 0:
            print(f"    Processed {i + 1}/{nx} rows")

    return corrected


def denoise_savgol(
    intensity_cube: np.ndarray,
    window_length: int = 5,
    polyorder: int = 2,
) -> np.ndarray:
    """Apply Savitzky-Golay denoising using ramanspy.

    Args:
        intensity_cube: 3D intensity array (nx, ny, nw).
        window_length: Length of the filter window (must be odd).
        polyorder: Order of the polynomial filter.

    Returns:
        Denoised intensity cube.
    """
    check_ramanspy()
    print("  Applying Savitzky-Golay denoising...")
    nx, ny, nw = intensity_cube.shape
    denoised = np.zeros_like(intensity_cube)

    for i in range(nx):
        for j in range(ny):
            spectrum = ramanspy.Spectrum(intensity_cube[i, j, :], np.arange(nw))
            if np.any(np.isfinite(spectrum.spectral_data)):
                try:
                    pipe = ramanspy.preprocessing.denoise.SavGol(
                        window_length=window_length, polyorder=polyorder
                    )
                    denoised[i, j, :] = pipe.apply(spectrum).spectral_data
                except Exception as e:
                    print(f"  Warning: denoising failed at ({i}, {j}): {e}")
                    denoised[i, j, :] = spectrum.spectral_data
            else:
                denoised[i, j, :] = spectrum.spectral_data
        if (i + 1) % max(1, nx // 10) == 0:
            print(f"    Processed {i + 1}/{nx} rows")

    return denoised


def denoise_gaussian(
    intensity_cube: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    """Apply Gaussian denoising using ramanspy.

    Args:
        intensity_cube: 3D intensity array (nx, ny, nw).
        sigma: Standard deviation of the Gaussian kernel.

    Returns:
        Denoised intensity cube.
    """
    check_ramanspy()
    print("  Applying Gaussian denoising...")
    nx, ny, nw = intensity_cube.shape
    denoised = np.zeros_like(intensity_cube)

    for i in range(nx):
        for j in range(ny):
            spectrum = ramanspy.Spectrum(intensity_cube[i, j, :], np.arange(nw))
            if np.any(np.isfinite(spectrum.spectral_data)):
                try:
                    pipe = ramanspy.preprocessing.denoise.Gaussian(sigma=sigma)
                    denoised[i, j, :] = pipe.apply(spectrum).spectral_data
                except Exception as e:
                    print(f"  Warning: denoising failed at ({i}, {j}): {e}")
                    denoised[i, j, :] = spectrum.spectral_data
            else:
                denoised[i, j, :] = spectrum.spectral_data
        if (i + 1) % max(1, nx // 10) == 0:
            print(f"    Processed {i + 1}/{nx} rows")

    return denoised


def remove_cosmic_rays(
    intensity_cube: np.ndarray,
    threshold: float = 3.0,
) -> np.ndarray:
    """Remove cosmic rays and spikes using ramanspy.

    Args:
        intensity_cube: 3D intensity array (nx, ny, nw).
        threshold: Threshold multiplier for spike detection.

    Returns:
        Intensity cube with cosmic rays removed.
    """
    check_ramanspy()
    print("  Removing cosmic rays...")
    nx, ny, nw = intensity_cube.shape
    cleaned = np.zeros_like(intensity_cube)

    for i in range(nx):
        for j in range(ny):
            spectrum = ramanspy.Spectrum(intensity_cube[i, j, :], np.arange(nw))  # Create a Spectrum object for processing
            #print(f"  Processing spectrum at ({i}, {j}) with shape {spectrum.spectral_data}...")
            if np.any(np.isfinite(spectrum.spectral_data)):  # Check if there are any finite values in the intensity
                try:
                    pipe = ramanspy.preprocessing.despike.WhitakerHayes(threshold=threshold)
                    print(f"  Removing cosmic rays at ({i}, {j})...")
                    #print(spectrum.spectral_data)
                    cleaned[i, j, :] = pipe.apply(spectrum).spectral_data
                    #print(f"  Cosmic ray removal successful at ({i}, {j}).")

                except Exception as e:
                    print(f"  Warning: cosmic ray removal failed at ({i}, {j}): {e}")
                    cleaned[i, j, :] = spectrum.spectral_data
                    sys.exit(1)  # Exit the program if cosmic ray removal fails, as this is a critical step
            else:
                cleaned[i, j, :] = spectrum.spectral_data  # If no finite values, keep the original spectrum (which may be all NaN or zeros)
        if (i + 1) % max(1, nx // 10) == 0:
            print(f"    Processed {i + 1}/{nx} rows")

    return cleaned


def spectral_crop(
    intensity_cube: np.ndarray,
    wave: np.ndarray,
    wave_min: float,
    wave_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop the spectral range to keep only wavenumbers in [wave_min, wave_max].

    Args:
        intensity_cube: 3D intensity array (nx, ny, nw).
        wave: 1D wavenumber axis.
        wave_min: Minimum wavenumber to keep.
        wave_max: Maximum wavenumber to keep.

    Returns:
        Cropped (intensity_cube, wave).
    """
    print(f"  Cropping spectral range to {wave_min}-{wave_max} cm^-1...")
    mask = (wave >= wave_min) & (wave <= wave_max)
    if not np.any(mask):
        raise ValueError(f"No spectral values found in range {wave_min}-{wave_max} cm^-1.")
    return intensity_cube[:, :, mask], wave[mask]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess a Raman hypercube using ramanspy. "
            "Applies baseline correction, denoising, and cosmic ray removal."
        )
    )
    parser.add_argument(
        "input_npz",
        type=Path,
        help="Path to the input hypercube .npz file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output path for the processed hypercube. Default: input_basename_processed.npz",
    )
    parser.add_argument(
        "--baseline",
        choices=["als", "snip", "none"],
        default="none",
        help="Baseline correction method.",
    )
    parser.add_argument(
        "--baseline-lam",
        type=float,
        default=1e4,
        help="Smoothness parameter for ALS baseline (higher = smoother).",
    )
    parser.add_argument(
        "--baseline-p",
        type=float,
        default=0.01,
        help="Asymmetry parameter for ALS baseline (lower = more sensitive to peaks).",
    )
    parser.add_argument(
        "--denoise",
        choices=["savgol", "gaussian", "none"],
        default="none",
        help="Denoising method.",
    )
    parser.add_argument(
        "--denoise-window",
        type=int,
        default=5,
        help="Window length for Savitzky-Golay filter (must be odd).",
    )
    parser.add_argument(
        "--denoise-order",
        type=int,
        default=2,
        help="Polynomial order for Savitzky-Goyal filter.",
    )
    parser.add_argument(
        "--denoise-sigma",
        type=float,
        default=1.0,
        help="Sigma for Gaussian denoising.",
    )
    parser.add_argument(
        "--cosmic-rays",
        action="store_true",
        help="Remove cosmic rays and spikes.",
    )
    parser.add_argument(
        "--cosmic-rays-threshold",
        type=float,
        default=3.0,
        help="Threshold multiplier for cosmic ray detection.",
    )
    parser.add_argument(
        "--crop-min",
        type=float,
        default=None,
        help="Minimum wavenumber to keep (cm^-1).",
    )
    parser.add_argument(
        "--crop-max",
        type=float,
        default=None,
        help="Maximum wavenumber to keep (cm^-1).",
    )

    args = parser.parse_args()

    print(f"Loading hypercube from: {args.input_npz}")
    x, y, wave, intensity = load_hypercube(args.input_npz)
    print(f"  Shape: {intensity.shape}")
    print(f"  Spectral range: {wave[0]:.1f}-{wave[-1]:.1f} cm^-1")
    print()

    # Baseline correction
    if args.baseline != "none":
        print(f"Step 1: Baseline correction ({args.baseline})")
        if args.baseline == "als":
            intensity = baseline_correct_als(
                intensity, lam=args.baseline_lam, p=args.baseline_p
            )
        elif args.baseline == "snip":
            intensity = baseline_correct_snip(intensity)
        print()

    # Cosmic ray removal
    if args.cosmic_rays:
        print("Step 2: Cosmic ray removal")
        intensity = remove_cosmic_rays(intensity, threshold=args.cosmic_rays_threshold)
        print()

    # Denoising
    if args.denoise != "none":
        print(f"Step 3: Denoising ({args.denoise})")
        if args.denoise == "savgol":
            intensity = denoise_savgol(
                intensity,
                window_length=args.denoise_window,
                polyorder=args.denoise_order,
            )
        elif args.denoise == "gaussian":
            intensity = denoise_gaussian(intensity, sigma=args.denoise_sigma)
        print()

    # Spectral cropping
    if args.crop_min is not None or args.crop_max is not None:
        crop_min = args.crop_min if args.crop_min is not None else wave[0]
        crop_max = args.crop_max if args.crop_max is not None else wave[-1]
        print("Step 4: Spectral cropping")
        intensity, wave = spectral_crop(intensity, wave, crop_min, crop_max)
        print(f"  New spectral range: {wave[0]:.1f}-{wave[-1]:.1f} cm^-1")
        print()

    # Save output
    if args.output is None:
        stem = args.input_npz.stem
        args.output = args.input_npz.parent / f"{stem}_processed.npz"

    print(f"Saving processed hypercube to: {args.output}")
    save_hypercube(args.output, x, y, wave, intensity)
    print(f"  Shape: {intensity.shape}")
    print("Done!")


if __name__ == "__main__":
    main()
