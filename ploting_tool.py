import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import json
#import tempfile
import webbrowser
from utils import *

def compute_peak_map(wave, intensity_cube, target_peak, search_width=3):
    """Return a 2D map of peak intensity for the given target peak."""
    peak_idx, lo, hi, window = get_peak_window(
        wave, intensity_cube, target_peak, search_width
    )
    masked = np.where(np.isnan(window), -np.inf, window)
    peak_map = np.max(masked, axis=-1)
    empty = np.all(np.isnan(window), axis=-1)
    peak_map = np.where(empty, np.nan, peak_map)
    return peak_map, wave[peak_idx], lo, hi


def compute_peak_to_base_ratio_map(wave, intensity_cube, target_peak, search_width=3):
    """Return a 2D map of peak-to-base ratio for the given target peak.

    The peak and base are calculated from the same local window used by
    compute_peak_map: peak=max intensity, base=min intensity.
    """
    peak_idx, lo, hi, window = get_peak_window(
        wave, intensity_cube, target_peak, search_width
    )
    masked_peak = np.where(np.isnan(window), -np.inf, window)
    masked_base = np.where(np.isnan(window), np.inf, window)
    peak_map = np.max(masked_peak, axis=-1)
    base_map = np.min(masked_base, axis=-1)
    empty = np.all(np.isnan(window), axis=-1)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_map = peak_map / base_map

    invalid_base = ~np.isfinite(base_map) | (base_map <= 0)
    invalid_ratio = ~np.isfinite(ratio_map)
    ratio_map = np.where(empty | invalid_base | invalid_ratio, np.nan, ratio_map)
    return ratio_map, wave[peak_idx], lo, hi


def _compute_grid_edges(values):
    diffs = np.diff(values) / 2.0
    left_edge = values[0] - diffs[0]
    right_edge = values[-1] + diffs[-1]
    edges = np.concatenate(([left_edge], values[:-1] + diffs, [right_edge]))
    return edges


def plot_peak_map(
    x,
    y,
    intensity_map,
    title,
    cmap="viridis",
    output_path=None,
    colorbar_label="Peak intensity",
    wave=None,
    intensity_cube=None,
    peak_window=None,
    matched_wave=None,
):
    x_edges = _compute_grid_edges(x)
    y_edges = _compute_grid_edges(y)
    show_peak_subplot = (
        wave is not None
        and intensity_cube is not None
        and peak_window is not None
        and matched_wave is not None
    )

    if show_peak_subplot:
        fig, (ax, peak_ax) = plt.subplots(
            1,
            2,
            figsize=(12, 5),
            constrained_layout=True,
            gridspec_kw={"width_ratios": [1.0, 1.15], "wspace": 0.3},
        )
    else:
        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        peak_ax = None

    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        intensity_map.T,
        cmap=cmap,
        shading="auto",
        vmin=np.nanmin(intensity_map),
        vmax=np.nanmax(intensity_map),
    )
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_title(title)
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label(colorbar_label)
    ax.set_aspect("equal")
    ax.invert_xaxis()

    if show_peak_subplot:
        lo, hi = peak_window
        peak_wave = wave[lo:hi]
        peak_spectra = intensity_cube[:, :, lo:hi][np.isfinite(intensity_map)]
        valid_rows = np.any(np.isfinite(peak_spectra), axis=1)
        peak_spectra = peak_spectra[valid_rows]

        if peak_spectra.size:
            base_values = np.nanmin(peak_spectra, axis=1)
            valid_base = np.isfinite(base_values) & (base_values > 0)
            ratio_spectra = peak_spectra[valid_base] / base_values[valid_base, None]

        if peak_spectra.size and ratio_spectra.size:
            mean_spectrum = np.nanmean(ratio_spectra, axis=0)
            std_spectrum = np.nanstd(ratio_spectra, axis=0)
            mean_peak_ratio = np.nanmax(mean_spectrum)

            peak_ax.plot(
                peak_wave,
                ratio_spectra.T,
                color="tab:blue",
                alpha=0.08,
                linewidth=0.8,
            )
            peak_ax.fill_between(
                peak_wave,
                mean_spectrum - std_spectrum,
                mean_spectrum + std_spectrum,
                color="tab:blue",
                alpha=0.18,
                linewidth=0,
                label="Mean +/- std",
            )
            peak_ax.plot(
                peak_wave,
                mean_spectrum,
                color="black",
                linewidth=2.0,
                label=f"Mean peak/base = {mean_peak_ratio:.2f}",
            )
            peak_ax.axvline(
                matched_wave,
                color="tab:red",
                linestyle="--",
                linewidth=1.2,
                label=f"Matched {matched_wave:.1f}",
            )
            peak_ax.legend(frameon=False, fontsize=8)

        peak_ax.set_xlabel("Wavenumber (cm^-1)")
        peak_ax.set_ylabel("Intensity / base")
        peak_ax.set_title("Peak-to-base profile")

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig, ax


def plot_peak_map_plotly(
    x,
    y,
    intensity_map,
    title,
    colorbar_label="Peak intensity",
    output_path: Path | None = None,
):
    try:
        import plotly.express as px
    except ImportError as exc:
        raise ImportError(
            "Plotly is required for interactive plotting. Install it with `pip install plotly`."
        ) from exc

    fig = px.imshow(
        intensity_map.T,
        x=x,
        y=y,
        origin="lower",
        aspect="equal",
        color_continuous_scale="viridis",
        labels={"x": "X position", "y": "Y position", "color": colorbar_label},
    )
    fig.update_layout(title=title)
    fig.update_xaxes(tickmode="linear")
    fig.update_yaxes(tickmode="linear")
    if output_path is not None:
        if output_path.suffix.lower() == ".html":
            fig.write_html(str(output_path))
        else:
            fig.write_image(str(output_path))
    else:
        fig.show()
    return fig


def plot_peak_dashboard(
    x,
    y,
    wave,
    intensity_cube,
    map_data,
    target_point: tuple[int, int] | None = None,
    title: str = "Raman intensity dashboard",
    colorbar_label: str = "Peak intensity",
    output_path: Path | None = None,
):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError(
            "Plotly is required for the dashboard. Install it with `pip install plotly`."
        ) from exc

    heatmap_data = np.nan_to_num(map_data, nan=0.0)
    if target_point is None:
        valid = np.argwhere(np.isfinite(map_data))
        target_point = tuple(valid[0]) if valid.size else (0, 0)

    x_idx, y_idx = target_point
    z = heatmap_data.T
    customdata = np.stack(
        np.meshgrid(np.arange(x.size), np.arange(y.size), indexing="xy"), axis=-1
    )
    initial_spectrum = intensity_cube[x_idx, y_idx, :]
    initial_spectrum = np.nan_to_num(initial_spectrum, nan=0.0)

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.55, 0.45],
        subplot_titles=["Spatial peak intensity", "Spectrum at selected pixel"],
    )
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=x,
            y=y,
            customdata=customdata,
            hovertemplate="X=%{x}<br>Y=%{y}<br>Intensity=%{z:.3f}<extra></extra>",
            colorbar=dict(title=colorbar_label),
            colorscale="Viridis",
            #zsmooth="best",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=wave,
            y=initial_spectrum,
            mode="lines",
            line=dict(color="firebrick"),
            name=f"Pixel ({x[x_idx]}, {y[y_idx]})",
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="X position", row=1, col=1)
    fig.update_yaxes(title_text="Y position", row=1, col=1)
    fig.update_xaxes(title_text="Wavenumber (cm^-1)", row=1, col=2)
    fig.update_yaxes(title_text="Intensity", row=1, col=2)
    fig.update_layout(title=title, height=600, width=1200)

    if output_path is not None:
        output_path = Path(output_path)
        if output_path.suffix.lower() != ".html":
            output_path = output_path.with_suffix(".html")

    wave_json = json.dumps(wave.tolist())
    intensity_json = json.dumps(intensity_cube.astype(float).tolist())
    js = f"""
    window.addEventListener('DOMContentLoaded', function() {{
        var plotDiv = document.getElementById('plotly-dashboard');
        var wave = {wave_json};
        var intensity = {intensity_json};
        plotDiv.on('plotly_click', function(data) {{
            var pt = data.points[0];
            var idx = pt.customdata;
            var x_i = idx[0];
            var y_i = idx[1];
            var newY = intensity[x_i][y_i];
            Plotly.restyle(plotDiv, {{ y: [newY], x: [wave] }}, [1]);
            var annText = 'Selected pixel X=' + pt.x + ', Y=' + pt.y;
            Plotly.relayout(plotDiv, {{ 'annotations[0].text': annText }});
        }});
    }});
    """
    html = fig.to_html(
        include_plotlyjs="cdn",
        full_html=True,
        div_id="plotly-dashboard",
        post_script=js,
    )
    if output_path is not None:
        output_path = output_path.resolve()
        with open(output_path, "w", encoding="utf-8") as html_file:
            html_file.write(html)
        webbrowser.open(output_path.as_uri())
    else:
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as tmpf:
            tmpf.write(html)
            tmp_path = Path(tmpf.name)
        webbrowser.open(tmp_path.as_uri())
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot a spatial peak map from a Raman hypercube npz file."
    )
    parser.add_argument("input_npz", type=Path, help="Path to the hypercube npz file.")
    parser.add_argument(
        "--peak", type=float, required=True, help="Target peak wavenumber to map."
    )
    parser.add_argument(
        "--map-type",
        choices=("intensity", "peak-to-base-ratio"),
        default="intensity",
        help="Map peak intensity or peak-to-base ratio.",
    )
    parser.add_argument(
        "--search-width",
        type=int,
        default=5,
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
        help="Optional output path for the peak map. Use .html for interactive output.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Generate an interactive Plotly HTML map instead of a static Matplotlib plot.",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Generate a Plotly dashboard with heatmap and selected spectrum.",
    )
    args = parser.parse_args()

    x, y, wave, intensity = load_hypercube(args.input_npz)
    if args.map_type == "peak-to-base-ratio":
        map_data, matched_wave, lo, hi = compute_peak_to_base_ratio_map(
            wave, intensity, args.peak, args.search_width
        )
        map_name = "Peak-to-base ratio map"
        colorbar_label = "Peak / base ratio"
        output_prefix = "peak_to_base_ratio_map"
        saved_label = "peak-to-base ratio map"
    else:
        map_data, matched_wave, lo, hi = compute_peak_map(
            wave, intensity, args.peak, args.search_width
        )
        map_name = "Peak map"
        colorbar_label = "Peak intensity"
        output_prefix = "peak_map"
        saved_label = "peak intensity map"

    if args.smooth_sigma and args.smooth_sigma > 0:
        map_data = smooth_map(map_data, args.smooth_sigma)
        title = (
            f"{map_name} at {args.peak:.1f} cm^-1 "
            f"(smoothed sigma={args.smooth_sigma:.1f})"
        )
    else:
        title = (
            f"{map_name} at {args.peak:.1f} cm^-1 "
            f"(matched {matched_wave:.1f} cm^-1)"
        )

    if args.output is None:
        output_dir = args.input_npz.parent
        output_path = output_dir / f"{output_prefix}_{int(args.peak)}_cm-1.png"
    else:
        output_path = args.output

    if args.dashboard:
        dashboard_path = args.output if args.output is not None else None
        plot_peak_dashboard(
            x,
            y,
            wave,
            intensity,
            map_data,
            title=title,
            colorbar_label=colorbar_label,
            output_path=dashboard_path,
        )
        if dashboard_path is not None:
            print(f"Saved dashboard {saved_label} to: {dashboard_path}")
        else:
            print(f"Opened dashboard {saved_label} in the default browser.")
    elif args.interactive:
        if output_path is None:
            output_path = output_dir / f"{output_prefix}_{int(args.peak)}_cm-1.html"
        plot_peak_map_plotly(
            x,
            y,
            map_data,
            title,
            colorbar_label=colorbar_label,
            output_path=output_path,
        )
        print(f"Saved interactive {saved_label} to: {output_path}")
    else:
        plot_peak_map(
            x,
            y,
            map_data,
            title,
            output_path=output_path,
            colorbar_label=colorbar_label,
            wave=wave,
            intensity_cube=intensity,
            peak_window=(lo, hi),
            matched_wave=matched_wave,
        )
        print(f"Saved {saved_label} to: {output_path}")
        plt.show()


if __name__ == "__main__":
    main()
