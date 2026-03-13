"""Prepare NSA dataset and generate corner plot (Fig 1) and nsa.pkl for subsequent scripts."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

import corner


VAR_NAMES = [
    "ZDIST",
    "ABSMAG",
    "log(B300)",
    "log(MASS)",
    "SERSIC_N",
    "ELPETRO_BA",
    "ELPETRO_TH50_R",
]


def _resolve_fits_path() -> Path:
    fits_path = Path.cwd() / "nsa_v1_0_1.fits"
    if fits_path.exists():
        return fits_path.resolve()
    raise FileNotFoundError(f"Could not find {fits_path}")


def _resolve_paper_plots_dir() -> Path:
    out_dir = Path("Plots_paper").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _load_fiducial_data(fits_path: Path) -> np.ndarray:
    with fits.open(fits_path) as hdul:
        data_hdu = hdul[1]
        zdist = data_hdu.data["ZDIST"]
        absmag = data_hdu.data["ELPETRO_ABSMAG"][:, 4]
        b300 = data_hdu.data["ELPETRO_B300"]
        mass = data_hdu.data["ELPETRO_MASS"]
        sersic_n = data_hdu.data["SERSIC_N"]
        ba = data_hdu.data["ELPETRO_BA"]
        th50 = data_hdu.data["ELPETRO_TH50_R"]

    data = np.stack((zdist, absmag, b300, mass, sersic_n, ba, th50), axis=1)

    mask = (
        (zdist < 0.15)
        & (absmag < -10)
        & (b300 > 1e-8)
        & (b300 < 10)
        & (mass > 1e6)
        & (mass < 1e12)
        & (sersic_n < 5.8)
        & (ba > 0)
        & (ba < 1)
        & (th50 > 0)
        & (th50 < 25)
    )
    data = data[mask]

    # Match the paper pipeline: log-transform B300 and MASS.
    data[:, 2] = np.log(data[:, 2])
    data[:, 3] = np.log(data[:, 3])

    data = data[~np.isnan(data).any(axis=1)]
    data = data[~np.isinf(data).any(axis=1)]
    return data


def main() -> None:
    fits_path = _resolve_fits_path()
    paper_plots_dir = _resolve_paper_plots_dir()

    print(f"Using FITS file: {fits_path}", flush=True)
    data = _load_fiducial_data(fits_path)
    print(f"Final dataset shape: {data.shape}", flush=True)

    # Save data in the current working directory for learn_causality_nsa.py.
    data_dict = {VAR_NAMES[i]: data[:, i] for i in range(len(VAR_NAMES))}
    output_pickle = Path.cwd() / "nsa.pkl"
    with open(output_pickle, "wb") as out:
        pickle.dump(data_dict, out)
    print(f"Saved {output_pickle}", flush=True)

    # Paper figure: data_corner.pdf
    levels = [0.393469, 0.864665, 0.988891]  # 1,2,3 sigma enclosed probabilities (2D)
    fig = corner.corner(
        pd.DataFrame(data, columns=VAR_NAMES).to_numpy(dtype=float),
        labels=VAR_NAMES,
        show_titles=False,
        label_kwargs={"fontsize": 18},
        color="#1f77b4",
        levels=levels,
        plot_datapoints=True,
        plot_contours=True,
        fill_contours=True,
        smooth=None,
    )
    for ax in fig.get_axes():
        ax.tick_params(axis="both", labelsize=16)
        if ax.get_xlabel() == "SERSIC_N":
            x0, x1 = ax.get_xlim()
            ax.set_xlim(x0, 6.0)
        if ax.get_ylabel() == "SERSIC_N":
            y0, y1 = ax.get_ylim()
            ax.set_ylim(y0, 6.0)

    pdf_out = paper_plots_dir / "data_corner.pdf"
    fig.savefig(pdf_out, dpi=300, bbox_inches="tight")
    print(f"Saved paper figure: {pdf_out}", flush=True)


if __name__ == "__main__":
    main()
