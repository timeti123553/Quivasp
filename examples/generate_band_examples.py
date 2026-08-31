"""Regenerate the publication-style figures embedded in the README."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from quivasp import BandStructure


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "images"


def render(calculation: str, filename: str, **options: object) -> None:
    bands = BandStructure.from_vasp(ROOT / "tests" / calculation)
    figure, _ = bands.plot(
        ylim=(-4, 4),
        figsize=(7.2, 5.2),
        font_size=12,
        linewidth=1.1,
        dpi=180,
        output=OUTPUT / filename,
        **options,
    )
    figure.clear()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    render("CrI3", "cri3_band_spin.png")
    render(
        "CrI3",
        "cri3_band_element_spin.png",
        elements=("Cr", "I"),
        projection_scale=30,
        projection_threshold=0.035,
    )
    render(
        "CrI3",
        "cri3_band_orbital_spd_spin.png",
        orbitals=("s", "p", "d"),
        projection_scale=28,
        projection_threshold=0.035,
    )
    render(
        "CrI3",
        "cri3_band_orbital_detailed_spin.png",
        orbitals=("px", "pz", "dxy", "dz2", "dx2-y2"),
        projection_scale=35,
        projection_threshold=0.045,
    )
    render("Si", "si_band.png")
    render(
        "Si",
        "si_band_element.png",
        elements=("Si",),
        projection_scale=30,
        projection_threshold=0.035,
    )
    render(
        "Si",
        "si_band_orbital_spd.png",
        orbitals=("s", "p", "d"),
        projection_scale=28,
        projection_threshold=0.035,
    )
    render(
        "Si",
        "si_band_orbital_detailed.png",
        orbitals=("s", "px", "py", "pz"),
        projection_scale=35,
        projection_threshold=0.045,
    )


if __name__ == "__main__":
    main()
