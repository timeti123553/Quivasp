from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from quivasp import BandStructure, plot_band_structure


CR_I3 = Path(__file__).parent / "CrI3"


@pytest.fixture(scope="module")
def cr_i3_bands() -> BandStructure:
    return BandStructure.from_vasp(CR_I3)


def test_parses_real_cr_i3_band_structure(cr_i3_bands: BandStructure):
    bands = cr_i3_bands

    assert bands.energies.shape == (2, 60, 44)
    assert bands.occupations.shape == bands.energies.shape
    assert bands.spin_count == 2
    assert bands.band_count == 44
    assert bands.efermi == pytest.approx(-1.98135951)
    assert bands.kpoints[0] == pytest.approx((0.0, 0.0, 0.0))
    assert bands.kpoints[-1] == pytest.approx((0.0, 0.0, 0.0))
    assert np.all(np.diff(bands.distances) >= 0)
    assert bands.tick_indices == (0, 19, 39, 59)
    assert bands.tick_labels == ("Γ", "M", "K", "Γ")
    assert np.shares_memory(bands.shifted_energies, bands.energies) is False
    assert bands.shifted_energies[0, 0, 0] == pytest.approx(
        bands.energies[0, 0, 0] - bands.efermi
    )


def test_plot_is_publication_ready_and_savable(cr_i3_bands: BandStructure, tmp_path: Path):
    output = tmp_path / "bands.png"
    figure, axis = cr_i3_bands.plot(
        ylim=(-3, 3),
        colors=("navy", "firebrick"),
        output=output,
        dpi=120,
    )

    assert output.is_file()
    assert output.stat().st_size > 1_000
    assert axis.get_ylim() == pytest.approx((-3, 3))
    assert axis.get_ylabel() == r"$E - E_\mathrm{F}$ (eV)"
    assert [tick.get_text() for tick in axis.get_xticklabels()] == ["Γ", "M", "K", "Γ"]
    assert len(axis.lines) >= 2 * cr_i3_bands.band_count
    figure.clear()


def test_convenience_api_accepts_explicit_xml_path(tmp_path: Path):
    output = tmp_path / "convenience.pdf"
    figure, axis = plot_band_structure(
        CR_I3 / "vasprun.xml",
        kpoints_filename="KPOINTS",
        output=output,
        show_fermi=False,
    )

    assert output.is_file()
    assert axis.get_xlim()[0] == pytest.approx(0.0)
    figure.clear()


def test_missing_vasprun_has_actionable_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="VASP output not found"):
        BandStructure.from_vasp(tmp_path)


def test_plot_rejects_inconsistent_custom_labels(cr_i3_bands: BandStructure):
    with pytest.raises(ValueError, match="custom labels"):
        cr_i3_bands.plot(labels=("Γ", "M"))
