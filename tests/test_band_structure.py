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
    assert bands.available_elements == ("Cr", "I")
    assert "dxy" in bands.available_orbitals
    assert bands.projections is not None
    assert bands.projections.shape == (2, 60, 44, 8, 9)
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


def test_element_projection_weights_and_plot(cr_i3_bands: BandStructure, tmp_path: Path):
    weights = cr_i3_bands.projection_weights(elements=("Cr", "I"))

    assert set(weights) == {"Cr", "I"}
    assert weights["Cr"].shape == cr_i3_bands.energies.shape
    assert float(weights["Cr"].max()) > 0
    assert float(weights["I"].max()) > 0

    output = tmp_path / "element_projected.png"
    figure, axis = cr_i3_bands.plot(elements=("Cr", "I"), output=output, dpi=120)

    assert output.is_file()
    assert axis.collections
    figure.clear()


def test_orbital_projection_groups(cr_i3_bands: BandStructure):
    grouped = cr_i3_bands.projection_weights(orbitals=("p", "d"))
    detailed = cr_i3_bands.projection_weights(orbitals=("px", "py", "pz", "dxy"))

    assert grouped["p"].shape == cr_i3_bands.energies.shape
    assert detailed["px"].shape == cr_i3_bands.energies.shape
    assert np.all(grouped["p"] >= detailed["px"])
    assert np.all(grouped["p"] >= detailed["py"])
    assert np.all(grouped["p"] >= detailed["pz"])


def test_projection_selection_validation(cr_i3_bands: BandStructure):
    with pytest.raises(ValueError, match="Unknown element"):
        cr_i3_bands.projection_weights(elements=("Fe",))
    with pytest.raises(ValueError, match="Unknown orbital"):
        cr_i3_bands.projection_weights(orbitals=("f",))
    with pytest.raises(ValueError, match="either elements or orbitals"):
        cr_i3_bands.plot(elements=("Cr",), orbitals=("d",))


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
