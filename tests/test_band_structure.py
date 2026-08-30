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


def test_parses_element_and_orbital_projections(cr_i3_bands: BandStructure):
    bands = cr_i3_bands

    assert bands.atom_elements == ("Cr", "Cr", "I", "I", "I", "I", "I", "I")
    assert bands.orbital_names == (
        "s",
        "py",
        "pz",
        "px",
        "dxy",
        "dyz",
        "dz2",
        "dxz",
        "dx2-y2",
    )
    assert bands.projections is not None
    assert bands.projections.shape == (2, 60, 44, 8, 9)

    element = bands.projection_weights(elements=("Cr", "I"))
    assert set(element) == {"Cr", "I"}
    assert element["Cr"].shape == bands.energies.shape
    assert np.all(element["Cr"] >= 0)

    orbital = bands.projection_weights(orbitals=("s", "p", "d", "dz2"))
    assert set(orbital) == {"s", "p", "d", "dz2"}
    assert np.allclose(
        orbital["p"],
        bands.projections[..., 1:4].sum(axis=(3, 4)),
    )

    combined = bands.projection_weights(elements=("Cr",), orbitals=("d", "px"))
    assert set(combined) == {"Cr d", "Cr px"}


def test_projection_plot_is_savable(cr_i3_bands: BandStructure, tmp_path: Path):
    output = tmp_path / "cri3-projected.png"
    figure, axis = cr_i3_bands.plot(
        elements=("Cr", "I"),
        projection_scale=20,
        projection_threshold=0.1,
        output=output,
        dpi=100,
    )

    assert output.is_file()
    assert output.stat().st_size > 1_000
    assert len(axis.collections) > 0
    assert {text.get_text() for text in axis.get_legend().get_texts()} >= {
        "Spin 1",
        "Spin 2",
        "Cr",
        "I",
    }
    figure.clear()


def test_projection_selection_rejects_unknown_values(cr_i3_bands: BandStructure):
    with pytest.raises(ValueError, match="Unknown elements"):
        cr_i3_bands.projection_weights(elements=("Fe",))
    with pytest.raises(ValueError, match="Unknown orbitals"):
        cr_i3_bands.projection_weights(orbitals=("f",))
