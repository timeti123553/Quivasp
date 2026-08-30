"""VASP band-structure parsing and publication-quality plotting.

The parser intentionally uses VASP's XML output directly.  This keeps the
core API small and makes band plotting available without requiring a larger
materials-science framework.  Energies are retained exactly as written by
VASP and shifted only when requested for plotting or data access.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def _rows(element: ET.Element, tag: str = "v") -> FloatArray:
    values = [
        [float(value) for value in (row.text or "").split()]
        for row in element.findall(tag)
    ]
    if not values:
        raise ValueError(f"VASP XML element {element.tag!r} contains no {tag!r} rows")
    return np.asarray(values, dtype=float)


def _normalise_label(label: str) -> str:
    cleaned = label.strip().strip("!")
    if cleaned.upper() in {"G", "GAMMA", "\\GAMMA", "Γ"}:
        return "Γ"
    return cleaned


def _normalise_orbital(label: str) -> str:
    cleaned = label.strip()
    if cleaned == "x2-y2":
        return "dx2-y2"
    return cleaned


def _orbital_group(label: str) -> str:
    if label == "s":
        return "s"
    if label in {"px", "py", "pz"}:
        return "p"
    if label.startswith("d") or label == "dx2-y2":
        return "d"
    return label


def _parse_ion_elements(root: ET.Element) -> tuple[str, ...]:
    atom_rows = root.findall("./atominfo/array[@name='atoms']/set/rc")
    elements: list[str] = []
    for row in atom_rows:
        cells = row.findall("c")
        if cells:
            elements.append((cells[0].text or "").strip())
    return tuple(elements)


def _parse_line_mode_kpoints(path: Path, count: int) -> tuple[list[int], list[str]]:
    """Return path indices and labels from a line-mode KPOINTS file."""

    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 5 or "line" not in lines[2].lower():
        raise ValueError(f"{path} is not a VASP line-mode KPOINTS file")
    try:
        points_per_segment = int(lines[1].split()[0])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid line-mode division count in {path}") from exc
    if points_per_segment < 2:
        raise ValueError("Line-mode KPOINTS requires at least two points per segment")

    endpoints: list[str] = []
    for line in lines[4:]:
        text = line.strip()
        if not text:
            continue
        before_comment, separator, comment = text.partition("!")
        fields = before_comment.split()
        if len(fields) < 3:
            continue
        label = comment.strip() if separator else (fields[3] if len(fields) > 3 else "")
        endpoints.append(_normalise_label(label))

    if len(endpoints) < 2 or len(endpoints) % 2:
        raise ValueError(f"Incomplete line-mode segment endpoints in {path}")
    segments = len(endpoints) // 2
    if segments * points_per_segment != count:
        raise ValueError(
            f"KPOINTS describes {segments * points_per_segment} points, "
            f"but vasprun.xml contains {count}"
        )

    indices: list[int] = []
    labels: list[str] = []
    for segment in range(segments):
        start_index = segment * points_per_segment
        end_index = start_index + points_per_segment - 1
        for index, label in (
            (start_index, endpoints[2 * segment]),
            (end_index, endpoints[2 * segment + 1]),
        ):
            if indices and index > 0 and index == indices[-1] + 1 and label == labels[-1]:
                # VASP duplicates the shared endpoint between line-mode segments.
                continue
            indices.append(index)
            labels.append(label)
    return indices, labels


def _parse_spin_set(spin: ET.Element) -> tuple[FloatArray, FloatArray]:
    energies: list[list[float]] = []
    occupations: list[list[float]] = []
    for kpoint in spin.findall("set"):
        rows = _rows(kpoint, "r")
        if rows.shape[1] < 2:
            raise ValueError("Each VASP eigenvalue row must contain energy and occupation")
        energies.append(rows[:, 0].tolist())
        occupations.append(rows[:, 1].tolist())
    if not energies:
        raise ValueError("VASP eigenvalue spin set contains no k-points")
    try:
        return np.asarray(energies, dtype=float), np.asarray(occupations, dtype=float)
    except ValueError as exc:
        raise ValueError("VASP eigenvalue rows have inconsistent band counts") from exc


def _parse_projected_weights(
    root: ET.Element,
    *,
    spin_count: int,
    kpoint_count: int,
    band_count: int,
    ion_count: int,
) -> tuple[FloatArray | None, tuple[str, ...]]:
    projected_array = root.find(".//calculation/projected/array")
    if projected_array is None:
        return None, ()

    orbital_labels = tuple(
        _normalise_orbital(field.text or "")
        for field in projected_array.findall("field")
    )
    if not orbital_labels:
        return None, ()

    root_set = projected_array.find("set")
    if root_set is None:
        return None, ()

    spins: list[list[list[FloatArray]]] = []
    for spin in root_set.findall("set"):
        kpoints: list[list[FloatArray]] = []
        for kpoint in spin.findall("set"):
            bands: list[FloatArray] = []
            for band in kpoint.findall("set"):
                rows = _rows(band, "r")
                bands.append(rows)
            kpoints.append(bands)
        spins.append(kpoints)

    try:
        weights = np.asarray(spins, dtype=float)
    except ValueError as exc:
        raise ValueError("Projected eigenvalue weights have inconsistent shapes") from exc

    expected = (spin_count, kpoint_count, band_count, ion_count, len(orbital_labels))
    if weights.shape != expected:
        raise ValueError(f"Projected eigenvalue weights have shape {weights.shape}, expected {expected}")
    return weights, orbital_labels


@dataclass(frozen=True)
class BandStructure:
    """Reusable electronic band data parsed from a VASP calculation.

    Arrays use the shape ``(spin, kpoint, band)``.  ``energies`` contains the
    absolute eigenvalues from VASP; use :attr:`shifted_energies` for values
    referenced to the Fermi level.
    """

    kpoints: FloatArray
    distances: FloatArray
    energies: FloatArray
    occupations: FloatArray
    efermi: float
    reciprocal_lattice: FloatArray
    tick_indices: tuple[int, ...] = ()
    tick_labels: tuple[str, ...] = ()
    ion_elements: tuple[str, ...] = ()
    orbital_labels: tuple[str, ...] = ()
    projections: FloatArray | None = None
    source: Path | None = None

    def __post_init__(self) -> None:
        if self.kpoints.ndim != 2 or self.kpoints.shape[1] != 3:
            raise ValueError("kpoints must have shape (n, 3)")
        if self.energies.ndim != 3 or self.occupations.shape != self.energies.shape:
            raise ValueError("energies and occupations must share shape (spin, kpoint, band)")
        if self.energies.shape[1] != len(self.kpoints):
            raise ValueError("eigenvalue and k-point counts differ")
        if self.distances.shape != (len(self.kpoints),):
            raise ValueError("distances must contain one value per k-point")
        if len(self.tick_indices) != len(self.tick_labels):
            raise ValueError("tick_indices and tick_labels must have equal length")
        if self.projections is not None:
            expected = (
                self.spin_count,
                len(self.kpoints),
                self.band_count,
                len(self.ion_elements),
                len(self.orbital_labels),
            )
            if self.projections.shape != expected:
                raise ValueError(f"projections must have shape {expected}")

    @property
    def shifted_energies(self) -> FloatArray:
        """Band energies relative to the Fermi level, in eV."""

        return self.energies - self.efermi

    @property
    def spin_count(self) -> int:
        return int(self.energies.shape[0])

    @property
    def band_count(self) -> int:
        return int(self.energies.shape[2])

    @property
    def available_elements(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(element for element in self.ion_elements if element))

    @property
    def available_orbitals(self) -> tuple[str, ...]:
        grouped = tuple(dict.fromkeys(_orbital_group(orbital) for orbital in self.orbital_labels))
        return tuple(dict.fromkeys((*grouped, *self.orbital_labels)))

    def projection_weights(
        self,
        *,
        elements: Sequence[str] | None = None,
        orbitals: Sequence[str] | None = None,
    ) -> dict[str, FloatArray]:
        """Return projection weights by selected element or orbital.

        Each returned array has shape ``(spin, kpoint, band)``.  Orbital
        selectors may be grouped labels such as ``s``, ``p``, and ``d`` or
        detailed labels such as ``px`` and ``dxy``.
        """

        if self.projections is None:
            raise ValueError("vasprun.xml does not contain projected eigenvalue weights")
        if elements and orbitals:
            raise ValueError("select either elements or orbitals, not both")

        if elements:
            available = set(self.available_elements)
            unknown = [element for element in elements if element not in available]
            if unknown:
                raise ValueError(f"Unknown element projection(s): {', '.join(unknown)}")
            weights = {}
            for element in elements:
                ion_indices = [i for i, item in enumerate(self.ion_elements) if item == element]
                weights[element] = self.projections[:, :, :, ion_indices, :].sum(axis=(3, 4))
            return weights

        if orbitals:
            available = set(self.available_orbitals)
            unknown = [orbital for orbital in orbitals if orbital not in available]
            if unknown:
                raise ValueError(f"Unknown orbital projection(s): {', '.join(unknown)}")
            weights = {}
            for orbital in orbitals:
                orbital_indices = [
                    i for i, item in enumerate(self.orbital_labels)
                    if item == orbital or _orbital_group(item) == orbital
                ]
                weights[orbital] = self.projections[:, :, :, :, orbital_indices].sum(axis=(3, 4))
            return weights

        return {}

    @classmethod
    def from_vasp(
        cls,
        calculation: str | Path,
        *,
        kpoints_filename: str | Path | None = None,
    ) -> "BandStructure":
        """Load band data from ``vasprun.xml`` and an optional KPOINTS file.

        ``calculation`` may be a calculation directory or the XML file itself.
        When omitted, ``kpoints_filename`` defaults to ``KPOINTS`` beside the
        XML.  A missing KPOINTS file is allowed; the band data remains usable,
        but high-symmetry axis labels are unavailable.
        """

        source = Path(calculation).expanduser()
        xml_path = source / "vasprun.xml" if source.is_dir() else source
        if not xml_path.is_file():
            raise FileNotFoundError(f"VASP output not found: {xml_path}")

        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError as exc:
            raise ValueError(f"Invalid or incomplete VASP XML: {xml_path}") from exc

        kpoint_element = root.find("./kpoints/varray[@name='kpointlist']")
        if kpoint_element is None:
            raise ValueError("vasprun.xml does not contain a k-point list")
        kpoints = _rows(kpoint_element)
        if kpoints.shape[1] != 3:
            raise ValueError("VASP k-points must contain three reciprocal coordinates")

        lattice_element = root.find(
            ".//structure[@name='finalpos']/crystal/varray[@name='rec_basis']"
        )
        if lattice_element is None:
            lattice_element = root.find(".//crystal/varray[@name='rec_basis']")
        reciprocal_lattice = _rows(lattice_element) if lattice_element is not None else np.eye(3)
        if reciprocal_lattice.shape != (3, 3):
            raise ValueError("VASP reciprocal lattice must have shape (3, 3)")

        # Use the last complete calculation: restart XML files can contain more
        # than one calculation and the final one is the converged result.
        eigenvalue_elements = root.findall(".//calculation/eigenvalues/array/set")
        if not eigenvalue_elements:
            raise ValueError("vasprun.xml does not contain band eigenvalues")
        spin_sets = eigenvalue_elements[-1].findall("set")
        parsed_spins = [_parse_spin_set(spin) for spin in spin_sets]
        energies = np.asarray([item[0] for item in parsed_spins], dtype=float)
        occupations = np.asarray([item[1] for item in parsed_spins], dtype=float)
        if energies.shape[1] != len(kpoints):
            raise ValueError(
                f"vasprun.xml has {len(kpoints)} k-points but {energies.shape[1]} eigenvalue sets"
            )

        efermi_elements = root.findall(".//dos/i[@name='efermi']")
        if not efermi_elements or efermi_elements[-1].text is None:
            raise ValueError("vasprun.xml does not contain a Fermi energy")
        efermi = float(efermi_elements[-1].text)

        cartesian = kpoints @ reciprocal_lattice
        step_lengths = np.linalg.norm(np.diff(cartesian, axis=0), axis=1)
        distances = np.concatenate(([0.0], np.cumsum(step_lengths)))

        if kpoints_filename is None:
            candidate = xml_path.with_name("KPOINTS")
        else:
            candidate = Path(kpoints_filename).expanduser()
            if not candidate.is_absolute():
                candidate = xml_path.parent / candidate
        tick_indices: list[int] = []
        tick_labels: list[str] = []
        if candidate.is_file():
            tick_indices, tick_labels = _parse_line_mode_kpoints(candidate, len(kpoints))

        ion_elements = _parse_ion_elements(root)
        projections, orbital_labels = _parse_projected_weights(
            root,
            spin_count=int(energies.shape[0]),
            kpoint_count=len(kpoints),
            band_count=int(energies.shape[2]),
            ion_count=len(ion_elements),
        )

        return cls(
            kpoints=kpoints,
            distances=distances,
            energies=energies,
            occupations=occupations,
            efermi=efermi,
            reciprocal_lattice=reciprocal_lattice,
            tick_indices=tuple(tick_indices),
            tick_labels=tuple(tick_labels),
            ion_elements=ion_elements,
            orbital_labels=orbital_labels,
            projections=projections,
            source=xml_path.resolve(),
        )

    def plot(
        self,
        *,
        ylim: tuple[float, float] | None = (-4.0, 4.0),
        figsize: tuple[float, float] = (7.2, 5.4),
        linewidth: float = 1.15,
        colors: str | Sequence[str] = ("#174A7E", "#C44E52"),
        show_fermi: bool = True,
        labels: Sequence[str] | None = None,
        elements: Sequence[str] | None = None,
        orbitals: Sequence[str] | None = None,
        projection_scale: float = 42.0,
        projection_alpha: float = 0.72,
        projection_colors: Sequence[str] = (
            "#D55E00",
            "#0072B2",
            "#009E73",
            "#CC79A7",
            "#E69F00",
        ),
        output: str | Path | None = None,
        dpi: int = 300,
        ax: plt.Axes | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot all bands and optionally save the figure.

        The returned figure remains open so callers can add annotations.  The
        output format is inferred from the filename suffix by Matplotlib.
        """

        if linewidth <= 0:
            raise ValueError("linewidth must be positive")
        if dpi <= 0:
            raise ValueError("dpi must be positive")
        if projection_scale <= 0:
            raise ValueError("projection_scale must be positive")
        if elements and orbitals:
            raise ValueError("select either elements or orbitals, not both")
        if ax is None:
            figure, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        else:
            figure = ax.figure

        palette = [colors] if isinstance(colors, str) else list(colors)
        if not palette:
            raise ValueError("colors must contain at least one color")
        shifted = self.shifted_energies
        for spin in range(self.spin_count):
            color = palette[spin % len(palette)]
            for band in range(self.band_count):
                alpha = 0.28 if elements or orbitals else 1.0
                ax.plot(self.distances, shifted[spin, :, band], color=color, lw=linewidth, alpha=alpha)

        selected_weights = self.projection_weights(elements=elements, orbitals=orbitals)
        for item_index, (name, weights) in enumerate(selected_weights.items()):
            color = projection_colors[item_index % len(projection_colors)]
            for spin in range(self.spin_count):
                marker = "o" if spin == 0 else "^"
                for band in range(self.band_count):
                    sizes = np.clip(weights[spin, :, band], 0.0, None) * projection_scale
                    if np.any(sizes > 0):
                        ax.scatter(
                            self.distances,
                            shifted[spin, :, band],
                            s=sizes,
                            color=color,
                            alpha=projection_alpha,
                            marker=marker,
                            linewidths=0,
                        )

        tick_labels = tuple(labels) if labels is not None else self.tick_labels
        if tick_labels:
            if len(tick_labels) != len(self.tick_indices):
                raise ValueError("custom labels must match the high-symmetry tick count")
            tick_positions = self.distances[np.asarray(self.tick_indices)]
            ax.set_xticks(tick_positions, tick_labels)
            for position in tick_positions:
                ax.axvline(position, color="#777777", lw=0.65, alpha=0.55, zorder=0)
        else:
            ax.set_xlabel(r"Wave vector $k$")

        if show_fermi:
            ax.axhline(0.0, color="#333333", linestyle="--", lw=0.85, alpha=0.8)
        if ylim is not None:
            if ylim[0] >= ylim[1]:
                raise ValueError("ylim lower bound must be smaller than upper bound")
            ax.set_ylim(*ylim)
        ax.set_xlim(float(self.distances[0]), float(self.distances[-1]))
        ax.set_ylabel(r"$E - E_\mathrm{F}$ (eV)")
        ax.tick_params(direction="in", top=True, right=True)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

        if self.spin_count > 1 or selected_weights:
            from matplotlib.lines import Line2D

            handles = []
            legend_labels = []
            if self.spin_count > 1:
                handles.extend(
                    Line2D([0], [0], color=palette[i % len(palette)], lw=linewidth)
                    for i in range(self.spin_count)
                )
                legend_labels.extend(f"Spin {i + 1}" for i in range(self.spin_count))
            handles.extend(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    color=projection_colors[i % len(projection_colors)],
                )
                for i, _ in enumerate(selected_weights)
            )
            legend_labels.extend(selected_weights.keys())
            ax.legend(handles, legend_labels, frameon=False)

        if output is not None:
            destination = Path(output).expanduser()
            destination.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(destination, dpi=dpi, bbox_inches="tight")
        return figure, ax


def plot_band_structure(
    calculation: str | Path,
    *,
    kpoints_filename: str | Path | None = None,
    **plot_options: Any,
) -> tuple[plt.Figure, plt.Axes]:
    """Load and plot a VASP band structure in one call."""

    band_structure = BandStructure.from_vasp(
        calculation, kpoints_filename=kpoints_filename
    )
    return band_structure.plot(**plot_options)
