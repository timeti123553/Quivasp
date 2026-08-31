"""VASP band-structure parsing and publication-quality plotting.

The parser intentionally uses VASP's XML output directly.  This keeps the
core API small and makes band plotting available without requiring a larger
materials-science framework.  Energies are retained exactly as written by
VASP and shifted only when requested for plotting or data access.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]

_GROUPED_ORBITALS: dict[str, tuple[str, ...]] = {
    "s": ("s",),
    "p": ("px", "py", "pz"),
    "d": ("dxy", "dyz", "dz2", "dxz", "dx2-y2"),
}


def _normalise_orbital(name: str) -> str:
    cleaned = name.strip().lower().replace("d_x2-y2", "dx2-y2")
    if cleaned == "x2-y2":
        return "dx2-y2"
    return cleaned


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


def _parse_atom_elements(root: ET.Element) -> tuple[str, ...]:
    atoms = root.find("./atominfo/array[@name='atoms']/set")
    if atoms is None:
        raise ValueError("vasprun.xml does not contain atom identities")
    elements: list[str] = []
    for row in atoms.findall("rc"):
        cells = row.findall("c")
        if not cells or cells[0].text is None:
            raise ValueError("vasprun.xml contains an atom without an element")
        elements.append(cells[0].text.strip())
    if not elements:
        raise ValueError("vasprun.xml contains no atoms")
    return tuple(elements)


def _parse_projections(
    root: ET.Element,
) -> tuple[FloatArray | None, tuple[str, ...], tuple[str, ...]]:
    """Parse projected weights as ``(spin, kpoint, band, ion, orbital)``."""

    arrays = root.findall(".//calculation/projected/array")
    if not arrays:
        return None, (), ()
    array = arrays[-1]
    orbitals = tuple(
        _normalise_orbital(field.text or "") for field in array.findall("field")
    )
    if not orbitals or any(not orbital for orbital in orbitals):
        raise ValueError("VASP projected data contains invalid orbital names")
    atom_elements = _parse_atom_elements(root)
    outer = array.find("set")
    if outer is None:
        raise ValueError("VASP projected array contains no spin data")

    spin_values: list[list[list[list[list[float]]]]] = []
    for spin in outer.findall("set"):
        kpoint_values: list[list[list[list[float]]]] = []
        for kpoint in spin.findall("set"):
            band_values: list[list[list[float]]] = []
            for band in kpoint.findall("set"):
                rows = _rows(band, "r")
                if rows.shape != (len(atom_elements), len(orbitals)):
                    raise ValueError(
                        "Projected band rows do not match the atom/orbital metadata"
                    )
                band_values.append(rows.tolist())
            kpoint_values.append(band_values)
        spin_values.append(kpoint_values)
    try:
        projections = np.asarray(spin_values, dtype=float)
    except ValueError as exc:
        raise ValueError("VASP projected data has inconsistent dimensions") from exc
    if projections.ndim != 5:
        raise ValueError("VASP projected data must have spin/kpoint/band/ion/orbital axes")
    return projections, atom_elements, orbitals


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
    projections: FloatArray | None = None
    atom_elements: tuple[str, ...] = ()
    orbital_names: tuple[str, ...] = ()
    tick_indices: tuple[int, ...] = ()
    tick_labels: tuple[str, ...] = ()
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
                len(self.atom_elements),
                len(self.orbital_names),
            )
            if self.projections.shape != expected:
                raise ValueError(
                    f"projections must have shape {expected}, got {self.projections.shape}"
                )

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

        projections, atom_elements, orbital_names = _parse_projections(root)
        if projections is not None and projections.shape[:3] != energies.shape:
            raise ValueError(
                "Projected data and eigenvalues have different spin/k-point/band dimensions"
            )

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

        return cls(
            kpoints=kpoints,
            distances=distances,
            energies=energies,
            occupations=occupations,
            efermi=efermi,
            reciprocal_lattice=reciprocal_lattice,
            projections=projections,
            atom_elements=atom_elements,
            orbital_names=orbital_names,
            tick_indices=tuple(tick_indices),
            tick_labels=tuple(tick_labels),
            source=xml_path.resolve(),
        )

    def projection_weights(
        self,
        *,
        elements: Sequence[str] | None = None,
        orbitals: Sequence[str] | None = None,
    ) -> dict[str, FloatArray]:
        """Return named projected weights for element and/or orbital selections.

        Grouped orbital names ``s``, ``p``, and ``d`` expand to their detailed
        components.  Supplying both element and orbital selections produces a
        component for every element/orbital pair.
        """

        if self.projections is None:
            raise ValueError("This vasprun.xml does not contain projected eigenvalues")
        available_elements = tuple(dict.fromkeys(self.atom_elements))
        selected_elements = tuple(elements) if elements else ()
        selected_orbitals = tuple(_normalise_orbital(item) for item in orbitals or ())

        unknown_elements = sorted(set(selected_elements) - set(available_elements))
        if unknown_elements:
            raise ValueError(
                "Unknown elements: "
                + ", ".join(unknown_elements)
                + "; available: "
                + ", ".join(available_elements)
            )
        unknown_orbitals = sorted(
            orbital
            for orbital in selected_orbitals
            if orbital not in _GROUPED_ORBITALS and orbital not in self.orbital_names
        )
        if unknown_orbitals:
            raise ValueError(
                "Unknown orbitals: "
                + ", ".join(unknown_orbitals)
                + "; available: "
                + ", ".join(self.orbital_names)
            )

        element_groups: list[tuple[str, NDArray[np.int_]]]
        orbital_groups: list[tuple[str, NDArray[np.int_]]]
        if selected_elements:
            element_groups = [
                (
                    element,
                    np.asarray(
                        [i for i, value in enumerate(self.atom_elements) if value == element],
                        dtype=int,
                    ),
                )
                for element in selected_elements
            ]
        else:
            element_groups = [("", np.arange(len(self.atom_elements), dtype=int))]

        if selected_orbitals:
            orbital_groups = []
            for name in selected_orbitals:
                members = _GROUPED_ORBITALS.get(name, (name,))
                indices = np.asarray(
                    [i for i, value in enumerate(self.orbital_names) if value in members],
                    dtype=int,
                )
                if not len(indices):
                    raise ValueError(f"Projection data has no components for orbital {name!r}")
                orbital_groups.append((name, indices))
        else:
            orbital_groups = [("", np.arange(len(self.orbital_names), dtype=int))]

        components: dict[str, FloatArray] = {}
        for element, ion_indices in element_groups:
            for orbital, orbital_indices in orbital_groups:
                label = " ".join(part for part in (element, orbital) if part)
                if not label:
                    label = "total"
                selected = np.take(self.projections, ion_indices, axis=3)
                selected = np.take(selected, orbital_indices, axis=4)
                components[label] = selected.sum(axis=(3, 4))
        return components

    def plot(
        self,
        *,
        ylim: tuple[float, float] | None = (-4.0, 4.0),
        figsize: tuple[float, float] = (7.2, 5.4),
        linewidth: float = 1.15,
        colors: str | Sequence[str] = ("#174A7E", "#C44E52"),
        font_size: float = 12.0,
        fermi_linewidth: float = 0.9,
        symmetry_linewidth: float = 0.7,
        show_fermi: bool = True,
        elements: Sequence[str] | None = None,
        orbitals: Sequence[str] | None = None,
        projection_colors: Sequence[str] | Mapping[str, str] | None = None,
        projection_scale: float = 44.0,
        projection_threshold: float = 0.015,
        show_legend: bool = True,
        labels: Sequence[str] | None = None,
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
        if font_size <= 0:
            raise ValueError("font_size must be positive")
        if projection_scale <= 0:
            raise ValueError("projection_scale must be positive")
        if projection_threshold < 0:
            raise ValueError("projection_threshold must be non-negative")
        if ax is None:
            figure, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        else:
            figure = ax.figure

        palette = [colors] if isinstance(colors, str) else list(colors)
        if not palette:
            raise ValueError("colors must contain at least one color")
        shifted = self.shifted_energies
        projecting = bool(elements or orbitals)
        for spin in range(self.spin_count):
            color = palette[spin % len(palette)]
            for band in range(self.band_count):
                ax.plot(
                    self.distances,
                    shifted[spin, :, band],
                    color=color,
                    lw=linewidth,
                    alpha=0.28 if projecting else 1.0,
                )

        if projecting:
            components = self.projection_weights(elements=elements, orbitals=orbitals)
            default_projection_colors = (
                "#0072B2",
                "#D55E00",
                "#009E73",
                "#CC79A7",
                "#E69F00",
                "#56B4E9",
                "#000000",
            )
            if isinstance(projection_colors, Mapping):
                component_colors = {
                    name: projection_colors.get(
                        name, default_projection_colors[i % len(default_projection_colors)]
                    )
                    for i, name in enumerate(components)
                }
            else:
                sequence = list(projection_colors or default_projection_colors)
                if not sequence:
                    raise ValueError("projection_colors must contain at least one color")
                component_colors = {
                    name: sequence[i % len(sequence)] for i, name in enumerate(components)
                }
            markers = ("o", "x")
            for component, weights in components.items():
                for spin in range(self.spin_count):
                    for band in range(self.band_count):
                        weight = np.clip(weights[spin, :, band], 0.0, None)
                        mask = weight >= projection_threshold
                        if not np.any(mask):
                            continue
                        ax.scatter(
                            self.distances[mask],
                            shifted[spin, mask, band],
                            s=np.maximum(4.0, projection_scale * weight[mask]),
                            c=component_colors[component],
                            marker=markers[spin % len(markers)],
                            linewidths=0.7,
                            alpha=0.72,
                            edgecolors="none" if markers[spin % len(markers)] == "o" else None,
                            zorder=3,
                        )

        tick_labels = tuple(labels) if labels is not None else self.tick_labels
        if tick_labels:
            if len(tick_labels) != len(self.tick_indices):
                raise ValueError("custom labels must match the high-symmetry tick count")
            tick_positions = self.distances[np.asarray(self.tick_indices)]
            ax.set_xticks(tick_positions, tick_labels)
            for position in tick_positions:
                ax.axvline(
                    position,
                    color="#777777",
                    lw=symmetry_linewidth,
                    alpha=0.55,
                    zorder=0,
                )
        else:
            ax.set_xlabel(r"Wave vector $k$")

        if show_fermi:
            ax.axhline(
                0.0,
                color="#333333",
                linestyle="--",
                lw=fermi_linewidth,
                alpha=0.85,
            )
        if ylim is not None:
            if ylim[0] >= ylim[1]:
                raise ValueError("ylim lower bound must be smaller than upper bound")
            ax.set_ylim(*ylim)
        ax.set_xlim(float(self.distances[0]), float(self.distances[-1]))
        ax.set_ylabel(r"$E - E_\mathrm{F}$ (eV)")
        ax.tick_params(direction="in", top=True, right=True, labelsize=font_size)
        ax.xaxis.label.set_size(font_size)
        ax.yaxis.label.set_size(font_size)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

        if show_legend and (self.spin_count > 1 or projecting):
            from matplotlib.lines import Line2D

            handles: list[Line2D] = []
            legend_labels: list[str] = []
            if self.spin_count > 1:
                for i in range(self.spin_count):
                    handles.append(
                        Line2D(
                            [0],
                            [0],
                            color=palette[i % len(palette)],
                            marker=("o", "x")[i % 2] if projecting else None,
                            lw=linewidth,
                        )
                    )
                    legend_labels.append(f"Spin {i + 1}")
            if projecting:
                for component in components:
                    handles.append(
                        Line2D(
                            [0],
                            [0],
                            color="none",
                            marker="o",
                            markerfacecolor=component_colors[component],
                            markeredgecolor="none",
                            markersize=7,
                        )
                    )
                    legend_labels.append(component)
            ax.legend(handles, legend_labels, frameon=False, fontsize=font_size - 1)

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
