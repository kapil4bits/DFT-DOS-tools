#!/usr/bin/env python3
"""
Analyze VASP DOSCAR files and calculate d-band centers.

Features
--------
- Plot total DOS for spin-polarized and non-spin-polarized calculations.
- Extract projected d-DOS for selected atoms.
- Calculate d-band centers relative to the Fermi level.
- Export processed DOS data to CSV.
- No ASE dependency is required.

Supported projected-DOS formats
-------------------------------
Common DOSCAR formats generated with LORBIT = 10 or 11.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ORBITAL_SETS: Dict[int, List[str]] = {
    3: ["s", "p", "d"],
    4: ["s", "p", "d", "f"],
    9: ["s", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "dx2-y2"],
    16: [
        "s",
        "py",
        "pz",
        "px",
        "dxy",
        "dyz",
        "dz2",
        "dxz",
        "dx2-y2",
        "fy(3x2-y2)",
        "fxyz",
        "fyz2",
        "fz3",
        "fxz2",
        "fz(x2-y2)",
        "fx(x2-3y2)",
    ],
}


@dataclass
class DOSData:
    """Container for parsed DOSCAR data."""

    nions: int
    fermi_energy: float
    energy: np.ndarray
    total_dos_up: np.ndarray
    total_dos_down: Optional[np.ndarray]
    projected_dos: Optional[List[Dict[str, np.ndarray]]]

    @property
    def energy_relative_to_fermi(self) -> np.ndarray:
        """Return energies relative to the Fermi level."""
        return self.energy - self.fermi_energy

    @property
    def is_spin_polarized(self) -> bool:
        """Return True for spin-polarized calculations."""
        return self.total_dos_down is not None


def _float_row(line: str) -> np.ndarray:
    """Convert one whitespace-separated DOSCAR row to a float array."""
    try:
        return np.asarray([float(value) for value in line.split()], dtype=float)
    except ValueError as exc:
        raise ValueError(f"Could not parse DOSCAR row: {line!r}") from exc


def _projected_labels(number_of_columns: int) -> List[str]:
    """
    Return projected-DOS labels for common VASP DOSCAR layouts.

    For spin-polarized calculations, VASP stores up and down values
    in an interleaved order for each orbital.
    """
    for number_of_orbitals, orbitals in ORBITAL_SETS.items():
        if number_of_columns == number_of_orbitals:
            return orbitals

        if number_of_columns == 2 * number_of_orbitals:
            labels: List[str] = []
            for orbital in orbitals:
                labels.extend([f"{orbital}_up", f"{orbital}_down"])
            return labels

    supported = sorted(list(ORBITAL_SETS) + [2 * n for n in ORBITAL_SETS])

    raise ValueError(
        f"Unsupported projected-DOS format with {number_of_columns} columns. "
        f"Supported projected-orbital column counts: {supported}."
    )


def read_doscar(path: Path) -> DOSData:
    """Read total and atom-projected DOS from a VASP DOSCAR file."""
    if not path.is_file():
        raise FileNotFoundError(f"DOSCAR file not found: {path}")

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    if len(lines) < 7:
        raise ValueError("DOSCAR file appears to be incomplete.")

    nions = int(lines[0].split()[0])

    metadata = _float_row(lines[5])
    nedos = int(round(metadata[2]))
    fermi_energy = float(metadata[3])

    total_start = 6
    total_end = total_start + nedos

    if total_end > len(lines):
        raise ValueError("The total-DOS section of the DOSCAR file is incomplete.")

    total_block = np.vstack(
        [_float_row(line) for line in lines[total_start:total_end]]
    )

    if total_block.shape[1] == 3:
        total_dos_up = total_block[:, 1]
        total_dos_down = None

    elif total_block.shape[1] >= 5:
        total_dos_up = total_block[:, 1]
        total_dos_down = total_block[:, 2]

    else:
        raise ValueError("Unsupported total-DOS format in DOSCAR.")

    projected_dos: Optional[List[Dict[str, np.ndarray]]] = None
    cursor = total_end

    if cursor < len(lines):
        projected_dos = []

        for atom_index in range(nions):
            if cursor >= len(lines):
                raise ValueError(
                    f"Projected DOS block missing for atom {atom_index + 1}."
                )

            cursor += 1  # Skip the projected-DOS header line.

            atom_end = cursor + nedos

            if atom_end > len(lines):
                raise ValueError(
                    f"Projected DOS block is incomplete for atom {atom_index + 1}."
                )

            atom_block = np.vstack(
                [_float_row(line) for line in lines[cursor:atom_end]]
            )

            cursor = atom_end
            labels = _projected_labels(atom_block.shape[1] - 1)

            projected_dos.append(
                {
                    label: atom_block[:, index + 1]
                    for index, label in enumerate(labels)
                }
            )

    return DOSData(
        nions=nions,
        fermi_energy=fermi_energy,
        energy=total_block[:, 0],
        total_dos_up=total_dos_up,
        total_dos_down=total_dos_down,
        projected_dos=projected_dos,
    )


def _d_dos_for_atom(
    atom_dos: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return the summed d-projected DOS for one atom."""
    if any(label.endswith("_up") for label in atom_dos):
        up_channels = [
            values
            for label, values in atom_dos.items()
            if label.startswith("d") and label.endswith("_up")
        ]

        down_channels = [
            values
            for label, values in atom_dos.items()
            if label.startswith("d") and label.endswith("_down")
        ]

        if not up_channels or not down_channels:
            raise ValueError("Could not identify d-orbital spin channels.")

        return np.sum(up_channels, axis=0), np.sum(down_channels, axis=0)

    d_channels = [
        values for label, values in atom_dos.items() if label.startswith("d")
    ]

    if not d_channels:
        raise ValueError("Could not identify d-orbital channels.")

    return np.sum(d_channels, axis=0), None


def selected_atoms_d_dos(
    dos_data: DOSData,
    atom_indices: Sequence[int],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Sum d-projected DOS for selected atoms using 1-based indexing."""
    if dos_data.projected_dos is None:
        raise ValueError(
            "Projected DOS not found. Run VASP with a suitable LORBIT setting."
        )

    invalid = [
        atom_index
        for atom_index in atom_indices
        if atom_index < 1 or atom_index > dos_data.nions
    ]

    if invalid:
        raise ValueError(
            f"Invalid atom indices {invalid}. "
            f"Valid atom indices are between 1 and {dos_data.nions}."
        )

    combined_up = np.zeros_like(dos_data.energy, dtype=float)
    combined_down: Optional[np.ndarray] = None

    for atom_index in atom_indices:
        atom_up, atom_down = _d_dos_for_atom(
            dos_data.projected_dos[atom_index - 1]
        )

        combined_up += atom_up

        if atom_down is not None:
            if combined_down is None:
                combined_down = np.zeros_like(atom_down, dtype=float)

            combined_down += atom_down

    return combined_up, combined_down


def d_band_center(
    energy: np.ndarray,
    dos: np.ndarray,
    emin: float,
    emax: float,
) -> float:
    """Calculate the d-band center within the chosen energy range."""
    mask = (energy >= emin) & (energy <= emax)

    if np.count_nonzero(mask) < 2:
        raise ValueError(
            "The selected energy window contains fewer than two data points."
        )

    denominator = np.trapz(dos[mask], energy[mask])

    if np.isclose(denominator, 0.0):
        raise ValueError(
            "The integrated d-DOS is zero in the selected energy window."
        )

    numerator = np.trapz(energy[mask] * dos[mask], energy[mask])

    return float(numerator / denominator)


def main() -> None:
    """Run DOS analysis from the command line."""
    parser = argparse.ArgumentParser(
        description="Analyze a VASP DOSCAR file and calculate d-band centers."
    )

    parser.add_argument(
        "--doscar",
        type=Path,
        default=Path("DOSCAR"),
        help="Path to the DOSCAR file. Default: DOSCAR",
    )

    parser.add_argument(
        "--atoms",
        type=int,
        nargs="+",
        help="Optional 1-based atom indices for projected d-DOS analysis.",
    )

    parser.add_argument(
        "--emin",
        type=float,
        default=-10.0,
        help="Minimum energy relative to the Fermi level in eV. Default: -10",
    )

    parser.add_argument(
        "--emax",
        type=float,
        default=5.0,
        help="Maximum energy relative to the Fermi level in eV. Default: 5",
    )

    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("dos_data.csv"),
        help="Output CSV filename. Default: dos_data.csv",
    )

    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("dos_plot.png"),
        help="Output PNG filename. Default: dos_plot.png",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the DOS plot interactively after saving it.",
    )

    args = parser.parse_args()

    if args.emin >= args.emax:
        raise ValueError("--emin must be smaller than --emax.")

    data = read_doscar(args.doscar)
    energy = data.energy_relative_to_fermi

    output = pd.DataFrame(
        {
            "energy_minus_fermi_eV": energy,
            "total_dos_up": data.total_dos_up,
        }
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        energy,
        data.total_dos_up,
        label="Total DOS up",
    )

    if data.total_dos_down is not None:
        output["total_dos_down"] = data.total_dos_down

        ax.plot(
            energy,
            -data.total_dos_down,
            label="Total DOS down",
        )

    if args.atoms:
        d_up, d_down = selected_atoms_d_dos(data, args.atoms)

        d_total = d_up if d_down is None else d_up + d_down

        center = d_band_center(
            energy=energy,
            dos=d_total,
            emin=args.emin,
            emax=args.emax,
        )

        output["selected_atoms_d_dos_up"] = d_up

        ax.plot(
            energy,
            d_up,
            label=f"d-DOS atoms {args.atoms} up",
        )

        if d_down is not None:
            output["selected_atoms_d_dos_down"] = d_down

            ax.plot(
                energy,
                -d_down,
                label=f"d-DOS atoms {args.atoms} down",
            )

        print(
            f"Combined d-band center for atoms {args.atoms}: "
            f"{center:.6f} eV relative to E_F"
        )

    output.to_csv(args.csv, index=False)

    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=1.0,
        label="Fermi level",
    )

    ax.set_xlim(args.emin, args.emax)
    ax.set_xlabel(r"Energy - $E_F$ (eV)")
    ax.set_ylabel("Density of states")
    ax.legend()

    fig.tight_layout()
    fig.savefig(args.plot, dpi=300)

    if args.show:
        plt.show()

    plt.close(fig)

    print(f"Parsed file: {args.doscar}")
    print(f"Number of ions: {data.nions}")
    print(f"Fermi energy: {data.fermi_energy:.6f} eV")
    print(f"Spin polarized: {data.is_spin_polarized}")
    print(f"CSV written to: {args.csv}")
    print(f"Plot written to: {args.plot}")


if __name__ == "__main__":
    main()
