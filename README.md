# DFT-DOS-tools

Python tools for extracting, analyzing, and visualizing density of states data from VASP calculations.

The repository provides a lightweight workflow for:

- parsing VASP `DOSCAR` files;
- detecting spin-polarized and non-spin-polarized calculations;
- extracting atom-projected d-orbital density of states;
- calculating d-band centers relative to the Fermi level;
- exporting processed DOS data to CSV;
- generating DOS plots suitable for further analysis.

The script does not require ASE.

## Installation

Clone the repository:

```bash
git clone https://github.com/kapil4bits/DFT-DOS-tools.git
cd DFT-DOS-tools
