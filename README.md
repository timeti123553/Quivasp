![Quivasp](img/1.png)

# Quivasp

Quivasp is an open-source Python package for VASP-related workflows, data processing, and analysis.

The project is currently under active development. Its goal is to provide simple and reusable tools for working with VASP calculations and related scientific workflows.

## Installation

Clone the repository:

```bash
git clone https://github.com/timeti123553/Quivasp.git
cd Quivasp
```

Install Quivasp in editable mode:

```bash
python -m pip install -e .
```

## Requirements

Quivasp requires Python 3.9 or later.

Additional dependencies will be added to `pyproject.toml` as the project develops.

## Usage

After installation, Quivasp can be imported in Python:

```python
import quivasp

print(quivasp.__version__)
```

More usage examples and documentation will be added as new features are implemented.

## Project Structure

```text
Quivasp/
├── src/
│   └── quivasp/
│       └── __init__.py
├── tests/
│   └── test_basic.py
├── README.md
├── pyproject.toml
├── CITATION.cff
├── LICENSE
└── .gitignore
```

The main source code is located in:

```text
src/quivasp/
```

Tests are located in:

```text
tests/
```

## Development

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/timeti123553/Quivasp.git
cd Quivasp
python -m pip install -e .
```

To run tests, install `pytest`:

```bash
python -m pip install pytest
pytest
```

## Contributing

Contributions, suggestions, and bug reports are welcome.

If you would like to contribute, you can:

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Add or update tests when appropriate.
5. Submit a pull request.

For bugs or feature requests, please open an issue on GitHub.

## Citation

If you use Quivasp in academic or scientific work, please cite the software.

Citation information is provided in the `CITATION.cff` file. GitHub can also use this file to generate citation information through the **Cite this repository** feature.

## License

Quivasp is released under the Apache License 2.0.

You may use, modify, and redistribute the software under the terms of the license. Please retain the required copyright and license notices when redistributing the software.

See the `LICENSE` file for details.

## Disclaimer

Quivasp is an independent open-source project and is not affiliated with, endorsed by, or maintained by the official VASP development team.

VASP is a separate software package developed and distributed by its respective authors and institutions.
