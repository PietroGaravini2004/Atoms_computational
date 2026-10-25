from dataclasses import dataclass
from math import pi, sqrt, prod
from pathlib import Path


BASIS_FILE = Path(__file__).parent / "sto-3G.txt"

ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe",
]


# Cartesian components: label -> (lx, ly, lz)
ANGULAR_COMPONENTS = {
    "S": [("s", (0, 0, 0))],
    "P": [
        ("px", (1, 0, 0)),
        ("py", (0, 1, 0)),
        ("pz", (0, 0, 1)),
    ],
    "D": [
        ("dxx", (2, 0, 0)),
        ("dyy", (0, 2, 0)),
        ("dzz", (0, 0, 2)),
        ("dxy", (1, 1, 0)),
        ("dxz", (1, 0, 1)),
        ("dyz", (0, 1, 1)),
    ],
}


@dataclass
class PrimitiveGaussian:
    alpha: float
    coefficient: float
    normalization: float


@dataclass
class BasisFunction:
    label: str
    angular: tuple
    center: tuple
    primitives: list


@dataclass
class BasisSet:
    Z: int
    symbol: str
    functions: list

    @property
    def n_functions(self):
        return len(self.functions)


def double_factorial(n):
    """Double factorial, including (-1)!! = 1."""
    if n <= 0:
        return 1
    return prod(range(n, 0, -2))


def primitive_normalization(alpha, angular):
    """Normalization constant for a Cartesian primitive Gaussian."""
    lx, ly, lz = angular
    L = lx + ly + lz

    denominator = (
        double_factorial(2 * lx - 1)
        * double_factorial(2 * ly - 1)
        * double_factorial(2 * lz - 1)
    )

    return (
        (2 * alpha / pi) ** 0.75
        * sqrt((4 * alpha) ** L / denominator)
    )


def read_basis_file(filename=BASIS_FILE):
    return filename.read_text(encoding="utf-8").splitlines()


def get_element_section(lines, Z):
    if not 1 <= Z <= len(ELEMENTS):
        raise ValueError("Z deve essere compreso tra 1 e 54.")

    symbol = ELEMENTS[Z - 1]
    section = []
    inside_element = False

    for line in lines:
        fields = line.split()

        if not fields or line.lstrip().startswith("!"):
            continue

        if fields[0] == "****":
            if inside_element:
                break
            continue

        if len(fields) >= 2 and fields[0] == symbol and fields[1] == "0":
            inside_element = True
            continue

        if inside_element:
            section.append(line.strip())

    if not section:
        raise ValueError(f"Base STO-3G non trovata per {symbol}.")

    return symbol, section


def parse_basis_section(section):
    functions = []
    i = 0

    while i < len(section):
        fields = section[i].split()
        shell = fields[0].upper()
        n_primitives = int(fields[1])
        scale = float(fields[2]) if len(fields) > 2 else 1.0

        if shell not in {"S", "P", "D", "SP"}:
            raise ValueError(f"Shell non riconosciuta: {shell}")

        rows = [
            [
                float(value.replace("D", "E").replace("d", "e"))
                for value in section[i + j].split()
            ]
            for j in range(1, n_primitives + 1)
        ]

        if shell == "SP":
            shell_data = [
                ("S", [(row[0] * scale**2, row[1]) for row in rows]),
                ("P", [(row[0] * scale**2, row[2]) for row in rows]),
            ]
        else:
            shell_data = [
                (shell, [(row[0] * scale**2, row[1]) for row in rows])
            ]

        for angular_shell, data in shell_data:
            for label, angular in ANGULAR_COMPONENTS[angular_shell]:
                primitives = [
                    PrimitiveGaussian(
                        alpha=alpha,
                        coefficient=coefficient,
                        normalization=primitive_normalization(
                            alpha, angular
                        ),
                    )
                    for alpha, coefficient in data
                ]

                functions.append(
                    BasisFunction(
                        label=label,
                        angular=angular,
                        center=(0.0, 0.0, 0.0),
                        primitives=primitives,
                    )
                )

        i += n_primitives + 1

    return functions


def get_basis(Z):
    lines = read_basis_file()
    symbol, section = get_element_section(lines, Z)
    functions = parse_basis_section(section)

    return BasisSet(
        Z=Z,
        symbol=symbol,
        functions=functions,
    )
