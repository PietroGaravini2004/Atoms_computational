import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt


# ============================================================
# COMPLETE BASIS SET AND NUMERICAL PARAMETERS
# ============================================================

alpha_all = np.array([
    0.298073,
    1.242567,
    5.782948,
    38.474970
])

HA_TO_EV = 27.211386

r = np.linspace(1e-6, 10.0, 2000)

tol = 1e-8
max_iter = 100


# ============================================================
# ANALYTICAL MATRIX ELEMENTS
# ============================================================

def build_matrices_analytic(alpha):

    P = len(alpha)

    S = np.zeros((P, P))
    T = np.zeros((P, P))
    V_ext = np.zeros((P, P))

    for p in range(P):
        for q in range(P):

            beta = alpha[p] + alpha[q]

            S[p, q] = (
                np.pi / beta
            )**(3/2)

            T[p, q] = (
                3 * alpha[p] * alpha[q] * np.pi**(3/2)
                / beta**(5/2)
            )

            V_ext[p, q] = -4 * np.pi / beta

    return S, T, V_ext


# ============================================================
# NUMERICAL MATRIX ELEMENTS
# ============================================================

def build_matrices_numeric(alpha, r):

    P = len(alpha)

    S = np.zeros((P, P))
    T = np.zeros((P, P))
    V_ext = np.zeros((P, P))

    for p in range(P):
        for q in range(P):

            beta = alpha[p] + alpha[q]

            gaussian = np.exp(-beta * r**2)

            # Overlap matrix
            S[p, q] = np.trapezoid(
                4 * np.pi * r**2 * gaussian,
                r
            )

            # Kinetic-energy matrix
            # T_pq = 1/2 integral grad(chi_p).grad(chi_q) d^3r
            T[p, q] = np.trapezoid(
                8 * np.pi * alpha[p] * alpha[q]
                * r**4 * gaussian,
                r
            )

            # Electron-nucleus potential
            # Nuclear charge Z = 2
            V_ext[p, q] = np.trapezoid(
                -8 * np.pi * r * gaussian,
                r
            )

    return S, T, V_ext


# ============================================================
# SCF FUNCTION
# ============================================================

def run_scf(P, method="analytic"):

    # Use the first P basis functions
    alpha = alpha_all[:P]

    # --------------------------------------------------------
    # Build one-electron matrices
    # --------------------------------------------------------

    if method == "analytic":

        S, T, V_ext = build_matrices_analytic(alpha)

    elif method == "numeric":

        S, T, V_ext = build_matrices_numeric(alpha, r)

    else:

        raise ValueError(
            "method must be 'analytic' or 'numeric'"
        )

    h = T + V_ext

    # --------------------------------------------------------
    # Reconstruct orbital
    # --------------------------------------------------------

    def build_orbital(C):

        phi = np.zeros_like(r)

        for p in range(P):
            phi += C[p] * np.exp(-alpha[p] * r**2)

        return phi

    # --------------------------------------------------------
    # Hartree potential
    # --------------------------------------------------------

    def hartree_potential(n):

        VH = np.zeros_like(r)

        for k in range(len(r)):

            denominator = np.maximum(r[k], r)

            integrand = r**2 * n / denominator

            VH[k] = 4 * np.pi * np.trapezoid(
                integrand, r
            )

        return VH

    # --------------------------------------------------------
    # Hartree matrix
    # --------------------------------------------------------

    def hartree_matrix(VH):

        V_H = np.zeros((P, P))

        for p in range(P):
            for q in range(P):

                chi_p = np.exp(-alpha[p] * r**2)
                chi_q = np.exp(-alpha[q] * r**2)

                integrand = (
                    4 * np.pi * r**2
                    * chi_p * VH * chi_q
                )

                V_H[p, q] = np.trapezoid(
                    integrand, r
                )

        return V_H

    # --------------------------------------------------------
    # Initial guess
    # --------------------------------------------------------

    C_old = np.zeros(P)
    C_old[0] = 1.0

    C_old /= np.sqrt(C_old @ S @ C_old)

    phi_old = build_orbital(C_old)
    n_old = np.abs(phi_old)**2

    # --------------------------------------------------------
    # SCF iterations
    # --------------------------------------------------------

    diff = 1.0
    iteration = 0
    energy_history = []

    while diff > tol and iteration < max_iter:

        # Hartree potential from old density
        VH = hartree_potential(n_old)

        # Hartree matrix
        V_H = hartree_matrix(VH)

        # Effective Hamiltonian
        H = h + V_H

        # Generalized eigenvalue problem
        eigenvalues, eigenvectors = eigh(H, S)

        # Ground-state orbital coefficients
        C_new = eigenvectors[:, 0]

        # New orbital and density
        phi_new = build_orbital(C_new)
        n_new = np.abs(phi_new)**2

        # Density convergence
        diff = np.max(np.abs(n_new - n_old))

        # Total energy
        E_one = C_new @ h @ C_new

        VH_new = hartree_potential(n_new)

        J = np.trapezoid(
            4 * np.pi * r**2 * n_new * VH_new,
            r
        )

        E_total = 2 * E_one + J

        energy_history.append(E_total)

        # Update density and coefficients
        C_old = C_new.copy()
        n_old = n_new.copy()

        iteration += 1

    # --------------------------------------------------------
    # Final quantities
    # --------------------------------------------------------

    phi_final = build_orbital(C_old)
    density_final = np.abs(phi_final)**2

    radial_density = (
        4 * np.pi * r**2 * density_final
    )

    return {
        "P": P,
        "method": method,
        "energy": E_total,
        "orbital_energy": eigenvalues[0],
        "C": C_old,
        "phi": phi_final,
        "density": density_final,
        "radial_density": radial_density,
        "energy_history": energy_history,
        "iterations": iteration,
        "converged": diff <= tol,
        "density_difference": diff
    }


# ============================================================
# RUN BOTH METHODS
# ============================================================


results_analytic = []
results_numeric = []

print("\nHELIUM SCF: ANALYTICAL VS NUMERICAL INTEGRALS\n")

for P in range(1, 5):

    result_a = run_scf(P, method="analytic")
    result_n = run_scf(P, method="numeric")

    results_analytic.append(result_a)
    results_numeric.append(result_n)

    delta_E = (
        result_n["energy"] - result_a["energy"]
    )

    print(f"P = {P}")

    print(
        f"  Analytical energy = "
        f"{result_a['energy'] * HA_TO_EV:.8f} eV"
    )

    print(
        f"  Numerical energy  = "
        f"{result_n['energy'] * HA_TO_EV:.8f} eV"
    )

    print(
        f"  Delta E            = "
        f"{delta_E * HA_TO_EV:+.6e} eV"
    )

    print(
        f"  Iterations         = "
        f"{result_a['iterations']} / "
        f"{result_n['iterations']}"
    )

    print(
        f"  Converged          = "
        f"{result_a['converged']} / "
        f"{result_n['converged']}"
    )

    print()


# ============================================================
# ENERGY COMPARISON
# ============================================================

P_values = [
    result["P"] for result in results_analytic
]

energies_analytic = [
    result["energy"] for result in results_analytic
]

energies_numeric = [
    result["energy"] for result in results_numeric
]

plt.figure(figsize=(8, 6))

plt.plot(
    P_values,
    energies_analytic,
    marker="o",
    label="Analytical integrals"
)

plt.plot(
    P_values,
    energies_numeric,
    marker="s",
    linestyle="--",
    label="Numerical integrals"
)

plt.axhline(
    -2.903,
    color="black",
    linestyle=":",
    label="Exact helium energy (-2.903 Ha)"
)

plt.xlabel("Number of basis functions P")
plt.ylabel("Ground-state energy (Ha)")
plt.title("Helium: analytical vs numerical integrals")

plt.xticks(P_values)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# ENERGY DIFFERENCE
# ============================================================

delta_energies = [
    result_n["energy"] - result_a["energy"]
    for result_a, result_n
    in zip(results_analytic, results_numeric)
]

plt.figure(figsize=(8, 5))

plt.plot(
    P_values,
    delta_energies,
    marker="o"
)

plt.axhline(0, color="black", linestyle="--")

plt.xlabel("Number of basis functions P")
plt.ylabel("E_numeric - E_analytic (Ha)")
plt.title("Energy difference between integration methods")

plt.xticks(P_values)
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# ORBITAL COMPARISON
# ============================================================

plt.figure(figsize=(8, 6))

for result in results_analytic:
    plt.plot(
        r,
        result["phi"],
        label=f"P = {result['P']}"
    )

plt.xlabel(r"$r\ (a_0)$")
plt.ylabel(r"$\phi(r)$")
plt.title("Helium ground-state orbital: analytical method")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# RADIAL PROBABILITY DENSITY
# ============================================================

plt.figure(figsize=(8, 6))

for result in results_analytic:
    plt.plot(
        r,
        result["radial_density"],
        label=f"P = {result['P']}"
    )

plt.xlabel(r"$r\ (a_0)$")
plt.ylabel(r"$4\pi r^2 |\phi(r)|^2$")
plt.title("Helium radial probability density")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

