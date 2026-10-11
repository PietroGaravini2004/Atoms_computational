import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt


# ============================================================
# COMPLETE BASIS SET
# ============================================================

alpha_all = np.array([
    0.298073,
    1.242567,
    5.782948,
    38.474970
])

r = np.linspace(1e-6, 10.0, 2000)

tol = 1e-8
max_iter = 100


# ============================================================
# SCF FUNCTION
# ============================================================

def run_scf(P):

    # Use only the first P basis functions
    alpha = alpha_all[:P]

    # --------------------------------------------------------
    # Matrices independent of SCF
    # --------------------------------------------------------

    S = np.zeros((P, P))
    T = np.zeros((P, P))
    V_ext = np.zeros((P, P))

    for p in range(P):
        for q in range(P):

            S[p, q] = (
                np.pi / (alpha[p] + alpha[q])
            )**(3/2)

            T[p, q] = (
                3
                * alpha[p]
                * alpha[q]
                * np.pi**(3/2)
                / (alpha[p] + alpha[q])**(5/2)
            )

            V_ext[p, q] = (
                -4 * np.pi
                / (alpha[p] + alpha[q])
            )

    h = T + V_ext


    # --------------------------------------------------------
    # Function to reconstruct orbital
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

            integrand = (
                r**2
                * n
                / denominator
            )

            VH[k] = (
                4
                * np.pi
                * np.trapezoid(integrand, r)
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
                    4
                    * np.pi
                    * r**2
                    * chi_p
                    * VH
                    * chi_q
                )

                V_H[p, q] = np.trapezoid(
                    integrand,
                    r
                )

        return V_H


    # --------------------------------------------------------
    # Initial guess
    # --------------------------------------------------------

    C_old = np.zeros(P)
    C_old[0] = 1.0

    # Normalize
    C_old /= np.sqrt(C_old @ S @ C_old)

    phi_old = build_orbital(C_old)
    n_old = np.abs(phi_old)**2


    # --------------------------------------------------------
    # SCF
    # --------------------------------------------------------

    diff = 1.0
    iteration = 0

    energy_history = []

    while diff > tol and iteration < max_iter:

        # Hartree potential
        VH = hartree_potential(n_old)

        # Hartree matrix
        V_H = hartree_matrix(VH)

        # Effective Hamiltonian
        H = h + V_H

        # Generalized eigenvalue problem
        eigenvalues, eigenvectors = eigh(H, S)

        # Ground-state coefficients
        C_new = eigenvectors[:, 0]

        # New orbital
        phi_new = build_orbital(C_new)

        # New density
        n_new = np.abs(phi_new)**2

        # Convergence
        diff = np.max(
            np.abs(n_new - n_old)
        )

        # ----------------------------------------------
        # Total energy
        # ----------------------------------------------

        E_one = C_new @ h @ C_new

        VH_new = hartree_potential(n_new)

        J = np.trapezoid(
            4
            * np.pi
            * r**2
            * n_new
            * VH_new,
            r
        )

        E_total = 2 * E_one + J

        energy_history.append(E_total)

        # Update
        C_old = C_new.copy()
        n_old = n_new.copy()

        iteration += 1


    # --------------------------------------------------------
    # Final quantities
    # --------------------------------------------------------

    phi_final = build_orbital(C_old)

    density_final = np.abs(phi_final)**2

    radial_density = (
        4
        * np.pi
        * r**2
        * density_final
    )

    return {
        "P": P,
        "energy": E_total,
        "orbital_energy": eigenvalues[0],
        "C": C_old,
        "phi": phi_final,
        "density": density_final,
        "radial_density": radial_density,
        "energy_history": energy_history,
        "iterations": iteration
    }


# ============================================================
# RUN FOR P = 1, 2, 3, 4
# ============================================================

results = []

for P in range(1, 5):

    result = run_scf(P)
    results.append(result)

    print(
        f"P = {P} | "
        f"E = {result['energy']:.8f} Ha | "
        f"iterations = {result['iterations']}"
    )


# ============================================================
# ENERGY COMPARISON
# ============================================================

P_values = [
    result["P"]
    for result in results
]

energies = [
    result["energy"]
    for result in results
]

plt.figure(figsize=(8, 6))

plt.plot(
    P_values,
    energies,
    marker="o",
    label="Hartree"
)

plt.axhline(
    -2.903,
    linestyle="--",
    label="Exact (-2.903 Ha)"
)

plt.xlabel("Number of basis functions P")
plt.ylabel("Ground-state energy (Ha)")
plt.title("Basis-set convergence of helium energy")

plt.xticks(P_values)
plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# ORBITAL COMPARISON
# ============================================================

plt.figure(figsize=(8, 6))

for result in results:

    plt.plot(
        r,
        result["phi"],
        label=f"P = {result['P']}"
    )

plt.xlabel(r"$r\;(a_0)$")
plt.ylabel(r"$\phi(r)$")
plt.title("Helium ground-state orbital")

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# RADIAL PROBABILITY DENSITY COMPARISON
# ============================================================

plt.figure(figsize=(8, 6))

for result in results:

    plt.plot(
        r,
        result["radial_density"],
        label=f"P = {result['P']}"
    )

plt.xlabel(r"$r\;(a_0)$")
plt.ylabel(r"$4\pi r^2 |\phi(r)|^2$")
plt.title("Helium radial probability density")

plt.legend()
plt.grid(True)

plt.show()
