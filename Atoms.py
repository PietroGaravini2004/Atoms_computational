
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt


# ============================================================
# BASIS SET AND NUMERICAL PARAMETERS
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
max_iter = 200
mixing = 0.3


# ============================================================
# USER INPUT
# ============================================================

Z = int(input("Nuclear charge Z: "))
N = int(input("Number of electrons N: "))
P = int(input("Number of basis functions P (1-4): "))

if Z < 1:
    raise ValueError("Z must be a positive integer.")

if N < 1:
    raise ValueError("N must be at least 1.")

if P < 1 or P > len(alpha_all):
    raise ValueError(f"P must be between 1 and {len(alpha_all)}.")

# Number of occupied spatial orbitals
n_orb = (N + 1) // 2

if n_orb > P:
    raise ValueError(
        f"Not enough basis functions: N={N} requires "
        f"at least {n_orb} spatial orbitals, so choose P >= {n_orb}."
    )

# Occupation numbers: 2 electrons per orbital, except possibly the last
occupations = np.full(n_orb, 2.0)
if N % 2 == 1:
    occupations[-1] = 1.0

alpha = alpha_all[:P]


# ============================================================
# ANALYTICAL MATRIX ELEMENTS
# ============================================================

def build_matrices(alpha, Z):

    P = len(alpha)

    S = np.zeros((P, P))
    T = np.zeros((P, P))
    V_ext = np.zeros((P, P))

    for p in range(P):
        for q in range(P):

            beta = alpha[p] + alpha[q]

            S[p, q] = (np.pi / beta)**(3 / 2)

            T[p, q] = (
                3 * alpha[p] * alpha[q] * np.pi**(3 / 2)
                / beta**(5 / 2)
            )

            V_ext[p, q] = -2 * np.pi * Z / beta

    return S, T, V_ext


# ============================================================
# SCF CALCULATION
# ============================================================

def run_scf(Z, N, P):

    if Z < 1 or N < 1:
        raise ValueError("Z and N must be positive integers.")

    if P < 1 or P > len(alpha_all):
        raise ValueError("Invalid basis size P.")

    n_orb = (N + 1) // 2

    if n_orb > P:
        raise ValueError("The basis is too small for the chosen N.")

    alpha = alpha_all[:P]

    occupations = np.full(n_orb, 2.0)
    if N % 2 == 1:
        occupations[-1] = 1.0

    S, T, V_ext = build_matrices(alpha, Z)
    h = T + V_ext

    # --------------------------------------------------------
    # Build all occupied orbitals
    # C has shape (n_orb, P)
    # --------------------------------------------------------

    def build_orbital(C):

        phi = np.zeros((n_orb, len(r)))

        for i in range(n_orb):
            for p in range(P):
                phi[i] += C[i, p] * np.exp(-alpha[p] * r**2)

        return phi

    # --------------------------------------------------------
    # Total electron density
    # --------------------------------------------------------

    def build_density(phi):

        density = np.zeros(len(r))

        for i in range(n_orb):
            density += occupations[i] * np.abs(phi[i])**2

        return density

    # --------------------------------------------------------
    # Hartree potential
    # --------------------------------------------------------

    def hartree_potential(density):

        VH = np.zeros_like(r)

        for k in range(len(r)):

            denominator = np.maximum(r[k], r)
            integrand = r**2 * density / denominator

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
            chi_p = np.exp(-alpha[p] * r**2)

            for q in range(P):
                chi_q = np.exp(-alpha[q] * r**2)

                integrand = (
                    4 * np.pi * r**2
                    * chi_p * VH * chi_q
                )

                V_H[p, q] = np.trapezoid(integrand, r)

        return V_H

    # --------------------------------------------------------
    # Initial guess
    # --------------------------------------------------------

    C_old = np.zeros((n_orb, P))

    for i in range(n_orb):
        C_old[i, i] = 1.0
        C_old[i] /= np.sqrt(C_old[i] @ S @ C_old[i])

    phi_old = build_orbital(C_old)
    density_old = build_density(phi_old)

    # --------------------------------------------------------
    # SCF iterations
    # --------------------------------------------------------

    diff = np.inf
    iteration = 0
    energy_history = []

    while diff > tol and iteration < max_iter:

        VH = hartree_potential(density_old)
        V_H = hartree_matrix(VH)

        H = h + V_H

        eigenvalues, eigenvectors = eigh(H, S)

        # Store orbitals as rows
        C_new = eigenvectors[:, :n_orb].T

        phi_new = build_orbital(C_new)
        density_new = build_density(phi_new)

        diff = np.max(np.abs(density_new - density_old))

        # One-electron contribution
        E_one = 0.0

        for i in range(n_orb):
            E_one += (
                occupations[i]
                * (C_new[i] @ h @ C_new[i])
            )

        # Electron-electron Hartree contribution
        VH_new = hartree_potential(density_new)

        J = np.trapezoid(
            4 * np.pi * r**2 * density_new * VH_new,
            r
        )

        E_total = E_one + 0.5 * J
        energy_history.append(E_total)

        # Mix densities to improve SCF stability
        density_old = (
            (1 - mixing) * density_old
            + mixing * density_new
        )

        C_old = C_new.copy()
        iteration += 1

    # --------------------------------------------------------
    # Final results
    # --------------------------------------------------------

    phi_final = build_orbital(C_old)
    density_final = build_density(phi_final)
    radial_density = 4 * np.pi * r**2 * density_final

    electron_count = np.trapezoid(radial_density, r)

    return {
        "Z": Z,
        "N": N,
        "P": P,
        "occupations": occupations,
        "energy": E_total,
        "coefficients": C_old,
        "orbitals": phi_final,
        "density": density_final,
        "radial_density": radial_density,
        "energy_history": energy_history,
        "iterations": iteration,
        "converged": diff <= tol,
        "electron_count": electron_count
    }


# ============================================================
# RUN CALCULATION
# ============================================================

result = run_scf(Z, N, P)

print("\nSCF RESULTS")
print("-" * 50)
print(f"Z = {Z}")
print(f"N = {N}")
print(f"P = {P}")
print(f"Occupations = {result['occupations']}")
print(f"Total energy = {result['energy']:.8f} Hartree")
print(f"Total energy = {result['energy'] * HA_TO_EV:.6f} eV")
print(f"Iterations = {result['iterations']}")
print(f"Converged = {result['converged']}")
print(f"Integrated electron count = {result['electron_count']:.6f}")


# ============================================================
# PLOT ORBITALS
# ============================================================

plt.figure()

for i in range(n_orb):
    plt.plot(
        r,
        result["orbitals"][i],
        label=f"Orbital {i + 1} (occ. {occupations[i]:.0f})"
    )

plt.xlabel("r (a.u.)")
plt.ylabel("Orbital")
plt.title(f"Occupied orbitals: Z={Z}, N={N}, P={P}")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# ============================================================
# PLOT TOTAL RADIAL ELECTRON DENSITY
# ============================================================

plt.figure()
plt.plot(r, result["radial_density"])
plt.xlabel("r (a.u.)")
plt.ylabel(r"$4\pi r^2 n(r)$")
plt.title("Total radial electron density")
plt.grid()
plt.tight_layout()
plt.show()


# ============================================================
# PLOT TOTAL ELECTRON DENSITY
# ============================================================

plt.figure()
plt.plot(r, result["density"])
plt.xlabel("r (a.u.)")
plt.ylabel(r"$n(r)$")
plt.title("Total electron density")
plt.grid()
plt.tight_layout()
plt.show()

