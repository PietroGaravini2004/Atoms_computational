
import numpy as np
import matplotlib.pyplot as plt

from basis import get_basis
from integrals import S, T, V, ERI
from    Hartree import hartree_scf


def radial_probability(basis, C, r):
    """
    Calcola la distribuzione radiale totale:

        P(r) = r^2 * integral rho(r,theta,phi) dOmega

    Per densità sfericamente simmetriche:
        P(r) = 4*pi*r^2*rho(r)

    L'integrale di P(r) su r vale il numero di elettroni.
    """

    # Integrazione angolare
    mu, weights = np.polynomial.legendre.leggauss(20)
    phi = np.linspace(0, 2 * np.pi, 40, endpoint=False)

    probability = np.zeros_like(r)

    for m, w in zip(mu, weights):

        sin_theta = np.sqrt(1 - m**2)

        for angle in phi:

            x = r * sin_theta * np.cos(angle)
            y = r * sin_theta * np.sin(angle)
            z = r * m

            basis_values = np.zeros(
                (basis.n_functions, len(r))
            )

            # Valutazione delle funzioni STO-3G
            for p, chi in enumerate(basis.functions):

                lx, ly, lz = chi.angular

                for primitive in chi.primitives:

                    basis_values[p] += (
                        primitive.coefficient
                        * primitive.normalization
                        * x**lx
                        * y**ly
                        * z**lz
                        * np.exp(-primitive.alpha * r**2)
                    )

            # Orbitali elettronici
            psi = C.T @ basis_values

            # Densità elettronica totale
            rho = np.sum(np.abs(psi)**2, axis=0)

            probability += (
                w * rho * (2 * np.pi / len(phi))
            )

    return r**2 * probability


def main():

    # 1. Scelta dell'atomo
    Z = int(input("Enter the atomic number Z: "))

    basis = get_basis(Z)

    print(f"\nElement: {basis.symbol} (Z = {Z})")
    print(f"Basis functions: {basis.n_functions}")

    # 2. Calcolo degli integrali
    s = S(basis)
    t = T(basis)
    v = V(basis)
    eri = ERI(basis)

    # 3. Algoritmo SCF
    energy, C, spins = hartree_scf(s, t, v, eri, Z)

    # 4. Energia totale
    hartree_to_ev = 27.2114

    print("\n===== RESULTS =====")
    print(f"Energy: {energy:.6f} Ha")
    print(f"Energy: {energy * hartree_to_ev:.6f} eV")

    print("\nOrbital coefficients:")
    print(C)

    print("\nSpins (0=up, 1=down):")
    print(spins)

    # 5. Distribuzione radiale di probabilità
    r = np.linspace(0, 15, 600)

    P_r = radial_probability(basis, C, r)

    # 6. Grafico
    plt.figure(figsize=(8, 5))

    plt.plot(
        r,
        P_r,
        linewidth=2,
        label="Total radial probability"
    )

    plt.xlabel("r (Bohr)")
    plt.ylabel(r"$P(r)$")
    plt.title(
        f"Radial probability distribution - {basis.symbol}"
    )

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

