from integrals import build_hartree_matrices
import numpy as np
from scipy.linalg import eigh


def orthonormalize(C, S, spins):
    """Ortonormalizza gli orbitali con lo stesso spin."""

    C = C.copy()

    for spin in (0, 1):
        idx = np.where(spins == spin)[0]

        if len(idx) == 0:
            continue

        A = C[:, idx]
        M = A.T @ S @ A

        values, vectors = np.linalg.eigh(M)

        if np.any(values <= 0):
            raise ValueError("Orbitali linearmente dipendenti.")

        M_inv_sqrt = (
            vectors
            @ np.diag(1 / np.sqrt(values))
            @ vectors.T
        )

        C[:, idx] = A @ M_inv_sqrt

    return C


def hartree_energy(C, H, eri):
    """Energia totale, senza auto-interazione."""

    density = np.einsum("pi,qi->ipq", C, C)

    energy = np.einsum("ipq,pq->", density, H)

    for i in range(C.shape[1]):
        for j in range(i + 1, C.shape[1]):
            energy += np.einsum(
                "pq,rs,pqrs->",
                density[i],
                density[j],
                eri
            )

    return energy


def hartree_scf(S, T, V, eri, Z,
                max_iter=2000, tol=1e-8):

    H = T + V
    P = S.shape[0]

    # Numero di elettroni per spin
    n_up = (Z + 1) // 2
    n_down = Z // 2

    if max(n_up, n_down) > P:
        raise ValueError(
            "La base contiene troppo poche funzioni "
            "per gli elettroni dello stesso spin."
        )

    spins = np.array(
        [0] * n_up + [1] * n_down
    )

    # Orbitali iniziali
    _, vectors = eigh(H, S)

    C = np.column_stack([
        vectors[:, i] for i in range(n_up)
    ] + [
        vectors[:, i] for i in range(n_down)
    ])

    C = orthonormalize(C, S, spins)

    energy = hartree_energy(C, H, eri)

    for iteration in range(max_iter):

        # Matrici di Hartree dei singoli elettroni
        VH = build_hartree_matrices(C, eri)

        # Fock-like operators di Hartree
        F = H[None, :, :] + VH

        # Gradiente rispetto ai coefficienti orbitali
        G = np.einsum("ipq,qi->pi", F, C)

        # Proiezione del gradiente per conservare
        # l'ortogonalità degli orbitali dello stesso spin
        direction = np.zeros_like(C)

        for spin in (0, 1):
            idx = np.where(spins == spin)[0]

            if len(idx) == 0:
                continue

            A = C[:, idx]
            B = G[:, idx]

            K = A.T @ B
            K = 0.5 * (K + K.T)

            direction[:, idx] = (
                np.linalg.solve(S, B) - A @ K
            )

        gradient_norm = np.linalg.norm(direction)

        if gradient_norm < tol:
            print(f"Converged in {iteration} iterations")
            return energy, C, spins

        # Aggiornamento con scelta adattiva del passo
        step = 1.0
        accepted = False

        while step > 1e-12:

            C_new = orthonormalize(
                C - step * direction, S, spins
            )

            energy_new = hartree_energy(C_new, H, eri)

            if energy_new <= energy + 1e-12:
                accepted = True
                break

            step *= 0.5

        if not accepted:
            if gradient_norm < 10 * tol:
                return energy, C, spins

            raise RuntimeError(
                "Non riesco a trovare un passo che riduca l'energia."
            )

        delta_energy = abs(energy_new - energy)

        C = C_new
        energy = energy_new

        if delta_energy < tol:
            print(f"Energy change below tolerance at iteration {iteration + 1}")
            return energy, C, spins

        if iteration % 10 == 0:
            print(
                f"Iteration {iteration + 1:4d}: "
                f"E = {energy:.10f} Ha"
            )

    raise RuntimeError("Hartree non convergente")

