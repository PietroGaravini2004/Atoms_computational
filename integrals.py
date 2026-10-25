import numpy as np
from math import gamma
from math import pi, factorial
from basis import double_factorial
from math import pi, sqrt, factorial, comb
from scipy.special import beta as beta_function

def S(basis):
    P = basis.n_functions

    S = np.zeros((P, P))

    for p in range(P):
        for q in range(P):

            chi_p = basis.functions[p]
            chi_q = basis.functions[q]

            for primitive_p in chi_p.primitives:
                for primitive_q in chi_q.primitives:

                    alpha = primitive_p.alpha
                    beta = primitive_q.alpha
                    gamma = alpha + beta

                    angular_p = chi_p.angular
                    angular_q = chi_q.angular

                    overlap = 1.0

                    for axis in range(3):

                        n = angular_p[axis] + angular_q[axis]

                        if n % 2 != 0:
                            overlap = 0.0
                            break

                        overlap *= (
                            np.sqrt(np.pi)
                            * np.prod(np.arange(1, n, 2))
                            / (
                                2 ** (n // 2)
                                * gamma ** ((n + 1) / 2)
                            )
                        )

                    S[p, q] += (
                        primitive_p.coefficient
                        * primitive_q.coefficient
                        * primitive_p.normalization
                        * primitive_q.normalization
                        * overlap
                    )

    return S


def gaussian_integral(n, gamma):
    """Integrale di x^n * exp(-gamma*x^2) su tutta la retta."""
    if n < 0 or n % 2 != 0:
        return 0.0

    double_fact = np.prod(np.arange(1, n, 2))
    return (
        np.sqrt(np.pi)
        * double_fact
        / (2 ** (n // 2) * gamma ** ((n + 1) / 2))
    )


def kinetic_primitive(primitive_p, primitive_q, angular_p, angular_q):
    """Integrale cinetico tra due gaussiane primitive cartesiane."""

    alpha = primitive_p.alpha
    beta = primitive_q.alpha
    gamma = alpha + beta

    def overlap_angular(lp, lq):
        result = 1.0
        for axis in range(3):
            result *= gaussian_integral(
                lp[axis] + lq[axis], gamma
            )
        return result

    kinetic = 0.0

    for axis in range(3):
        l = angular_q[axis]

        # Termine derivante da x^(l-2)
        if l >= 2:
            shifted = list(angular_q)
            shifted[axis] -= 2
            kinetic += (
                l * (l - 1)
                * overlap_angular(angular_p, shifted)
            )

        # Termine derivante da x^l
        kinetic += (
            -2 * beta * (2 * l + 1)
            * overlap_angular(angular_p, angular_q)
        )

        # Termine derivante da x^(l+2)
        shifted = list(angular_q)
        shifted[axis] += 2
        kinetic += (
            4 * beta**2
            * overlap_angular(angular_p, shifted)
        )

    return (
        -0.5
        * primitive_p.normalization
        * primitive_q.normalization
        * kinetic
    )


def T(basis):
    """Costruisce la matrice dell'energia cinetica."""

    P = basis.n_functions
    matrix = np.zeros((P, P))

    for p in range(P):
        for q in range(P):

            chi_p = basis.functions[p]
            chi_q = basis.functions[q]

            for primitive_p in chi_p.primitives:
                for primitive_q in chi_q.primitives:

                    integral = kinetic_primitive(
                        primitive_p,
                        primitive_q,
                        chi_p.angular,
                        chi_q.angular
                    )

                    matrix[p, q] += (
                        primitive_p.coefficient
                        * primitive_q.coefficient
                        * integral
                    )

    return matrix

def nuclear_primitive(primitive_p, primitive_q, angular_p, angular_q, Z):
    """Integrale del potenziale nucleare tra due primitive."""

    alpha = primitive_p.alpha
    beta = primitive_q.alpha
    gamma = alpha + beta

    powers = [
        angular_p[axis] + angular_q[axis]
        for axis in range(3)
    ]

    # Integrale nullo per parità
    if any(n % 2 != 0 for n in powers):
        return 0.0

    i, j, k = [n // 2 for n in powers]
    m = i + j + k

    angular_factor = (
        double_factorial(2 * i - 1)
        * double_factorial(2 * j - 1)
        * double_factorial(2 * k - 1)
    )

    integral = (
    -2 * pi * Z
    * factorial(m)
    * angular_factor
    / (
        double_factorial(2 * m + 1)
        * gamma ** (m + 1)
    )
)

    return (
        primitive_p.normalization
        * primitive_q.normalization
        * integral
    )


def V(basis):
    """Matrice del potenziale esterno elettrone-nucleo."""

    P = basis.n_functions
    matrix = np.zeros((P, P))

    for p in range(P):
        for q in range(P):

            chi_p = basis.functions[p]
            chi_q = basis.functions[q]

            for primitive_p in chi_p.primitives:
                for primitive_q in chi_q.primitives:

                    integral = nuclear_primitive(
                        primitive_p,
                        primitive_q,
                        chi_p.angular,
                        chi_q.angular,
                        basis.Z
                    )

                    matrix[p, q] += (
                        primitive_p.coefficient
                        * primitive_q.coefficient
                        * integral
                    )

    return matrix



import numpy as np


def build_density_matrices(C):
    """
    Costruisce la matrice densità di ogni elettrone.

    C[p, i] = coefficiente della funzione di base p
              nell'orbitale dell'elettrone i.
    """
    return np.einsum("pi,qi->ipq", C, C)


def build_hartree_matrices(C, eri):
    """
    Costruisce la matrice di Hartree per ogni elettrone,
    escludendo l'interazione dell'elettrone con sé stesso.
    """

    densities = build_density_matrices(C)

    total_density = np.sum(densities, axis=0)

    hartree_matrices = []

    for i in range(C.shape[1]):

        other_density = total_density - densities[i]

        VH = np.einsum(
            "rs,pqrs->pq",
            other_density,
            eri
        )

        hartree_matrices.append(VH)

    return np.array(hartree_matrices)

def eri_primitive(angulars, exponents):
    """
    Integrale elettronico tra quattro gaussiane primitive
    cartesiane, tutte centrate nell'origine.

    Non include i fattori di normalizzazione.
    """

    a1, a2, a3, a4 = angulars
    alpha, beta, gamma, delta = exponents

    u = alpha + beta
    v = gamma + delta

    terms = {(0, 0, 0): 1.0}

    # Calcolo dei momenti gaussiani nelle tre direzioni
    for axis in range(3):

        A = a1[axis] + a2[axis]
        B = a3[axis] + a4[axis]

        axis_terms = {}

        for k in range(min(A, B) + 1):

            if (A - k) % 2 or (B - k) % 2:
                continue

            i = (A - k) // 2
            j = (B - k) // 2

            coeff = (
                factorial(A) * factorial(B)
                / (
                    factorial(i)
                    * factorial(j)
                    * factorial(k)
                    * 4 ** (i + j)
                    * 2 ** k
                )
            )

            axis_terms[(i, j, k)] = coeff

        if not axis_terms:
            return 0.0

        new_terms = {}

        for (i, j, k), c in terms.items():
            for (di, dj, dk), dc in axis_terms.items():

                key = (i + di, j + dj, k + dk)

                new_terms[key] = (
                    new_terms.get(key, 0.0) + c * dc
                )

        terms = new_terms

    # Integrazione analitica del termine coulombiano
    result = 0.0

    for (i, j, k), coeff in terms.items():

        h = 1.5 + i + j + k

        for a in range(i + 1):
            for b in range(j + 1):

                n = k + a + b

                integral = (
                    0.5
                    * (u * v) ** (n + 0.5 - h)
                    / (u + v) ** (n + 0.5)
                    * beta_function(
                        n + 0.5,
                        h - n - 0.5
                    )
                )

                result += (
                    coeff
                    * comb(i, a)
                    * comb(j, b)
                    * v ** (i - a)
                    * u ** (j - b)
                    * integral
                )

    return 2 * pi**3 / sqrt(pi) * result


def eri_contracted(chi_p, chi_q, chi_r, chi_s):
    """
    Integrale elettronico tra quattro funzioni
    STO-3G contratte.
    """

    angulars = (
        chi_p.angular,
        chi_q.angular,
        chi_r.angular,
        chi_s.angular
    )

    result = 0.0

    for gp in chi_p.primitives:
        for gq in chi_q.primitives:
            for gr in chi_r.primitives:
                for gs in chi_s.primitives:

                    exponents = (
                        gp.alpha,
                        gq.alpha,
                        gr.alpha,
                        gs.alpha
                    )

                    integral = eri_primitive(
                        angulars, exponents
                    )

                    factor = 1.0

                    for g in (gp, gq, gr, gs):
                        factor *= (
                            g.coefficient * g.normalization
                        )

                    result += factor * integral

    return result


def ERI(basis):
    """
    Costruisce il tensore ERI[p,q,r,s] = (pq|rs).
    """

    P = basis.n_functions

    eri = np.zeros((P, P, P, P))

    for p in range(P):
        for q in range(P):
            for r in range(P):
                for s in range(P):

                    eri[p, q, r, s] = eri_contracted(
                        basis.functions[p],
                        basis.functions[q],
                        basis.functions[r],
                        basis.functions[s]
                    )

    return eri