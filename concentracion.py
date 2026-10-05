import math
import numpy as np


# ============================================================
# 1. VALIDAR NÚMERO DE EMPRESAS
# ============================================================

def validar_N(N):

    if N < 2 or N > 100:
        raise ValueError(
            "El número de empresas debe estar entre 2 y 100."
        )


# ============================================================
# 2. VALIDAR Y NORMALIZAR CUOTAS
# ============================================================

def normalizar_cuotas(cuotas):

    cuotas = list(cuotas)

    if len(cuotas) == 0:
        raise ValueError(
            "La lista de cuotas no puede estar vacía."
        )

    if any(c < 0 for c in cuotas):
        raise ValueError(
            "Las cuotas no pueden ser negativas."
        )

    suma = sum(cuotas)

    if math.isclose(
        suma,
        100,
        rel_tol=1e-9,
        abs_tol=1e-9
    ):
        cuotas = [
            c / 100
            for c in cuotas
        ]

    elif math.isclose(
        suma,
        1,
        rel_tol=1e-9,
        abs_tol=1e-9
    ):
        cuotas = list(cuotas)

    else:
        raise ValueError(
            f"Las cuotas deben sumar 100% o 1. "
            f"La suma actual es {suma}."
        )

    if any(c > 1 for c in cuotas):
        raise ValueError(
            "Una cuota individual no puede superar el 100%."
        )

    return np.array(
        cuotas,
        dtype=float
    )


# ============================================================
# 3. INDICADORES DE CONCENTRACIÓN
# ============================================================

def indicadores_concentracion(
    cuotas,
    k=4
):

    cuotas = normalizar_cuotas(cuotas)

    cuotas = np.sort(
        cuotas
    )[::-1]

    k = min(
        k,
        len(cuotas)
    )

    CR_k = np.sum(
        cuotas[:k]
    )

    IHH = np.sum(
        cuotas ** 2
    )

    if IHH == 0:

        indice_dominancia = 0

    else:

        indice_dominancia = np.sum(
            (cuotas ** 2 / IHH) ** 2
        )

    indice_entropia = -np.sum(
        [
            c * math.log(c)
            for c in cuotas
            if c > 0
        ]
    )

    return {
        "CR_k": float(CR_k),
        "IHH": float(IHH),
        "Indice_Dominancia": float(
            indice_dominancia
        ),
        "Indice_Entropia": float(
            indice_entropia
        )
    }


# ============================================================
# 4. GENERAR UN VECTOR ALEATORIO
# ============================================================

def generar_cuotas_aleatorias(N):

    validar_N(N)

    cuotas = np.random.dirichlet(
        np.ones(N)
    )

    cuotas[-1] = (
        1.0 -
        np.sum(
            cuotas[:-1]
        )
    )

    return cuotas


# ============================================================
# 5. SIMULACIÓN MONTE CARLO
# ============================================================

def simulacion_monte_carlo(
    N,
    M=1000
):

    validar_N(N)

    if M <= 0:
        raise ValueError(
            "El número de iteraciones debe ser mayor que 0."
        )

    resultados = np.random.dirichlet(
        np.ones(N),
        size=M
    )

    resultados[:, -1] = (
        1.0 -
        np.sum(
            resultados[:, :-1],
            axis=1
        )
    )

    return resultados


# ============================================================
# 6. CALCULAR INDICADORES PARA MONTE CARLO
# ============================================================

def calcular_indicadores_simulacion(
    resultados,
    k=4
):

    resultados_indicadores = []

    for escenario in resultados:

        indicadores = indicadores_concentracion(
            escenario,
            k=k
        )

        resultados_indicadores.append(
            indicadores
        )

    return resultados_indicadores