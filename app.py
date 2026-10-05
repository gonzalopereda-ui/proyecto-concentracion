import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

from concentracion import (
    indicadores_concentracion,
    generar_cuotas_aleatorias
)


# ============================================================
# CONFIGURACIÓN DE LA APLICACIÓN
# ============================================================

st.set_page_config(
    page_title="Simulador de Concentración de Mercado",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# ESTILOS
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       FONDO
       ====================================================== */

    .stApp {
        background-color: #f4f8fc;
    }


    /* ======================================================
       SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] {
        background-color: #082b59;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p {
        color: white !important;
    }


    /* ======================================================
       TÍTULOS
       ====================================================== */

    h1, h2, h3 {
        color: #082b59 !important;
    }


    /* ======================================================
       BOTONES
       ====================================================== */

    .stButton > button {
        background-color: #087f5b !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
    }

    .stButton > button:hover {
        background-color: #066b4c !important;
        color: white !important;
    }


    /* ======================================================
       MÉTRICAS
       ====================================================== */

    [data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #d7e3ef;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.05);
    }

    [data-testid="stMetricLabel"] {
        color: #526579 !important;
    }

    [data-testid="stMetricValue"] {
        color: #082b59 !important;
    }


    /* ======================================================
       TABS
       ====================================================== */

    button[data-baseweb="tab"] {
        color: #082b59 !important;
        font-weight: 600 !important;
    }


    /* ======================================================
       EXPANDERS
       ====================================================== */

    div[data-testid="stExpander"] {
        background-color: white;
        border: 1px solid #d7e3ef;
        border-radius: 12px;
    }


    /* ======================================================
       TABLAS
       ====================================================== */

    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def calcular_indicadores_vectorizados(resultados, k):
    """
    Calcula los cuatro indicadores para todas las simulaciones
    utilizando operaciones vectorizadas de NumPy.
    """

    # Ordenar cuotas de mayor a menor
    ordenadas = np.sort(
        resultados,
        axis=1
    )[:, ::-1]

    # --------------------------------------------------------
    # CRk
    # --------------------------------------------------------

    crk = np.sum(
        ordenadas[:, :k],
        axis=1
    )

    # --------------------------------------------------------
    # IHH
    # --------------------------------------------------------

    ihh = np.sum(
        resultados ** 2,
        axis=1
    )

    # --------------------------------------------------------
    # DOMINANCIA
    # --------------------------------------------------------

    dominancia = np.sum(
        (
            (resultados ** 2) /
            ihh[:, None]
        ) ** 2,
        axis=1
    )

    # --------------------------------------------------------
    # ENTROPÍA
    # --------------------------------------------------------

    cuotas_seguras = np.where(
        resultados > 0,
        resultados,
        1
    )

    entropia = -np.sum(
        resultados *
        np.log(cuotas_seguras),
        axis=1
    )

    return {
        "CRk": crk,
        "IHH": ihh,
        "Dominancia": dominancia,
        "Entropia": entropia
    }


def interpretar_ihh(valor):
    """
    Criterio de referencia para IHH normalizado.
    """

    if valor < 0.15:

        return (
            "🟢 Baja concentración",
            "El IHH se encuentra por debajo de 0,15."
        )

    elif valor < 0.25:

        return (
            "🟡 Concentración moderada",
            "El IHH se encuentra entre 0,15 y 0,25."
        )

    else:

        return (
            "🔴 Alta concentración",
            "El IHH es igual o superior a 0,25."
        )


def interpretar_crk(valor):

    if valor < 0.40:

        return (
            "🟢 Baja concentración",
            "Las principales empresas concentran menos del 40%."
        )

    elif valor < 0.70:

        return (
            "🟡 Concentración moderada",
            "Las principales empresas concentran entre 40% y 70%."
        )

    else:

        return (
            "🔴 Alta concentración",
            "Las principales empresas concentran 70% o más."
        )


def interpretar_dominancia(valor):

    if valor < 0.20:

        return (
            "🟢 Baja dominancia",
            "El indicador presenta un nivel relativamente bajo."
        )

    elif valor < 0.50:

        return (
            "🟡 Dominancia moderada",
            "El indicador presenta un nivel intermedio."
        )

    else:

        return (
            "🔴 Alta dominancia",
            "El indicador presenta un nivel elevado."
        )


def interpretar_entropia(valor, N):

    max_entropia = np.log(N)

    proporcion = valor / max_entropia

    if proporcion >= 0.75:

        return (
            "🟢 Baja concentración",
            "La distribución de las cuotas es relativamente equilibrada."
        )

    elif proporcion >= 0.50:

        return (
            "🟡 Concentración moderada",
            "Existe una distribución intermedia de las cuotas."
        )

    else:

        return (
            "🔴 Alta concentración",
            "Las cuotas se encuentran relativamente concentradas."
        )


def generar_excel(
    resultados,
    indicadores_mc,
    cuotas_particular,
    indicadores_particular
):
    """
    Genera archivo Excel descargable.
    """

    cuotas_df = pd.DataFrame(
        resultados,
        columns=[
            f"Empresa_{i + 1}"
            for i in range(resultados.shape[1])
        ]
    )

    indicadores_df = pd.DataFrame(
        {
            "CRk": indicadores_mc["CRk"],
            "IHH": indicadores_mc["IHH"],
            "Dominancia": indicadores_mc["Dominancia"],
            "Entropia": indicadores_mc["Entropia"]
        }
    )

    simulaciones_df = pd.concat(
        [
            cuotas_df,
            indicadores_df
        ],
        axis=1
    )

    particular_df = pd.DataFrame(
        {
            "Empresa": [
                f"Empresa {i + 1}"
                for i in range(len(cuotas_particular))
            ],
            "Cuota": cuotas_particular
        }
    )

    particular_indicadores_df = pd.DataFrame(
        {
            "Indicador": [
                "CRk",
                "IHH",
                "Dominancia",
                "Entropia"
            ],
            "Valor": [
                indicadores_particular["CR_k"],
                indicadores_particular["IHH"],
                indicadores_particular["Indice_Dominancia"],
                indicadores_particular["Indice_Entropia"]
            ]
        }
    )

    archivo = BytesIO()

    with pd.ExcelWriter(
        archivo,
        engine="openpyxl"
    ) as writer:

        simulaciones_df.to_excel(
            writer,
            sheet_name="Monte Carlo",
            index=False
        )

        particular_df.to_excel(
            writer,
            sheet_name="Caso Particular",
            index=False
        )

        particular_indicadores_df.to_excel(
            writer,
            sheet_name="Indicadores",
            index=False
        )

    archivo.seek(0)

    return archivo


# ============================================================
# HEADER
# ============================================================

st.title(
    "📊 Simulador de Concentración de Mercado"
)

st.markdown(
    "### Análisis de estructuras de mercado mediante indicadores de concentración y simulación Monte Carlo"
)

st.info(
    "Esta herramienta permite analizar la concentración de un mercado "
    "mediante CRk, IHH, Índice de Dominancia e Índice de Entropía."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Configuración")

    st.write(
        "Configure los parámetros del mercado."
    )

    N = st.number_input(
        "Número de empresas",
        min_value=2,
        max_value=100,
        value=4,
        step=1
    )

    M = st.number_input(
        "Iteraciones Monte Carlo",
        min_value=100,
        max_value=100000,
        value=1000,
        step=100
    )

    k = st.number_input(
        "Valor de k para CRk",
        min_value=1,
        max_value=int(N),
        value=min(4, int(N)),
        step=1
    )

    seed = st.number_input(
        "Semilla aleatoria",
        min_value=0,
        max_value=999999,
        value=12345,
        step=1,
        help="Permite reproducir exactamente una simulación."
    )

    st.divider()

    st.subheader("📌 Indicadores")

    indicadores_seleccionados = st.multiselect(
        "Indicadores a mostrar",
        [
            "CRk",
            "IHH",
            "Índice de Dominancia",
            "Índice de Entropía"
        ],
        default=[
            "CRk",
            "IHH",
            "Índice de Dominancia",
            "Índice de Entropía"
        ]
    )

    st.divider()

    st.write(
        f"**Empresas:** {int(N)}"
    )

    st.write(
        f"**Iteraciones:** {int(M):,}"
    )

    st.write(
        f"**k:** {int(k)}"
    )

    st.write(
        f"**Seed:** {int(seed)}"
    )

    if M > 10000:

        st.warning(
            "⚠️ Un número elevado de iteraciones "
            "puede aumentar el tiempo de procesamiento "
            "y el consumo de recursos."
        )


# ============================================================
# RESUMEN
# ============================================================

st.subheader(
    "📌 Resumen de configuración"
)

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Empresas",
        int(N)
    )


with col2:

    st.metric(
        "Simulaciones",
        f"{int(M):,}"
    )


with col3:

    st.metric(
        "k",
        int(k)
    )


with col4:

    st.metric(
        "Seed",
        int(seed)
    )


# ============================================================
# CASO PARTICULAR
# ============================================================

st.divider()

st.subheader(
    "1️⃣ Caso particular"
)

st.write(
    "Defina las participaciones de mercado que desea analizar."
)


opcion_caso = st.radio(
    "Seleccione una alternativa:",
    [
        "Ingresar cuotas manualmente",
        "Generar mercado aleatorio"
    ],
    horizontal=True
)


cuotas_particulares = None


# ============================================================
# CUOTAS MANUALES
# ============================================================

if opcion_caso == "Ingresar cuotas manualmente":

    st.info(
        "💡 Las cuotas deben sumar exactamente 100%."
    )

    columnas = st.columns(
        min(int(N), 5)
    )

    cuotas_ingresadas = []

    for i in range(int(N)):

        with columnas[i % len(columnas)]:

            cuota = st.number_input(
                f"Empresa {i + 1}",
                min_value=0.0,
                max_value=100.0,
                value=100.0 / int(N),
                step=0.1,
                format="%.1f",
                key=f"cuota_{i}"
            )

            cuotas_ingresadas.append(
                cuota
            )


    suma_cuotas = sum(
        cuotas_ingresadas
    )


    if abs(suma_cuotas - 100) < 0.000001:

        st.success(
            f"✓ Mercado válido. "
            f"La suma de las cuotas es "
            f"{suma_cuotas:.2f}%."
        )

        cuotas_particulares = (
            np.array(
                cuotas_ingresadas
            ) / 100
        )

    else:

        st.warning(
            f"⚠️ Las cuotas suman "
            f"{suma_cuotas:.2f}%. "
            f"Debe sumar exactamente 100%."
        )


# ============================================================
# MERCADO ALEATORIO
# ============================================================

else:

    st.info(
        "🎲 Se generará un mercado aleatorio "
        "con cuotas que suman exactamente 100%."
    )

    if st.button(
        "🎲 Generar mercado aleatorio",
        use_container_width=True
    ):

        np.random.seed(
            int(seed)
        )

        st.session_state[
            "cuotas_aleatorias"
        ] = generar_cuotas_aleatorias(
            int(N)
        )


    if "cuotas_aleatorias" in st.session_state:

        cuotas_particulares = (
            st.session_state[
                "cuotas_aleatorias"
            ]
        )

        datos = pd.DataFrame(
            {
                "Empresa": [
                    f"Empresa {i + 1}"
                    for i in range(int(N))
                ],
                "Cuota": [
                    f"{c * 100:.2f}%"
                    for c in cuotas_particulares
                ]
            }
        )

        st.dataframe(
            datos,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# ANÁLISIS
# ============================================================

if cuotas_particulares is not None:

    # ========================================================
    # INDICADORES DEL CASO PARTICULAR
    # ========================================================

    st.divider()

    st.subheader(
        "2️⃣ Indicadores del caso particular"
    )


    indicadores_particular = (
        indicadores_concentracion(
            cuotas_particulares,
            k=int(k)
        )
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            f"CR{int(k)}",
            f"{indicadores_particular['CR_k']:.4f}"
        )


    with col2:

        st.metric(
            "IHH",
            f"{indicadores_particular['IHH']:.4f}"
        )


    with col3:

        st.metric(
            "Dominancia",
            f"{indicadores_particular['Indice_Dominancia']:.4f}"
        )


    with col4:

        st.metric(
            "Entropía",
            f"{indicadores_particular['Indice_Entropia']:.4f}"
        )


    # ========================================================
    # EXPLICACIÓN DE LOS RESULTADOS
    # ========================================================

    with st.expander(
        "📖 Ver interpretación de los indicadores"
    ):

        st.write(
            f"**CR{int(k)}:** "
            f"{indicadores_particular['CR_k']:.4f} "
            f"→ "
            f"{indicadores_particular['CR_k'] * 100:.2f}% "
            f"del mercado está concentrado en las "
            f"{int(k)} empresas principales."
        )

        st.write(
            f"**IHH:** "
            f"{indicadores_particular['IHH']:.4f}"
        )

        st.write(
            f"**Índice de Dominancia:** "
            f"{indicadores_particular['Indice_Dominancia']:.4f}"
        )

        st.write(
            f"**Índice de Entropía:** "
            f"{indicadores_particular['Indice_Entropia']:.4f}"
        )


    # ========================================================
    # MONTE CARLO
    # ========================================================

    st.divider()

    st.subheader(
        "3️⃣ Simulación Monte Carlo"
    )

    st.write(
        "La simulación genera mercados aleatorios "
        "utilizando una distribución de Dirichlet. "
        "Cada mercado cumple Σ si = 1."
    )


    if M > 50000:

        st.warning(
            "⚠️ Está utilizando más de 50.000 iteraciones. "
            "Esto puede aumentar significativamente "
            "el tiempo de procesamiento."
        )


    # --------------------------------------------------------
    # FIJAR SEED
    # --------------------------------------------------------

    np.random.seed(
        int(seed)
    )


    # --------------------------------------------------------
    # GENERAR MERCADOS
    # --------------------------------------------------------

    with st.spinner(
        "🔄 Ejecutando simulación Monte Carlo..."
    ):

        resultados = np.random.dirichlet(
            np.ones(int(N)),
            size=int(M)
        )


        # Garantizar cierre exacto
        resultados[:, -1] = (
            1.0 -
            np.sum(
                resultados[:, :-1],
                axis=1
            )
        )


        # Cálculo vectorizado
        indicadores_mc = (
            calcular_indicadores_vectorizados(
                resultados,
                int(k)
            )
        )


    st.success(
        f"✓ Simulación completada: "
        f"{int(M):,} mercados generados."
    )


    # ========================================================
    # VALIDACIÓN DEL CIERRE
    # ========================================================

    sumas = np.sum(
        resultados,
        axis=1
    )


    cierre_correcto = np.allclose(
        sumas,
        1.0
    )


    if cierre_correcto:

        st.success(
            "✓ Validación matemática correcta: "
            "todas las simulaciones cumplen Σ si = 1."
        )

    else:

        st.error(
            "❌ Se detectaron errores en el cierre "
            "de las cuotas."
        )


    # ========================================================
    # SELECCIÓN DEL INDICADOR PARA GRÁFICO
    # ========================================================

    st.divider()

    st.subheader(
        "4️⃣ Distribución Monte Carlo"
    )


    indicador_grafico = st.selectbox(
        "Seleccione el indicador que desea visualizar:",
        [
            "CRk",
            "IHH",
            "Índice de Dominancia",
            "Índice de Entropía"
        ]
    )


    if indicador_grafico == "CRk":

        valores_mc = indicadores_mc["CRk"]

        valor_particular = (
            indicadores_particular["CR_k"]
        )


    elif indicador_grafico == "IHH":

        valores_mc = indicadores_mc["IHH"]

        valor_particular = (
            indicadores_particular["IHH"]
        )


    elif indicador_grafico == "Índice de Dominancia":

        valores_mc = indicadores_mc["Dominancia"]

        valor_particular = (
            indicadores_particular[
                "Indice_Dominancia"
            ]
        )


    else:

        valores_mc = indicadores_mc["Entropia"]

        valor_particular = (
            indicadores_particular[
                "Indice_Entropia"
            ]
        )


    # ========================================================
    # ESTADÍSTICAS
    # ========================================================

    promedio_mc = np.mean(
        valores_mc
    )

    mediana_mc = np.median(
        valores_mc
    )

    minimo_mc = np.min(
        valores_mc
    )

    maximo_mc = np.max(
        valores_mc
    )

    percentil = (
        np.mean(
            valores_mc <= valor_particular
        ) * 100
    )


    # ========================================================
    # GRÁFICO
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )


    rango = (
        maximo_mc -
        minimo_mc
    )


    if rango < 1e-12:

        ax.axvline(
            valores_mc[0],
            color="#2f80ed",
            linewidth=3,
            label="Monte Carlo"
        )

        ax.set_xlim(
            valores_mc[0] - 0.01,
            valores_mc[0] + 0.01
        )

    else:

        ax.hist(
            valores_mc,
            bins=40,
            density=True,
            color="#2f80ed",
            edgecolor="white",
            alpha=0.80,
            label="Distribución Monte Carlo"
        )


    # Caso particular
    ax.axvline(
        valor_particular,
        color="#087f5b",
        linestyle="--",
        linewidth=3,
        label=f"Caso particular = {valor_particular:.4f}"
    )


    # Promedio
    ax.axvline(
        promedio_mc,
        color="#f2994a",
        linestyle=":",
        linewidth=3,
        label=f"Promedio = {promedio_mc:.4f}"
    )


    ax.set_xlabel(
        indicador_grafico,
        fontsize=11
    )

    ax.set_ylabel(
        "Densidad",
        fontsize=11
    )

    ax.set_title(
        f"Distribución empírica del {indicador_grafico}",
        fontsize=14,
        fontweight="bold"
    )

    ax.legend()

    ax.grid(
        alpha=0.20
    )


    st.pyplot(
        fig,
        use_container_width=True
    )


    # ========================================================
    # POSICIÓN DEL CASO PARTICULAR
    # ========================================================

    st.subheader(
        "5️⃣ Posición del caso particular"
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Caso particular",
            f"{valor_particular:.4f}"
        )


    with col2:

        st.metric(
            "Promedio Monte Carlo",
            f"{promedio_mc:.4f}"
        )


    with col3:

        st.metric(
            "Mediana",
            f"{mediana_mc:.4f}"
        )


    with col4:

        st.metric(
            "Percentil",
            f"{percentil:.2f}%"
        )


    # ========================================================
    # EVALUACIÓN AUTOMÁTICA
    # ========================================================

    st.divider()

    st.subheader(
        "6️⃣ Evaluación automática"
    )


    if indicador_grafico == "IHH":

        nivel, explicacion = interpretar_ihh(
            valor_particular
        )


    elif indicador_grafico == "CRk":

        nivel, explicacion = interpretar_crk(
            valor_particular
        )


    elif indicador_grafico == "Índice de Dominancia":

        nivel, explicacion = interpretar_dominancia(
            valor_particular
        )


    else:

        nivel, explicacion = interpretar_entropia(
            valor_particular,
            int(N)
        )


    if "Baja" in nivel:

        st.success(
            f"{nivel}\n\n{explicacion}"
        )

    elif "moderada" in nivel.lower():

        st.warning(
            f"{nivel}\n\n{explicacion}"
        )

    elif "Alta" in nivel:

        st.error(
            f"{nivel}\n\n{explicacion}"
        )

    else:

        st.info(
            f"{nivel}\n\n{explicacion}"
        )


    # ========================================================
    # EXPLICACIÓN CUANTITATIVA
    # ========================================================

    st.markdown(
        "### 📐 Justificación cuantitativa"
    )


    diferencia = (
        valor_particular -
        promedio_mc
    )


    diferencia_porcentual = (
        abs(diferencia) /
        abs(promedio_mc)
        * 100
        if promedio_mc != 0
        else 0
    )


    if diferencia > 0:

        st.write(
            f"El valor del caso particular es "
            f"**{diferencia:.4f}** superior al promedio "
            f"de Monte Carlo. "
            f"En términos relativos, la diferencia es de "
            f"aproximadamente **{diferencia_porcentual:.2f}%**."
        )

    elif diferencia < 0:

        st.write(
            f"El valor del caso particular es "
            f"**{abs(diferencia):.4f}** inferior al promedio "
            f"de Monte Carlo. "
            f"En términos relativos, la diferencia es de "
            f"aproximadamente **{diferencia_porcentual:.2f}%**."
        )

    else:

        st.write(
            "El caso particular coincide aproximadamente "
            "con el promedio de los mercados simulados."
        )


    st.write(
        f"El caso particular se encuentra en el "
        f"**percentil {percentil:.2f}** de la distribución "
        f"Monte Carlo."
    )


    if percentil >= 90:

        st.warning(
            "📈 El caso particular se encuentra entre los "
            "mercados con mayores valores del indicador "
            "dentro de la simulación."
        )

    elif percentil <= 10:

        st.info(
            "📉 El caso particular se encuentra entre los "
            "mercados con menores valores del indicador "
            "dentro de la simulación."
        )

    else:

        st.info(
            "➡️ El caso particular se encuentra dentro de "
            "la zona central de la distribución simulada."
        )


    # ========================================================
    # OPINIÓN DEL USUARIO
    # ========================================================

    st.divider()

    st.subheader(
        "7️⃣ Evaluación del usuario"
    )


    st.write(
        "Antes de conocer la clasificación automática, "
        "¿cómo clasificarías el nivel de concentración "
        "del mercado?"
    )


    opinion_usuario = st.radio(
        "Tu respuesta:",
        [
            "🟢 Baja concentración",
            "🟡 Concentración moderada",
            "🔴 Alta concentración"
        ],
        horizontal=True
    )


    # Convertir resultado técnico a categoría
    if "Baja" in nivel:

        resultado_tecnico = (
            "🟢 Baja concentración"
        )

    elif "moderada" in nivel.lower():

        resultado_tecnico = (
            "🟡 Concentración moderada"
        )

    else:

        resultado_tecnico = (
            "🔴 Alta concentración"
        )


    if st.button(
        "🔍 Comparar mi evaluación",
        use_container_width=True
    ):

        if opinion_usuario == resultado_tecnico:

            st.success(
                "✅ Tu evaluación coincide con el "
                "resultado técnico del indicador."
            )

        else:

            st.warning(
                "ℹ️ Tu evaluación no coincide con la "
                "clasificación técnica utilizada por "
                "la aplicación."
            )

        st.write(
            f"**Resultado técnico:** "
            f"{resultado_tecnico}"
        )

        st.write(
            f"**Tu respuesta:** "
            f"{opinion_usuario}"
        )


    # ========================================================
    # TABLA DE SIMULACIONES
    # ========================================================

    st.divider()

    st.subheader(
        "8️⃣ Resultados de las simulaciones"
    )


    st.write(
        "Se muestran las primeras 10 simulaciones "
        "generadas."
    )


    cantidad_mostrar = min(
        10,
        int(M)
    )


    cuotas_tabla = pd.DataFrame(
        resultados[
            :cantidad_mostrar
        ],
        columns=[
            f"Empresa {i + 1}"
            for i in range(int(N))
        ]
    )


    indicadores_tabla = pd.DataFrame(
        {
            f"CR{int(k)}": indicadores_mc["CRk"][
                :cantidad_mostrar
            ],

            "IHH": indicadores_mc["IHH"][
                :cantidad_mostrar
            ],

            "Dominancia": indicadores_mc["Dominancia"][
                :cantidad_mostrar
            ],

            "Entropía": indicadores_mc["Entropia"][
                :cantidad_mostrar
            ]
        }
    )


    tabla_final = pd.concat(
        [
            cuotas_tabla,
            indicadores_tabla
        ],
        axis=1
    )


    st.dataframe(
        tabla_final,
        use_container_width=True
    )


    # ========================================================
    # DESCARGAR EXCEL
    # ========================================================

    st.subheader(
        "9️⃣ Descargar resultados"
    )


    archivo_excel = generar_excel(
        resultados,
        indicadores_mc,
        cuotas_particulares,
        indicadores_particular
    )


    st.download_button(
        label="📥 Descargar resultados en Excel",
        data=archivo_excel,
        file_name="resultados_monte_carlo.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        use_container_width=True
    )


    # ========================================================
    # VALIDACIÓN TÉCNICA
    # ========================================================

    st.divider()

    with st.expander(
        "🔎 Ver validación técnica completa"
    ):

        st.write(
            "### Validación del número de empresas"
        )

        st.write(
            f"N = {int(N)}"
        )

        if 2 <= N <= 100:

            st.success(
                "✓ El número de empresas está dentro "
                "del rango permitido: 2 ≤ N ≤ 100."
            )

        else:

            st.error(
                "❌ N está fuera del rango permitido."
            )


        st.write(
            "### Validación de las cuotas"
        )

        suma_minima = np.min(
            sumas
        )

        suma_maxima = np.max(
            sumas
        )


        st.write(
            f"Mínimo de Σsi: "
            f"{suma_minima:.15f}"
        )

        st.write(
            f"Máximo de Σsi: "
            f"{suma_maxima:.15f}"
        )


        if cierre_correcto:

            st.success(
                "✓ Todas las simulaciones cumplen "
                "Σsi = 1."
            )

        else:

            st.error(
                "❌ No todas las simulaciones cumplen "
                "Σsi = 1."
            )


        st.write(
            "### Reproducibilidad"
        )

        st.write(
            f"Semilla utilizada: **{int(seed)}**"
        )

        st.write(
            "Utilizar la misma semilla, N e iteraciones "
            "permite reproducir la misma simulación."
        )


        st.write(
            "### Eficiencia"
        )

        st.write(
            "Los indicadores de las simulaciones se calculan "
            "mediante operaciones vectorizadas de NumPy, "
            "reduciendo el uso de ciclos Python y mejorando "
            "el rendimiento para grandes cantidades de iteraciones."
        )


# ============================================================
# METODOLOGÍA
# ============================================================

st.divider()

st.subheader(
    "📚 Metodología y fundamentos"
)


with st.expander(
    "Ver metodología completa"
):

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "CRk",
            "IHH",
            "Dominancia",
            "Entropía",
            "Monte Carlo",
            "Interpretación"
        ]
    )


    # --------------------------------------------------------
    # CRk
    # --------------------------------------------------------

    with tab1:

        st.markdown(
            "### Índice de concentración CRk"
        )

        st.latex(
            r"CR_k = \sum_{i=1}^{k}s_i"
        )

        st.write(
            "El CRk mide la participación conjunta de las "
            "k empresas con mayor participación de mercado."
        )

        st.write(
            "Un valor elevado significa que una mayor "
            "proporción del mercado se encuentra concentrada "
            "en las principales empresas."
        )


    # --------------------------------------------------------
    # IHH
    # --------------------------------------------------------

    with tab2:

        st.markdown(
            "### Índice de Herfindahl-Hirschman (IHH)"
        )

        st.latex(
            r"IHH = \sum_{i=1}^{N}s_i^2"
        )

        st.write(
            "El IHH asigna mayor peso a las empresas con "
            "participaciones grandes debido a que las cuotas "
            "son elevadas al cuadrado."
        )

        st.write(
            "En esta aplicación se utilizan participaciones "
            "expresadas como proporciones entre 0 y 1."
        )


    # --------------------------------------------------------
    # DOMINANCIA
    # --------------------------------------------------------

    with tab3:

        st.markdown(
            "### Índice de Dominancia"
        )

        st.latex(
            r"ID = \sum_{i=1}^{N}"
            r"\left(\frac{s_i^2}{IHH}\right)^2"
        )

        st.write(
            "Este indicador evalúa cuánto pesa la concentración "
            "de cada empresa respecto de la concentración total."
        )

        st.write(
            "Un valor elevado indica que la concentración está "
            "más asociada a unas pocas empresas."
        )


    # --------------------------------------------------------
    # ENTROPÍA
    # --------------------------------------------------------

    with tab4:

        st.markdown(
            "### Índice de Entropía"
        )

        st.latex(
            r"IE = -\sum_{i=1}^{N}s_i\ln(s_i)"
        )

        st.write(
            "La entropía mide la dispersión de las "
            "participaciones de mercado."
        )

        st.write(
            "Valores mayores indican una distribución más "
            "equilibrada de las cuotas y, por lo tanto, "
            "menor concentración."
        )

        st.write(
            "El máximo teórico para N empresas es:"
        )

        st.latex(
            r"IE_{max} = \ln(N)"
        )


    # --------------------------------------------------------
    # MONTE CARLO
    # --------------------------------------------------------

    with tab5:

        st.markdown(
            "### Simulación Monte Carlo"
        )

        st.write(
            "La aplicación genera vectores aleatorios de "
            "participaciones mediante una distribución "
            "de Dirichlet."
        )

        st.latex(
            r"\sum_{i=1}^{N}s_i = 1"
        )

        st.write(
            "Cada iteración representa un posible mercado "
            "con N empresas."
        )

        st.write(
            "Para cada mercado generado se calculan los "
            "cuatro indicadores de concentración."
        )

        st.write(
            "Posteriormente, los resultados se utilizan para "
            "construir una distribución empírica y comparar "
            "el caso particular con ella."
        )


    # --------------------------------------------------------
    # INTERPRETACIÓN
    # --------------------------------------------------------

    with tab6:

        st.markdown(
            "### Interpretación"
        )

        st.write(
            "La aplicación combina dos enfoques:"
        )

        st.write(
            "1. **Evaluación normativa:** "
            "clasifica el nivel de concentración utilizando "
            "criterios de referencia."
        )

        st.write(
            "2. **Evaluación relativa:** "
            "compara el caso particular con los mercados "
            "generados mediante Monte Carlo."
        )

        st.write(
            "El percentil indica qué posición ocupa el caso "
            "particular dentro de la distribución simulada."
        )

        st.info(
            "⚠️ Los umbrales utilizados para CRk, Dominancia "
            "y Entropía deben entenderse como criterios de "
            "referencia del modelo. La interpretación económica "
            "final depende del mercado y de la metodología "
            "adoptada."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "📊 Simulador académico de concentración de mercado | "
    "CRk · IHH · Dominancia · Entropía · Monte Carlo"
)