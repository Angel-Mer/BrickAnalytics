
import streamlit as st
import duckdb
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from scipy import stats

# CONFIGURACION DE LA PAGINA

st.set_page_config(
    page_title="Brick Analytics",
    page_icon="🧱",
    layout="wide"
)

# DISENO PERSONALIZADO

st.markdown("""
<style>
.stApp {
    background-color: #101010;
    color: white;
}

[data-testid="stSidebar"] {
    background-color: #191919;
}

[data-testid="stMetric"] {
    background-color: #222222;
    border: 1px solid #444444;
    border-radius: 12px;
    padding: 18px;
}

h1, h2, h3 {
    color: #FFD500 !important;
}

.stButton > button {
    background-color: #D71920;
    color: white;
    border-radius: 8px;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# CARGAR DATOS

@st.cache_data
def cargar_datos():
    con = duckdb.connect("lego.db", read_only=True)
    datos = con.execute("SELECT * FROM sets_lego").df()
    con.close()
    return datos

df = cargar_datos()

df["retirado"] = (
    df["estado"] == "Retirado"
).astype(int)

# MODELO ESTADISTICO

variables = [
    "precio_original",
    "antiguedad",
    "retirado"
]

X = sm.add_constant(df[variables])
Y = df["precio_reventa"]

modelo = sm.OLS(Y, X).fit()

df["precio_predicho"] = modelo.predict(X)
df["residuos"] = Y - df["precio_predicho"]

rmse = np.sqrt(np.mean(df["residuos"] ** 2))

# MENU LATERAL

st.sidebar.title("🧱 BRICK ANALYTICS")
st.sidebar.caption("LEGO Investment Dashboard")

pagina = st.sidebar.radio(
    "Menu de navegacion",
    [
        "Inicio",
        "Catalogo LEGO",
        "Analisis estadistico",
        "Pruebas de hipotesis",
        "Predictor de precios",
        "Calculadora de inversion",
        "Acerca del proyecto"
    ]
)

st.sidebar.divider()
st.sidebar.write("Proyecto Integrador - Unidad 4")
st.sidebar.caption("Angel Leonardo Mercado Miranda")
st.sidebar.caption("23680421")

# PAGINA DE INICIO

if pagina == "Inicio":

    st.title("🧱 BRICK ANALYTICS")
    st.subheader("LEGO INVESTMENT DASHBOARD")

    st.info("Proyecto academico basado en datos simulados.")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Observaciones",
        len(df)
    )

    c2.metric(
        "Sets diferentes",
        df["set_id"].nunique()
    )

    c3.metric(
        "Rentabilidad media",
        f'{df["rentabilidad"].mean():.2f}%'
    )

    c4.metric(
        "R² del modelo",
        f"{modelo.rsquared * 100:.2f}%"
    )

    st.divider()

    izquierda, derecha = st.columns(2)

    with izquierda:

        ranking = df.groupby(
            "tematica", as_index=False
        )["rentabilidad"].mean()

        fig = px.bar(
            ranking.sort_values("rentabilidad"),
            x="rentabilidad",
            y="tematica",
            orientation="h",
            title="Rentabilidad por tematica",
            color="rentabilidad",
            color_continuous_scale=[
                "#D71920", "#FFD500"
            ],
            template="plotly_dark"
        )

        st.plotly_chart(
            fig, use_container_width=True
        )

    with derecha:

        estados = df["estado"].value_counts()
        estados = estados.rename_axis("estado")
        estados = estados.reset_index(name="cantidad")

        fig = px.pie(
            estados,
            names="estado",
            values="cantidad",
            hole=0.5,
            title="Distribucion por estado simulado",
            color_discrete_sequence=[
                "#FFD500", "#D71920"
            ],
            template="plotly_dark"
        )

        st.plotly_chart(
            fig, use_container_width=True
        )

    st.subheader("Sets destacados")

    mejores = df.groupby(
        ["set_id", "nombre", "tematica"],
        as_index=False
    )["rentabilidad"].mean()

    st.dataframe(
        mejores.sort_values(
            "rentabilidad",
            ascending=False
        ).head(10),
        use_container_width=True,
        hide_index=True
    )

# CATALOGO

elif pagina == "Catalogo LEGO":

    st.title("📦 Catalogo LEGO")

    temas = st.multiselect(
        "Selecciona las tematicas",
        sorted(df["tematica"].unique()),
        default=["DC"]
    )

    estado = st.selectbox(
        "Estado simulado",
        ["Todos", "Retirado", "Disponible"]
    )

    buscar = st.text_input(
        "Buscar por nombre o numero de set"
    )

    filtrado = df.copy()

    if temas:
        filtrado = filtrado[
            filtrado["tematica"].isin(temas)
        ]

    if estado != "Todos":
        filtrado = filtrado[
            filtrado["estado"] == estado
        ]

    if buscar:
        filtrado = filtrado[
            filtrado["nombre"].str.contains(
                buscar, case=False, regex=False
            )
            |
            filtrado["set_id"].astype(str).str.contains(
                buscar, regex=False
            )
        ]

    st.metric("Observaciones encontradas", len(filtrado))

    st.dataframe(
        filtrado[[
            "set_id",
            "nombre",
            "tematica",
            "piezas",
            "estado",
            "precio_original",
            "precio_reventa",
            "rentabilidad"
        ]],
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Descargar catalogo filtrado",
        filtrado.to_csv(index=False).encode("utf-8"),
        "catalogo_lego.csv",
        "text/csv"
    )

# ANALISIS ESTADISTICO

elif pagina == "Analisis estadistico":

    st.title("📊 Analisis estadistico")

    columnas = [
        "piezas",
        "antiguedad",
        "precio_original",
        "retirado",
        "precio_reventa"
    ]

    correlacion = df[columnas].corr()

    fig = px.imshow(
        correlacion,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Matriz de correlacion de Pearson",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)

    c1.metric("R²", f"{modelo.rsquared:.4f}")

    c2.metric(
        "R² ajustado",
        f"{modelo.rsquared_adj:.4f}"
    )

    c3.metric("RMSE", f"${rmse:,.2f}")

    fig = px.scatter(
        df,
        x="precio_predicho",
        y="precio_reventa",
        color="tematica",
        hover_name="nombre",
        title="Precios observados vs predichos",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Resultados del modelo OLS")
    st.code(modelo.summary().as_text(), language="text")

    st.caption(
        "Metricas calculadas sobre los datos de entrenamiento."
    )

# PRUEBAS DE HIPOTESIS

elif pagina == "Pruebas de hipotesis":

    st.title("🔬 Pruebas de hipotesis")

    st.write(
        "H0: Los sets retirados no presentan "
        "una rentabilidad promedio superior."
    )

    st.write(
        "H1: Los sets retirados presentan "
        "una rentabilidad promedio superior."
    )

    retirados = df.loc[
        df["estado"] == "Retirado",
        "rentabilidad"
    ]

    disponibles = df.loc[
        df["estado"] == "Disponible",
        "rentabilidad"
    ]

    t_stat, p_value = stats.ttest_ind(
        retirados,
        disponibles,
        equal_var=False,
        alternative="greater"
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "Rentabilidad retirados",
        f"{retirados.mean():.2f}%"
    )

    c2.metric(
        "Rentabilidad disponibles",
        f"{disponibles.mean():.2f}%"
    )

    st.write(f"Estadistico t: {t_stat:.4f}")
    st.write(f"Valor p: {p_value:.6e}")

    alfa = st.selectbox(
        "Nivel de significancia",
        [0.01, 0.05, 0.10],
        index=1
    )

    for nombre, grupo in [
        ("Retirados", retirados),
        ("Disponibles", disponibles)
    ]:

        inferior, superior = stats.t.interval(
            0.95,
            df=len(grupo)-1,
            loc=grupo.mean(),
            scale=stats.sem(grupo)
        )

        st.write(
            f"IC 95% {nombre}: "
            f"[{inferior:.2f}%, {superior:.2f}%]"
        )

    if p_value < alfa:
        st.success("Se rechaza H0 en la simulacion.")
    else:
        st.warning("No se rechaza H0.")

    fig = px.box(
        df,
        x="estado",
        y="rentabilidad",
        color="estado",
        template="plotly_dark",
        title="Comparacion de rentabilidad"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Prueba ilustrativa con observaciones repetidas "
        "y una ventaja incorporada en la simulacion."
    )

# PREDICTOR DE PRECIOS

elif pagina == "Predictor de precios":

    st.title("🔮 Predictor de precios LEGO")

    precio = st.number_input(
        "Precio original (MXN)",
        min_value=100.0,
        value=5000.0,
        step=100.0
    )

    antiguedad = st.slider(
        "Antiguedad del set",
        0, 12, 5
    )

    estado = st.selectbox(
        "Estado del set",
        ["Disponible", "Retirado"]
    )

    if st.button("Predecir valor de reventa"):

        nuevo = pd.DataFrame({
            "const": [1.0],
            "precio_original": [precio],
            "antiguedad": [antiguedad],
            "retirado": [
                int(estado == "Retirado")
            ]
        })

        prediccion = modelo.predict(nuevo).iloc[0]

        ganancia = prediccion - precio

        rentabilidad = ganancia / precio * 100

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Reventa estimada",
            f"${prediccion:,.2f}"
        )

        c2.metric(
            "Ganancia estimada",
            f"${ganancia:,.2f}"
        )

        c3.metric(
            "Rentabilidad estimada",
            f"{rentabilidad:.2f}%"
        )

        if prediccion < 0:
            st.warning(
                "El modelo produjo un valor no realista."
            )

    st.info(
        "Estimaciones academicas. "
        "No representan cotizaciones reales."
    )

# CALCULADORA

elif pagina == "Calculadora de inversion":

    st.title("💰 Calculadora de inversion LEGO")

    compra = st.number_input(
        "Precio de compra",
        min_value=1.0,
        value=4500.0
    )

    venta = st.number_input(
        "Precio de venta",
        min_value=0.0,
        value=7200.0
    )

    gastos = st.number_input(
        "Gastos adicionales",
        min_value=0.0,
        value=0.0
    )

    inversion = compra + gastos
    ganancia = venta - inversion
    roi = ganancia / inversion * 100

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Inversion total",
        f"${inversion:,.2f}"
    )

    c2.metric(
        "Ganancia neta",
        f"${ganancia:,.2f}"
    )

    c3.metric(
        "Rentabilidad",
        f"{roi:.2f}%"
    )

# INFORMACION DEL PROYECTO

elif pagina == "Acerca del proyecto":

    st.title("🧱 Acerca de Brick Analytics")

    st.markdown("""
    ### Instituto Tecnologico de Cuautla

    **Alumno:** Angel Leonardo Mercado Miranda

    **Numero de control:** 23680421

    **Materia:** Fundamentos de Analisis de Datos

    **Unidad 4:** Analisis Estadistico Inferencial

    **Docente:** Aristides Caballero Alfaro

    ### Tecnologias utilizadas

    - Python
    - DuckDB
    - Pandas
    - NumPy
    - SciPy
    - Statsmodels
    - Plotly
    - Streamlit

    ### Objetivo

    Analizar la rentabilidad de los sets LEGO
    mediante tecnicas estadisticas inferenciales
    y modelos de regresion lineal multiple.

    Proyecto educativo no afiliado a LEGO.
    Los precios y estados son simulados.
    """)
