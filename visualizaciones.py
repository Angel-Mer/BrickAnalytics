
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import os

df = pd.read_csv("sets_lego_analizados.csv")

os.makedirs("graficas", exist_ok=True)

sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(2, 3, figsize=(19, 11))

fig.suptitle(
    "BRICK ANALYTICS - LEGO INVESTMENT",
    fontsize=23,
    fontweight="bold",
    color="#C8102E"
)

# 1. Matriz de correlacion

variables = [
    "piezas",
    "antiguedad",
    "precio_original",
    "retirado",
    "precio_reventa"
]

sns.heatmap(
    df[variables].corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    ax=axes[0, 0]
)

axes[0, 0].set_title("1. Correlacion de Pearson")

# 2. Precio original vs precio de reventa

sns.regplot(
    data=df,
    x="precio_original",
    y="precio_reventa",
    color="#C8102E",
    ax=axes[0, 1]
)

axes[0, 1].set_title("2. Precio original vs Reventa")
axes[0, 1].set_xlabel("Precio original (MXN)")
axes[0, 1].set_ylabel("Precio reventa (MXN)")

# 3. Valores reales vs predichos

axes[0, 2].scatter(
    df["precio_predicho"],
    df["precio_reventa"],
    color="#DAA520",
    alpha=0.7
)

minimo = min(
    df["precio_reventa"].min(),
    df["precio_predicho"].min()
)

maximo = max(
    df["precio_reventa"].max(),
    df["precio_predicho"].max()
)

axes[0, 2].plot(
    [minimo, maximo],
    [minimo, maximo],
    "r--"
)

axes[0, 2].set_title("3. Observados vs Predichos")
axes[0, 2].set_xlabel("Precio predicho")
axes[0, 2].set_ylabel("Precio observado")

# 4. Rentabilidad por estado

sns.boxplot(
    data=df,
    x="estado",
    y="rentabilidad",
    color="#F2C300",
    ax=axes[1, 0]
)

axes[1, 0].set_title("4. Rentabilidad por estado")
axes[1, 0].set_ylabel("Rentabilidad (%)")

# 5. Rentabilidad por tematica

ranking = df.groupby(
    "tematica"
)["rentabilidad"].mean().sort_values()

colores = [
    "#202020" if tema == "DC" else "#C8102E"
    for tema in ranking.index
]

axes[1, 1].barh(
    ranking.index,
    ranking.values,
    color=colores
)

axes[1, 1].set_title("5. Rentabilidad por tematica")
axes[1, 1].set_xlabel("Rentabilidad promedio (%)")

# 6. Grafico Q-Q de residuos

sm.qqplot(
    df["residuos"],
    line="45",
    fit=True,
    ax=axes[1, 2]
)

axes[1, 2].set_title("6. Normalidad de residuos")

# Guardar panel estadistico

plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig(
    "graficas/panel_estadistico.png",
    dpi=300,
    bbox_inches="tight"
)

print("=" * 65)
print("BRICK ANALYTICS - FASE 4")
print("PANEL ESTADISTICO GENERADO CORRECTAMENTE")
print("Archivo: graficas/panel_estadistico.png")
print("=" * 65)

plt.show()
