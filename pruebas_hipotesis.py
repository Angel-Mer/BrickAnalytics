
import duckdb
import numpy as np
from scipy import stats

con = duckdb.connect("lego.db")

df = con.execute("SELECT * FROM sets_lego").df()

retirados = df.loc[
    df["estado"] == "Retirado", "rentabilidad"
]

disponibles = df.loc[
    df["estado"] == "Disponible", "rentabilidad"
]

print("=" * 65)
print("BRICK ANALYTICS - FASE 3")
print("PRUEBAS DE HIPOTESIS")
print("=" * 65)

print("\nESTADISTICOS DESCRIPTIVOS")

print(f"Observaciones retiradas: {len(retirados)}")
print(f"Observaciones disponibles: {len(disponibles)}")

print(f"Rentabilidad media retirados: {retirados.mean():.2f}%")
print(f"Rentabilidad media disponibles: {disponibles.mean():.2f}%")

# Prueba t de Welch unilateral

t_stat, p_value = stats.ttest_ind(
    retirados,
    disponibles,
    equal_var=False,
    alternative="greater"
)

print("\nPRUEBA DE HIPOTESIS")

print(f"Estadistico t: {t_stat:.4f}")
print(f"Valor p: {p_value:.6e}")

# Intervalo de confianza del 95%

for nombre, datos in [
    ("Retirados", retirados),
    ("Disponibles", disponibles)
]:
    inferior, superior = stats.t.interval(
        0.95,
        df=len(datos)-1,
        loc=datos.mean(),
        scale=stats.sem(datos)
    )

    print(f"\nINTERVALO DE CONFIANZA: {nombre}")
    print(f"IC 95%: [{inferior:.2f}%, {superior:.2f}%]")

diferencia = retirados.mean() - disponibles.mean()

print("\nDIFERENCIA DE RENTABILIDAD")
print(f"{diferencia:.2f} puntos porcentuales")

print("\nDECISION ESTADISTICA")

if p_value < 0.05:
    print("Se rechaza la hipotesis nula H0.")
    print("Los resultados apoyan H1 en la simulacion.")
else:
    print("No se rechaza la hipotesis nula H0.")

print("\nNOTA: Resultados obtenidos con datos simulados.")

con.close()
