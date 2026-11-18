import duckdb
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Conexion a la base de datos
con = duckdb.connect("lego.db")

df = con.execute("""
    SELECT *
    FROM sets_lego
    ORDER BY registro_id
""").df()

# Convertir el estado a una variable numerica
df["retirado"] = (df["estado"] == "Retirado").astype(int)

# MATRIZ DE CORRELACION DE PEARSON

variables = [
    "piezas",
    "antiguedad",
    "precio_original",
    "retirado",
    "precio_reventa"
]

matriz_correlacion = df[variables].corr()

print("=" * 65)
print("BRICK ANALYTICS - FASE 2")
print("=" * 65)

print("\nMATRIZ DE CORRELACION DE PEARSON:")
print(matriz_correlacion.round(4))

# REGRESION LINEAL MULTIPLE

X = df[[
    "precio_original",
    "antiguedad",
    "retirado"
]]

Y = df["precio_reventa"]

X_const = sm.add_constant(X)

modelo = sm.OLS(Y, X_const).fit()

print("\nRESULTADOS DE LA REGRESION:")
print(modelo.summary())

# Predicciones y residuos
df["precio_predicho"] = modelo.predict(X_const)

df["residuos"] = (
    df["precio_reventa"] - df["precio_predicho"]
)

rmse = np.sqrt(np.mean(df["residuos"] ** 2))

print("\nMETRICAS DEL MODELO:")
print(f"R2: {modelo.rsquared:.4f}")
print(f"R2 ajustado: {modelo.rsquared_adj:.4f}")
print(f"RMSE: ${rmse:.2f} MXN")

df.to_csv("sets_lego_analizados.csv", index=False)

con.close()