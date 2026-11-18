
import duckdb
import pandas as pd

# ==============================================
# BRICK ANALYTICS - PROYECTO INTEGRADOR UNIDAD 4
# Alumno: Angel Leonardo Mercado Miranda
# Numero de control: 23680421
# Fase 1: Preparacion e ingesta de datos
# ==============================================

con = duckdb.connect("lego.db")

con.execute("SET threads = 1;")
con.execute("SELECT setseed(0.42);")

# Catalogo base de sets LEGO

con.execute("""
CREATE OR REPLACE TABLE catalogo AS
SELECT * FROM (VALUES
    (76252, 'Batcave Shadow Box', 'DC', 3981, 2023),
    (76139, '1989 Batmobile', 'DC', 3306, 2019),
    (76052, 'Classic TV Series Batcave', 'DC', 2526, 2016),
    (76161, '1989 Batwing', 'DC', 2363, 2020),
    (76023, 'The Tumbler', 'DC', 1869, 2014),
    (76240, 'Batmobile Tumbler', 'DC', 2049, 2021),
    (76271, 'Batman Animated Series Gotham City', 'DC', 4210, 2024),
    (76182, 'Batman Cowl', 'DC', 410, 2021),
    (75192, 'Millennium Falcon', 'Star Wars', 7541, 2017),
    (75252, 'Imperial Star Destroyer', 'Star Wars', 4784, 2019),
    (75313, 'AT-AT', 'Star Wars', 6785, 2021),
    (75290, 'Mos Eisley Cantina', 'Star Wars', 3187, 2020),
    (76269, 'Avengers Tower', 'Marvel', 5201, 2023),
    (76178, 'Daily Bugle', 'Marvel', 3772, 2021),
    (76218, 'Sanctum Sanctorum', 'Marvel', 2708, 2022),
    (21336, 'The Office', 'Ideas', 1164, 2022),
    (21330, 'Home Alone', 'Ideas', 3955, 2021),
    (21341, 'Hocus Pocus', 'Ideas', 2316, 2023),
    (10294, 'Titanic', 'Icons', 9090, 2021),
    (10307, 'Eiffel Tower', 'Icons', 10001, 2022),
    (42115, 'Lamborghini Sian', 'Technic', 3696, 2020),
    (42083, 'Bugatti Chiron', 'Technic', 3599, 2018),
    (71043, 'Hogwarts Castle', 'Harry Potter', 6020, 2018),
    (71741, 'Ninjago City Gardens', 'Ninjago', 5685, 2021),
    (75936, 'Jurassic Park T-Rex Rampage', 'Jurassic World', 3120, 2019)
) AS t(set_id, nombre, tematica, piezas, lanzamiento);
""")

# Generacion de 200 observaciones simuladas

con.execute("""
CREATE OR REPLACE TABLE sets_lego AS
WITH datos AS (
    SELECT
        c.*,
        2026 - lanzamiento AS antiguedad,
        observacion,
        random() AS r_precio,
        random() AS r_estado,
        random() AS r_mercado
    FROM catalogo c
    CROSS JOIN generate_series(1, 8) AS t(observacion)
),
base AS (
    SELECT *,
        ROUND(piezas * 2.1 + r_precio * 500, 2)
            AS precio_original,
        CASE
            WHEN r_estado < 0.60 THEN 'Retirado'
            ELSE 'Disponible'
        END AS estado
    FROM datos
),
precios AS (
    SELECT *,
        ROUND(
            precio_original *
            (0.85 + antiguedad * 0.045
            + CASE
                WHEN estado = 'Retirado' THEN 0.35
                ELSE 0.0
              END
            + r_mercado * 0.30),
            2
        ) AS precio_reventa
    FROM base
)
SELECT
    ROW_NUMBER() OVER (
        ORDER BY set_id, observacion
    ) AS registro_id,
    set_id,
    nombre,
    tematica,
    piezas,
    lanzamiento,
    antiguedad,
    estado,
    precio_original,
    precio_reventa,
    ROUND(
        (precio_reventa - precio_original)
        / precio_original * 100, 2
    ) AS rentabilidad
FROM precios;
""")

df = con.execute("SELECT * FROM sets_lego").df()

print("=" * 65)
print("BRICK ANALYTICS - LEGO INVESTMENT")
print("FASE 1: GENERACION DE DATOS")
print("=" * 65)

print("\nPRIMEROS 5 REGISTROS:")
print(df.head().to_string(index=False))

print("\nTOTAL DE REGISTROS:", len(df))

print("\nREGISTROS POR TEMATICA:")
print(df["tematica"].value_counts())

print("\nVALORES FALTANTES:")
print(df.isnull().sum())

df.to_csv("sets_lego.csv", index=False)

print("\nBase de datos y CSV generados correctamente.")

con.close()
