"""
Caso de uso 3: con los 4 clasificados reales del bracket previo, arma el
cruce de Semifinales, lo predice con la misma cascada ya entrenada, guarda
la tabla de probabilidades y genera la imagen final de finalistas. Esta
es la unica ronda que el modelo efectivamente predice.
"""
import pandas as pd

from src.core import bracket
from src.core import features
from src.core import predictor
from src.infra import image_report
from src.infra.paths import OUTPUT_DIR


def ejecutar(ctx: dict) -> pd.DataFrame:
    print("\n[3/3] Armando el cruce de Semifinales y prediciendolo con el mismo modelo...")
    cruces = bracket.construir_cruces_semis(ctx["df_modelo"])
    historico_aug, matches_aug = bracket.agregar_fixture_semis(
        ctx["historico"], ctx["matches_2026"], ctx["teams"], cruces
    )
    df_modelo_semis = features.construir_dataset_modelo(
        historico_aug, matches_aug, ctx["teams"], ctx["valor_mercado"]
    )

    pendientes_mask = df_modelo_semis["home_score"].isna() & (df_modelo_semis["stage_id"] == 5)
    df_pendientes = df_modelo_semis[pendientes_mask].reset_index(drop=True)
    resultados = predictor.predecir_partidos_pendientes(df_pendientes, ctx["modelo"])

    for _, r in resultados.sort_values("match_id").iterrows():
        print(f"      {r['equipo_local']:<24} vs {r['equipo_visitante']:<24} -> "
              f"avanza {r['ganador_predicho']:<20} ({r['confianza']*100:.0f}% conf.) "
              f"marcador probable {r['marcador_probable']}")

    resultados.to_csv(OUTPUT_DIR / "predicciones_semifinales.csv", index=False)

    print("      Generando 'clasificados_semifinales.png'...")
    ruta_img = OUTPUT_DIR / "clasificados_semifinales.png"
    image_report.generar_imagen(
        resultados, ruta_img, fecha_corte="2026-07-13",
        titulo="Clasificados de Semifinales a la Final",
    )
    print(f"      Imagen guardada en {ruta_img}")

    return resultados
