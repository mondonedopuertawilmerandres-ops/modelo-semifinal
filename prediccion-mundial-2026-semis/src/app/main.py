"""
Punto de entrada unico del proyecto. Orquesta 3 casos de uso encadenados,
todos resueltos por la MISMA cascada XGBoost entrenada una sola vez:

  1. train_cascade         -- carga datos, calcula variables, entrena la cascada.
  2. resolve_round_of_16    -- confirma quien llega a Cuartos (real + predicho).
  3. predict_semifinals     -- arma el cruce de Semifinales (a partir de los
                                4 clasificados reales de Cuartos) y lo predice.
                                Es la unica ronda que el modelo efectivamente
                                predice; Cuartos ya se jugo por completo y se
                                lee tal cual del dato real.

Uso (desde la raiz del proyecto):
    python -m src.app.main
"""
from src.app.use_cases import predict_semifinals, resolve_round_of_16, train_cascade


def main():
    print("=" * 70)
    print("PROYECTO PREDICTIVO MUNDIAL 2026 -- Semifinales -> Final")
    print("=" * 70)

    ctx = train_cascade.ejecutar()
    resolve_round_of_16.ejecutar(ctx)
    resultados_semis = predict_semifinals.ejecutar(ctx)

    print("\n" + "=" * 70)
    print("LOS 2 EQUIPOS QUE CLASIFICAN A LA FINAL (segun el modelo):")
    for _, r in resultados_semis.sort_values("match_id").iterrows():
        rival = r["equipo_visitante"] if r["ganador_predicho"] == r["equipo_local"] else r["equipo_local"]
        print(f"   -> {r['ganador_predicho']:<20} (vence a {rival}, {r['marcador_probable']}, "
              f"{r['confianza']*100:.0f}% conf.)")
    print("=" * 70)


if __name__ == "__main__":
    main()
