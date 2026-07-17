"""
Construccion del cruce de Semifinales a partir del resultado real de cada
partido previo del bracket (match_id 97-100, ya jugados y cargados en
'data/matches.csv' y 'data/results.csv').

El nucleo del modelo predictivo (csv_sources, features, cascade_model,
predictor) no se toca aqui. Este modulo solo arma las 2 filas de partido
nuevas (con sus equipos, fecha y sede) para que esa misma cascada, ya
entrenada, las evalue -- Semifinales es la unica ronda que el modelo
efectivamente predice; los partidos previos se leen tal cual de
'df_modelo' (columna 'ganador_final'), sin pasar por la cascada. Si el
cruce de Semis ya venia incluido en 'results.csv' de antemano (caso de
Espana-Francia y Argentina-Inglaterra) no se duplica esa fila -- se
reutiliza la que ya trae el historico y solo se agrega la fila de
'matches_2026' (para asignarle stage_id=5 y que participe del mismo flujo
que el resto de partidos pendientes).

Los pares del bracket oficial (match_id 97 Francia-Marruecos, 98
Inglaterra-Noruega, 99 Belgica-Espana, 100 Suiza-Argentina) dan:
Semifinal 1 (97, 99) = Francia vs Espana en Arlington (AT&T Stadium);
Semifinal 2 (98, 100) = Inglaterra vs Argentina en Atlanta (Mercedes-Benz
Stadium). El ganador del partido de match_id MAS ALTO de cada pareja
juega de Local y el de match_id mas bajo de Visitante. Nota: al ser sede
neutral, el "truco del espejo" en core/predictor.py promedia ambas
orientaciones, asi que esta eleccion de Local/Visitante no sesga la
prediccion.
"""
import numpy as np
import pandas as pd

PARES_SEMIS = [(97, 99), (98, 100)]

FECHA_SEMIS = {
    (97, 99): "2026-07-14",
    (98, 100): "2026-07-15",
}

SEDE_SEMIS = {
    (97, 99): ("Arlington", "United States"),
    (98, 100): ("Atlanta", "United States"),
}


def construir_cruces_semis(df_modelo: pd.DataFrame) -> list:
    """Devuelve los 2 cruces de Semifinales [{home_team, away_team, date,
    city, country}] leyendo los ganadores reales de los partidos previos
    del bracket (match_id 97-100, ya jugados) directamente de 'df_modelo'."""
    ids_previos = [m for par in PARES_SEMIS for m in par]
    previos = df_modelo[df_modelo["match_id"].isin(ids_previos)]
    ganador_por_match = {
        int(fila["match_id"]): (fila["home_team"] if fila["ganador_final"] == "home" else fila["away_team"])
        for _, fila in previos.iterrows()
    }
    cruces = []
    for par in PARES_SEMIS:
        m_bajo, m_alto = par
        home = ganador_por_match[m_alto]
        away = ganador_por_match[m_bajo]
        fecha = FECHA_SEMIS[par]
        ciudad, pais = SEDE_SEMIS[par]
        cruces.append({"home_team": home, "away_team": away, "date": fecha, "city": ciudad, "country": pais})
    return cruces


def agregar_fixture_semis(historico: pd.DataFrame, matches_2026: pd.DataFrame,
                           teams: pd.DataFrame, cruces: list) -> tuple:
    """Agrega los 2 partidos de Semifinales como filas nuevas de 'historico'
    (al estilo results.csv, sin marcador) y de 'matches_2026' (al estilo
    matches.csv, stage_id=5, status Scheduled), para que
    feature_engineering.construir_dataset_modelo los procese exactamente
    igual que cualquier otro partido pendiente."""
    id_por_nombre = teams.set_index("team_name_hist")["team_id"]
    next_match_id = int(matches_2026["match_id"].max()) + 1

    filas_hist, filas_matches = [], []
    for cruce in cruces:
        fecha = pd.Timestamp(cruce["date"])

        ya_existe = (
            (historico["home_team"] == cruce["home_team"])
            & (historico["away_team"] == cruce["away_team"])
            & (historico["date"] >= "2026-07-08")
            & historico["home_score"].isna()
        ).any()
        if not ya_existe:
            filas_hist.append({
                "date": fecha, "home_team": cruce["home_team"], "away_team": cruce["away_team"],
                "home_score": np.nan, "away_score": np.nan, "tournament": "FIFA World Cup",
                "city": cruce["city"], "country": cruce["country"], "neutral": True,
            })
        filas_matches.append({
            "match_id": next_match_id, "date": fecha, "stage_id": 5,
            "home_team_id": id_por_nombre.get(cruce["home_team"]),
            "away_team_id": id_por_nombre.get(cruce["away_team"]),
            "home_score": np.nan, "away_score": np.nan, "status": "Scheduled",
            "home_xg": np.nan, "away_xg": np.nan,
            "home_team": cruce["home_team"], "away_team": cruce["away_team"],
        })
        next_match_id += 1

    historico_aug = pd.concat([historico, pd.DataFrame(filas_hist)], ignore_index=True)
    matches_aug = pd.concat([matches_2026, pd.DataFrame(filas_matches)], ignore_index=True)
    return historico_aug, matches_aug
