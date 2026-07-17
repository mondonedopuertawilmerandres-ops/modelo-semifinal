# Proyecto Predictivo Mundial 2026 — Semifinales

Modelo de Machine Learning (cascada de XGBoost) que predice qué selecciones
avanzan de **Semifinales** a la **Final** del Mundial 2026, entrenado con el
historial real de fútbol internacional (1872-2026).

La arquitectura combina dos regresores XGBoost (objetivo Tweedie) que
predicen goles esperados, apilados (*stacking*) con un clasificador
XGBoost 1X2 calibrado isotónicamente, más el "truco del espejo" (sede
neutral), ajuste de temperatura y una simulación de Monte Carlo con
penaltis para resolver cada cruce eliminatorio.

## Resultado

Los 4 Cuartos de Final (jugados del 9 al 11 de julio) ya son resultado real,
tomado tal cual de `data/results.csv`: clasificaron Francia, España,
Inglaterra y Argentina. El modelo no predice esa ronda — solo lee esos 4
clasificados para armar el cruce de Semifinales y predecirlo.

### Semifinales (pronóstico del modelo, se juegan el 14 y 15 de julio)

| Partido | Prob. Local | Prob. Empate | Prob. Visitante | Ganador predicho | Confianza (Monte Carlo) | Marcador probable |
|---|---|---|---|---|---|---|
| España 🆚 Francia | 40.8% | 22.8% | 36.3% | **España** | 53.4% | 1-0 |
| Argentina 🆚 Inglaterra | 30.1% | 22.4% | 47.5% | **Inglaterra** | 61.7% | 0-1 |

El resultado final es una imagen: **`output/clasificados_semifinales.png`**,
con los 2 finalistas predichos por el modelo (España e Inglaterra), cada uno
con su % de confianza. La tabla completa de probabilidades está en
`output/predicciones_semifinales.csv`.

## Estructura del proyecto

```
prediccion-mundial-2026-semis/
├── data/                              # CSV de entrada (equipos, partidos, historial, plantillas)
│   ├── teams.csv                          # 48 selecciones, Elo, confederacion, ranking FIFA
│   ├── matches.csv                        # calendario oficial Mundial 2026 (match_id, stage_id, venue, xG...)
│   ├── match_team_stats.csv               # estadisticas por equipo y partido (posesion, tiros, corners...)
│   ├── results.csv                        # historico 1872-2026, incluye el Mundial 2026 en curso
│   ├── shootouts.csv                      # ganador real de las tandas de penaltis
│   └── squads_and_players.csv             # plantillas y valor de mercado
├── src/
│   ├── core/
│   │   ├── config.py                      # constantes y parametros del modelo
│   │   ├── features.py                    # medias moviles, H2H, Elo, diff_*, etc.
│   │   ├── cascade_model.py               # entrenamiento de la cascada XGBoost + calibracion
│   │   ├── predictor.py                   # espejo + temperatura + Monte Carlo
│   │   └── bracket.py                     # arma el cruce de Semifinales a partir de los 4 clasificados reales de Cuartos
│   ├── app/
│   │   ├── main.py                        # orquesta los 3 casos de uso, de punta a punta
│   │   └── use_cases/
│   │       ├── train_cascade.py               # [1/3] carga datos + entrena la cascada
│   │       ├── resolve_round_of_16.py         # [2/3] confirma Octavos (ya 100% reales)
│   │       └── predict_semifinals.py          # [3/3] arma y predice el cruce de Semifinales (unica ronda que el modelo predice)
│   └── infra/
│       ├── csv_sources.py                 # carga y normalizacion de las fuentes crudas
│       ├── image_report.py                # genera clasificados_semifinales.png
│       └── paths.py                       # rutas de archivos del proyecto
├── output/
│   ├── clasificados_semifinales.png       # <- AQUI SE VEN LOS 2 FINALISTAS PREDICHOS
│   ├── predicciones_semifinales.csv       # tabla con las probabilidades de cada semifinal
│   └── models/modelo_cascada_mundial2026.pkl  # modelo entrenado (joblib)
├── requirements.txt
└── README.md
```

## Cómo se conectan los datos y la lógica

El pipeline avanza ronda por ronda, y en cada ronda un partido puede estar en
uno de dos estados: **ya jugado** (`home_score` con valor en `results.csv`) o
**pendiente** (`home_score` en blanco). La cascada solo predice lo pendiente;
todo lo demás se toma tal cual del dato real:

1. **`results.csv`** es la única fuente de verdad para el marcador
   (`home_score`/`away_score`) y por lo tanto de quién ganó cada partido,
   desde 1872 hasta hoy — incluido el Mundial 2026 en curso. Un partido
   pendiente aparece ahí con marcador en blanco.
2. **`matches.csv`** + **`match_team_stats.csv`** aportan `stage_id` y
   estadísticas ricas (xG, posesión, tiros, corners, paradas) que solo
   existen para el Mundial 2026. `csv_sources.preparar_historico()` las
   *pega* sobre las filas de `results.csv` cruzando por el par
   (`home_team`, `away_team`) dentro de la ventana del torneo — por eso el
   marcador de `matches.csv` casi no se usa: la verdad del resultado siempre
   viene de `results.csv`.
3. **`core/features.py`** convierte ese histórico unido en variables
   (`diff_*`, Elo, H2H, forma reciente) para las ~48 selecciones del Mundial.
4. **`core/cascade_model.py`** entrena la cascada con todos los partidos que
   ya tienen marcador (incluye ahora los 4 Cuartos reales).
5. **`resolve_round_of_16`** separa, dentro de su `stage_id`, los partidos
   con marcador (`ya_jugado=True`) de los pendientes, y solo predice estos
   últimos con `core/predictor.py` (paso informativo: Octavos ya se jugó
   por completo).
6. Como Cuartos también se jugó por completo, la única ronda pendiente es
   Semifinales: **`core/bracket.py`** lee los 4 ganadores reales de Cuartos
   directamente del dataset (`ganador_final`, sin pasar por la cascada) y
   arma ese cruce (quién juega contra quién, fecha, sede), agregándolo como
   partido nuevo (`stage_id=5`) para que la MISMA cascada ya entrenada lo
   evalúe. Si el cruce ya estaba anotado de antemano en `results.csv` (caso
   de España-Francia y Argentina-Inglaterra, agregados en cuanto los 4
   resultados de Cuartos fueron reales) no se duplica la fila.
7. **`src/infra/image_report.py`** convierte la tabla de salida
   (`predict_semifinals.py`) en `clasificados_semifinales.png`.

Este mismo mecanismo se reutilizará para pasar de Semifinales a la Final en
cuanto esos 2 partidos (14 y 15 de julio) tengan marcador real en
`results.csv`.

## Requisitos

- **Python 3.10 o superior** (probado con 3.13)
- Las librerías listadas en `requirements.txt`: `pandas`, `numpy`, `scipy`,
  `scikit-learn`, `xgboost`, `joblib`, `matplotlib`

## Instalación

```bash
# 1. Clonar el repositorio y entrar a la carpeta del proyecto
cd prediccion-mundial-2026-semis

# 2. (Recomendado) crear un entorno virtual
python -m venv venv
source venv/bin/activate      # en Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

## Cómo correrlo

Todo el pipeline (carga de datos → ingeniería de variables → entrenamiento
del modelo → confirmación de Octavos → predicción de Semifinales → imagen
final) se ejecuta con un solo comando desde la raíz del proyecto:

```bash
python -m src.app.main
```

Esto va a:
1. Cargar y unir los CSV de `data/` (equipos, partidos del Mundial 2026,
   estadísticas por partido, historial internacional desde 1872, plantillas).
2. Calcular todas las variables del modelo (forma reciente, head-to-head,
   Elo, tier, peso por confederación, valor de mercado, etc.).
3. Entrenar los 3 modelos de la cascada (2 regresores de goles + 1
   clasificador 1X2 calibrado) con ~7.600 partidos históricos entre las 48
   selecciones del Mundial 2026, y guardarlos en `output/models/`.
4. Confirmar los resultados reales de Octavos de Final (ya jugada por
   completo).
5. Armar el cruce de Semifinales a partir de los 4 clasificados reales de
   Cuartos de Final y predecirlo con la cascada — es la única ronda que el
   modelo efectivamente predice.
6. Generar `output/clasificados_semifinales.png`.

El entrenamiento completo tarda entre 1 y 3 minutos en una laptop normal (no
requiere GPU).

## ¿Dónde veo los equipos que pasan a la Final?

Abre la imagen generada en:

```
output/clasificados_semifinales.png
```

Ahí aparecen las 2 selecciones predichas, cada una marcada con ✓, junto al
rival al que vence, el marcador más probable según el modelo, y su % de
confianza calculado por simulación de Monte Carlo.

También puedes revisar el detalle numérico de cada partido (probabilidades
Local/Empate/Visitante, % de Monte Carlo, marcador probable) en:

```
output/predicciones_semifinales.csv
```

## Notas sobre los datos

- `data/teams.csv`, `data/matches.csv`, `data/match_team_stats.csv`: datos
  oficiales del Mundial 2026 (equipos, partidos, estadísticas por partido),
  actualizados hasta el **13 de julio de 2026** (incluye los 4 resultados
  reales de Cuartos de Final: Francia 2-0 Marruecos, España 2-1 Bélgica,
  Inglaterra 2-1 Noruega y Argentina 3-1 Suiza).
- `data/results.csv`, `data/shootouts.csv`: historial de fútbol
  internacional 1872-2026 (incluye el propio Mundial 2026 en curso), usado
  para entrenar el modelo con más de 7.600 partidos reales.
- `data/squads_and_players.csv`: valor de mercado de las plantillas.

No se necesita ninguna llave de API ni conexión a internet: todo corre en
local a partir de estos CSV.

*(Fuente: `output/predicciones_semifinales.csv`, corte de datos 2026-07-13.
"Prob. Local/Empate/Visitante" son las probabilidades 1X2 del clasificador
calibrado; "Confianza" es el % del ganador predicho tras la simulación de
Monte Carlo con penaltis.)*

**Finalistas predichos por el modelo: España e Inglaterra.**
