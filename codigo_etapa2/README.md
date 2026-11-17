# Etapa 2 — Análisis de sentimiento supervisado (Opiniones de Steam) · versión corregida

Clasificador positive / negative con aprendizaje automático tradicional (scikit-learn).
No se usan LLMs ni modelos generativos. Esta versión aplica las correcciones del profesor:
pasos de normalización individuales y combinados, experimentos ordenados (línea base → una
técnica a la vez → combinación de lo que aporta), impacto de cada técnica con números
(accuracy y F1), y técnicas nuevas: corrección ortográfica, normalización de repeticiones,
SentiWordNet, emojis/emoticones, SVD y cuatro métodos de balanceo.

## Requisitos

Python 3.10 o posterior. Desde la carpeta principal del proyecto:

```bash
python -m pip install -r "ETAPA 2/codigo_etapa2/requirements.txt"
```

NLTK descarga sus recursos solo la primera vez (necesita internet).

## Cómo ejecutar (desde `ETAPA 2/codigo_etapa2`)

```bash
python experimentos.py              # 132 experimentos + prueba final en test (~1.5 min)
python experimentos.py --sin-test   # solo desarrollo (no toca test)
python interfaz.py                  # menú interactivo
python interfaz.py buscar "cheaters" --sentimiento negative
python interfaz.py clasificar "not worth the money"
python interfaz.py metricas
python interfaz.py errores
```

## Orden de los experimentos

| Etapa | Qué se prueba | Referencia |
|---|---|---|
| E1 | Línea base (solo tokenización) y cada paso solo: limpieza, contracciones, repeticiones, ortografía, lematización, stopwords | línea base |
| E2 | Todas las combinaciones de los pasos que aportaron solos (máx. 4) + combinaciones con ortografía y repeticiones + todos los pasos | mejor paso individual |
| E3 | Desbalance solo en train: class_weight, oversampling, undersampling, SMOTE | mejor normalización |
| E4 | Binaria / frecuencia / TF-IDF × unigramas / bigramas / ambos; control TF-IDF + class_weight; SVD 100 y 300 (solo LR y LinearSVC) | estado anterior |
| E5 | Mejoras de sentimiento una a la vez: negación marcada, negadores eliminados (control), Hu & Liu, SentiWordNet, emojis | línea base sin aumento lingüístico |
| E6 | Combinación de las dos mejores mejoras de E5 | línea base sin aumento lingüístico |
| E7 | GridSearchCV de alpha / C con PredefinedSplit (train para ajustar, dev para calificar) | estado anterior |

Un cambio se conserva solo si sube macro-F1 en dev; en empate se mantiene lo anterior.
Test se usa una sola vez: la mejor configuración se congela, se reentrena con train+dev y se evalúa.

## Archivos de resultados (`resultados/`)

| Archivo | Contenido |
|---|---|
| `labeled_documents.tsv` | document_id, review, text, rating, label, split + metadatos |
| `classification_predictions.tsv` / `classification_metrics.tsv` / `classification_reports.txt` | Dev (mejor config. de cada modelo) y test (`experiment_id = test`) |
| `experiments.tsv` / `.xlsx` | Bitácora completa con decisión de cada experimento |
| `evolucion_incremental.tsv` | Tabla cronológica: accuracy, macro-F1 y F1 negative tras cada etapa |
| `impacto_tecnicas.tsv` | Sin vs con cada técnica (macro-F1, accuracy, F1 negative) |
| `efecto_marginal_normalizacion.tsv` | Promedio con / sin cada paso en E1+E2 (idea del equipo) |
| `ablacion_mejoras.tsv` | Aporte de cada mejora de sentimiento |
| `cambios_ortografia.tsv`, `cambios_ortografia_v1_sin_proteccion.tsv`, `cambios_repeticiones.tsv`, `ejemplos_ortografia_repeticiones.tsv` | Original vs transformado |
| `ejemplos_negacion.tsv` | Conservar / marcar / eliminar negadores |
| `recursos_linguisticos.tsv`, `palabras_frecuentes_fuera_del_lexico.tsv` | Cobertura de Hu & Liu, SentiWordNet y emojis |
| `demostracion_balanceo_antes_de_particion.tsv` | Por qué no se balancea antes de partir (solo train+dev) |
| `verificacion_datos.tsv`, `distribucion_clases.tsv`, `revision_manual_etiquetas.tsv` | Calidad de datos y etiquetas |
| `errores_dev.tsv`, `analisis_errores_dev.tsv`, `efecto_mejoras_en_errores.tsv` | Errores y efecto de las mejoras en falsos positivos / negativos |
| `entorno.txt`, `figuras/` | Versiones, semillas y figuras |

`analisis_errores_dev.tsv`, `revision_manual_etiquetas.tsv` y `cambios_ortografia_v1_sin_proteccion.tsv`
son anotaciones o respaldos manuales: `experimentos.py` no los sobrescribe.

## Recursos (`recursos/`)

- `positive-words.txt`, `negative-words.txt`: Opinion Lexicon de Hu & Liu (2004), inglés.
- `sentiwordnet_compacto.tsv`: generado desde `SentiWordNet_3.0.0.txt` (Lexicons.zip del curso) con
  `construir_sentiwordnet_compacto()` de `preprocesamiento.py`.
- `emoji_lexicon.tsv`: columnas emoji, name, positive, negative de `Emojis lexicon.XLSX` del curso.
