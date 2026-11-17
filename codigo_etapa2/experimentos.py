"""
experimentos.py — Etapa 2: clasificación supervisada de sentimiento (versión corregida)

Uso (desde esta carpeta):
    python experimentos.py

Orden de trabajo (cada etapa cambia UN factor y se queda con lo que mejora
macro-F1 en dev; si nada mejora, se conserva lo anterior):

  E1  Normalización individual: línea base (solo tokenización) y cada paso
      por separado: limpieza, contracciones, repeticiones, ortografía,
      lematización, stopwords.
  E2  Combinaciones de normalización: se combinan los pasos que mejoraron solos
      (todas sus combinaciones) y se prueban también las combinaciones con
      corrección ortográfica y normalización de repeticiones.
  E3  Desbalance de clases (solo en train): class_weight, oversampling,
      undersampling y SMOTE.
  E4  Representación: binaria / frecuencia / TF-IDF con unigramas, bigramas y
      ambos; SVD para LR y LinearSVC.
  E5  Mejoras de sentimiento, una a la vez contra la línea base sin aumento
      lingüístico: negación marcada, negadores eliminados (control), léxico de
      Hu & Liu, SentiWordNet y emojis/emoticones.
  E6  Combinación de las dos mejores mejoras de sentimiento.
  E7  Hiperparámetros con GridSearchCV (PredefinedSplit: entrena en train,
      califica en dev).

Después: mejor configuración por modelo, mejor global, reentrenamiento con
train+dev y evaluación ÚNICA en test.
"""

import json
import pickle
import platform
import sys
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, make_scorer, precision_recall_fscore_support,
                             recall_score)
from sklearn.model_selection import GridSearchCV, PredefinedSplit, train_test_split

from preprocesamiento import (CONFIG_BASE, LEXICO_NEG, LEXICO_POS, NEGADORES, PASOS, SEMILLA,
                              SWN, construir_pipeline, corregir_palabra, describir, limpiar,
                              normalizar, normalizar_repeticiones, pasos_a_texto,
                              puntajes_sentimiento, texto_a_pasos)

CARPETA = Path(__file__).resolve().parent
SALIDA = CARPETA / "resultados"
FIGURAS = SALIDA / "figuras"
MODELOS_DIR = CARPETA / "modelos"

CLASES = ["negative", "positive"]
MODELOS = ["MultinomialNB", "LogisticRegression", "LinearSVC"]
REJILLA = {
    "MultinomialNB": ("alpha", [0.01, 0.1, 0.5, 1.0, 2.0]),
    "LogisticRegression": ("C", [0.01, 0.1, 1.0, 10.0, 100.0]),
    "LinearSVC": ("C", [0.01, 0.1, 1.0, 10.0, 100.0]),
}
BALANCEOS = ["class_weight", "oversampling", "undersampling", "smote"]
MEJORAS = [("negacion", "marcar", "negación marcada (NEG_)"),
           ("negacion", "eliminar", "negadores eliminados (control)"),
           ("recurso", "huliu", "léxico Hu&Liu"),
           ("recurso", "sentiwordnet", "SentiWordNet"),
           ("recurso", "emojis", "emojis/emoticones")]
COLORES = {"MultinomialNB": "#2a78d6", "LogisticRegression": "#eb6834", "LinearSVC": "#1baf7a"}
ETAPAS = ["E1 individual", "E2 combinaciones", "E3 desbalance", "E4 representación",
          "E5 mejoras", "E6 dos mejores", "E7 hiperparámetros"]


# ---------------------------------------------------------------------------
# 1. Datos, etiquetas y particiones
# ---------------------------------------------------------------------------
def localizar_corpus():
    """Busca opinions_raw_corpus.tsv en la carpeta principal del proyecto (dos
    niveles arriba), un nivel arriba o en esta misma carpeta (idea del equipo)."""
    for d in [CARPETA.parents[1], CARPETA.parent, CARPETA]:
        if (d / "opinions_raw_corpus.tsv").is_file():
            return d / "opinions_raw_corpus.tsv"
    raise FileNotFoundError("No se encontró opinions_raw_corpus.tsv (debe estar en la carpeta "
                            "principal del proyecto, junto a recuperacion.py).")


def preparar_datos():
    ruta = localizar_corpus()
    df = pd.read_csv(ruta, sep="\t", dtype=str, keep_default_na=False)
    df["doc_id_etapa1"] = range(1, len(df) + 1)

    # Verificaciones de calidad pedidas en las correcciones
    revision = {
        "reseñas en el corpus": len(df),
        "rating vacío (faltantes)": int((df["rating"].str.strip() == "").sum()),
        "rating con valor fuera de {positive, negative}": int((~df["rating"].isin(["positive", "negative"])).sum()),
        "review vacía": int((df["review"].str.strip() == "").sum()),
        "review_id duplicado": int(df["review_id"].duplicated().sum()),
        "texto de reseña duplicado": int(df["review"].duplicated().sum()),
    }

    # Mapeo documentado: recomendación de Steam (voted_up) -> etiqueta
    mapeo = {"positive": "positive", "negative": "negative"}
    df["label"] = df["rating"].map(mapeo)
    if df["label"].isna().any():
        raise ValueError("Hay valores de rating fuera del mapeo documentado.")

    # Partición estratificada 70/15/15 ANTES de cualquier ajuste o balanceo
    indices = np.arange(len(df))
    idx_train, idx_resto = train_test_split(
        indices, test_size=0.30, shuffle=True, random_state=SEMILLA, stratify=df["label"])
    idx_dev, idx_test = train_test_split(
        idx_resto, test_size=0.50, shuffle=True, random_state=SEMILLA,
        stratify=df["label"].iloc[idx_resto])
    df["split"] = ""
    df.loc[idx_train, "split"] = "train"
    df.loc[idx_dev, "split"] = "dev"
    df.loc[idx_test, "split"] = "test"

    etiquetados = pd.DataFrame({
        "document_id": df["review_id"], "review": df["review"], "text": df["review"],
        "rating": df["rating"], "label": df["label"], "split": df["split"],
        "doc_id_etapa1": df["doc_id_etapa1"], "item_name": df["item_name"],
        "category": df["category"], "votes_up": df["votes_up"],
        "playtime_hours": df["playtime_hours"],
    })
    etiquetados.to_csv(SALIDA / "labeled_documents.tsv", sep="\t", index=False)
    pd.DataFrame(list(revision.items()), columns=["verificación", "resultado"]).to_csv(
        SALIDA / "verificacion_datos.tsv", sep="\t", index=False)

    dist = (etiquetados.groupby(["split", "label"]).size().unstack(fill_value=0)
            .reindex(["train", "dev", "test"]))
    dist.loc["total"] = dist.sum()
    dist["total"] = dist.sum(axis=1)
    dist["% negative"] = (100 * dist["negative"] / dist["total"]).round(1)
    dist.to_csv(SALIDA / "distribucion_clases.tsv", sep="\t")
    print("Distribución de clases:\n", dist)
    print("Verificación de datos:", revision)
    return etiquetados


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def calcular_metricas(y_real, y_pred):
    p, r, f, s = precision_recall_fscore_support(y_real, y_pred, labels=CLASES, zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y_real, y_pred, labels=CLASES,
                                                    average="macro", zero_division=0)
    m = {"accuracy": accuracy_score(y_real, y_pred),
         "macro_precision": mp, "macro_recall": mr, "macro_f1": mf}
    for i, c in enumerate(CLASES):
        m[f"precision_{c}"], m[f"recall_{c}"], m[f"f1_{c}"], m[f"support_{c}"] = p[i], r[i], f[i], int(s[i])
    m["confusion_matrix"] = json.dumps(confusion_matrix(y_real, y_pred, labels=CLASES).tolist())
    return m


# ---------------------------------------------------------------------------
# Registro de experimentos
# ---------------------------------------------------------------------------
class Registro:
    def __init__(self, X_train, y_train, X_dev, y_dev):
        self.X_train, self.y_train, self.X_dev, self.y_dev = X_train, y_train, X_dev, y_dev
        self.filas, self.preds, self.contador, self.cache = [], {}, 0, {}

    def _agregar(self, cfg, etapa, cambio, metricas, referencia=None, pred=None):
        self.contador += 1
        eid = f"C{self.contador:03d}"
        fila = {"experiment_id": eid, "etapa": etapa, "model": cfg["modelo"],
                "cambio_probado": cambio, **describir(cfg)}
        fila.update({k: (round(v, 4) if isinstance(v, float) else v) for k, v in metricas.items()})
        if referencia is not None:
            fila["referencia_id"] = referencia["experiment_id"]
            fila["dif_vs_referencia"] = round(metricas["macro_f1"] - referencia["macro_f1"], 4)
        fila["config"] = json.dumps(cfg)
        self.filas.append(fila)
        if pred is not None:
            self.preds[eid] = pred
        print(f"{eid} {cfg['modelo']:<18} {etapa:<18} {cambio[:44]:<44} "
              f"acc={metricas['accuracy']:.3f} macro-F1={metricas['macro_f1']:.4f}")
        return fila

    def correr(self, cfg, etapa, cambio, referencia=None):
        """Entrena con train, predice dev, califica y anota en la bitácora.
        Si la misma configuración ya se midió, reutiliza su resultado."""
        clave = json.dumps(cfg, sort_keys=True)
        if clave in self.cache:
            m, pred = self.cache[clave]
        else:
            pipe = construir_pipeline(cfg)
            pipe.fit(self.X_train, self.y_train)
            pred = pipe.predict(self.X_dev)
            m = calcular_metricas(self.y_dev, pred)
            self.cache[clave] = (m, pred)
        return self._agregar(cfg, etapa, cambio, m, referencia, pred)

    def rejilla(self, cfg, etapa, referencia):
        """E7: GridSearchCV con PredefinedSplit (ajusta en train, califica en dev)."""
        nombre, valores = REJILLA[cfg["modelo"]]
        X = pd.concat([self.X_train, self.X_dev])
        y = np.concatenate([self.y_train, self.y_dev])
        pliegue = PredefinedSplit([-1] * len(self.X_train) + [0] * len(self.X_dev))
        puntajes = {
            "macro_f1": "f1_macro", "accuracy": "accuracy",
            "macro_precision": "precision_macro", "macro_recall": "recall_macro",
            "f1_negative": make_scorer(f1_score, pos_label="negative", zero_division=0),
            "f1_positive": make_scorer(f1_score, pos_label="positive", zero_division=0),
            "recall_negative": make_scorer(recall_score, pos_label="negative", zero_division=0),
        }
        busqueda = GridSearchCV(construir_pipeline(cfg),
                                {f"clasificador__estimador__{nombre}": valores},
                                scoring=puntajes, cv=pliegue, refit=False)
        busqueda.fit(X, y)
        filas = []
        for i, valor in enumerate(valores):
            nuevo = dict(cfg, hiper={nombre: valor})
            met = {k: float(busqueda.cv_results_[f"mean_test_{k}"][i]) for k in puntajes}
            filas.append(self._agregar(nuevo, etapa, f"GridSearch {nombre}={valor}", met, referencia))
        return filas


def mejor(filas):
    """Mayor macro-F1; en empate gana la primera que se probó (la más simple)."""
    return max(filas, key=lambda f: (round(f["macro_f1"], 4), -int(f["experiment_id"][1:])))


def decidir(filas, referencia):
    """Se conserva un cambio solo si mejora macro-F1 frente a la referencia."""
    ganadora = mejor([referencia] + filas)
    for f in filas:
        f["decision"] = "se conserva" if f is ganadora else (
            "empate: se mantiene la referencia" if round(f["macro_f1"], 4) == round(referencia["macro_f1"], 4)
            else "se descarta")
    return ganadora


def explorar_modelo(reg, modelo):
    """Búsqueda por etapas para un modelo. Devuelve la ganadora y los puntos clave."""
    hitos = {}
    cfg0 = dict(CONFIG_BASE, modelo=modelo)

    # ----- E1: línea base y cada paso de normalización por separado -----
    base = reg.correr(cfg0, "E1 individual", "línea base: solo tokenización")
    base["decision"] = "línea base"
    hitos["base"] = base
    individuales = {}
    for paso in PASOS:
        f = reg.correr(dict(cfg0, pasos=paso), "E1 individual", f"+ {paso} (solo)", base)
        f["decision"] = "aporta" if f["macro_f1"] > base["macro_f1"] else "no aporta"
        individuales[paso] = f
    mejor_individual = mejor([base] + list(individuales.values()))

    # ----- E2: combinaciones de los pasos que aportaron solos -----
    utiles = sorted([p for p, f in individuales.items() if f["macro_f1"] > base["macro_f1"]],
                    key=lambda p: -individuales[p]["macro_f1"])[:4]
    conjuntos = []
    for k in range(2, len(utiles) + 1):
        conjuntos += [set(c) for c in combinations(utiles, k)]
    # Combinaciones pedidas con ortografía y repeticiones
    extra = [{"ortografia", "repeticiones"}, set(utiles) | {"ortografia"},
             set(utiles) | {"repeticiones"}, set(utiles) | {"ortografia", "repeticiones"},
             set(PASOS)]
    for c in extra:
        if len(c) >= 2 and c not in conjuntos:
            conjuntos.append(c)
    filas = [reg.correr(dict(cfg0, pasos=pasos_a_texto(c)), "E2 combinaciones",
                        "+ " + " + ".join(p for p in PASOS if p in c), mejor_individual)
             for c in conjuntos]
    actual = decidir(filas, mejor_individual)
    hitos["normalizacion"] = actual
    cfg = json.loads(actual["config"])

    # ----- E3: desbalance (solo afecta al entrenamiento) -----
    filas = [reg.correr(dict(cfg, balanceo=b), "E3 desbalance", b, actual) for b in BALANCEOS]
    actual = decidir(filas, actual)
    hitos["desbalance"] = actual
    cfg = json.loads(actual["config"])

    # ----- E4: representación vectorial, n-gramas y SVD -----
    filas = []
    for rep in ["binaria", "frecuencia", "tfidf"]:
        for ng in [(1, 1), (2, 2), (1, 2)]:
            if rep == cfg["representacion"] and list(ng) == list(cfg["ngramas"]):
                continue
            filas.append(reg.correr(dict(cfg, representacion=rep, ngramas=ng),
                                    "E4 representación", f"{rep} {ng}", actual))
    if not cfg["balanceo"]:
        # Control: TF-IDF con pesos de clase (sin balanceo, TF-IDF tiende a predecir solo 'positive')
        filas.append(reg.correr(dict(cfg, representacion="tfidf", ngramas=(1, 2), balanceo="class_weight"),
                                "E4 representación", "tfidf (1, 2) + class_weight (control)", actual))
    actual = decidir(filas, actual)
    cfg = json.loads(actual["config"])
    if modelo != "MultinomialNB":           # SVD da valores negativos: incompatible con NB
        filas = [reg.correr(dict(cfg, svd=n), "E4 representación", f"+ SVD {n} componentes", actual)
                 for n in (100, 300)]
        actual = decidir(filas, actual)
        cfg = json.loads(actual["config"])
    hitos["sin_mejoras"] = actual           # línea base SIN aumento lingüístico

    # ----- E5: mejoras de sentimiento, una a la vez -----
    sin_mejoras = actual
    filas = []
    for tipo, valor, nombre in MEJORAS:
        nuevo = dict(cfg, negacion=valor) if tipo == "negacion" else dict(cfg, recursos=[valor])
        f = reg.correr(nuevo, "E5 mejoras", f"+ {nombre}", sin_mejoras)
        f["decision"] = ("control (no se usa)" if valor == "eliminar"
                         else "aporta" if f["macro_f1"] > sin_mejoras["macro_f1"] else "no aporta")
        f["_mejora"] = (tipo, valor, nombre)
        filas.append(f)
    hitos["mejoras"] = filas

    # ----- E6: combinar las dos mejores mejoras -----
    reales = sorted([f for f in filas if f["_mejora"][1] != "eliminar"],
                    key=lambda f: (-round(f["macro_f1"], 4), int(f["experiment_id"][1:])))
    dos = reales[:2]
    nuevo = dict(cfg)
    for f in dos:
        tipo, valor, _ = f["_mejora"]
        if tipo == "negacion":
            nuevo["negacion"] = valor
        else:
            nuevo["recursos"] = sorted(set(nuevo.get("recursos", [])) | {valor})
    comb = reg.correr(nuevo, "E6 dos mejores",
                      "+ " + " + ".join(f["_mejora"][2] for f in dos), sin_mejoras)
    candidatas = [f for f in reales if f["macro_f1"] > sin_mejoras["macro_f1"]] + [comb]
    actual = mejor([sin_mejoras] + candidatas)
    comb["decision"] = "se conserva" if actual is comb else "se descarta"
    for f in reales:
        if f is actual:
            f["decision"] = "se conserva"
    hitos["con_mejoras"] = actual
    cfg = json.loads(actual["config"])

    # ----- E7: hiperparámetros -----
    filas = reg.rejilla(cfg, "E7 hiperparámetros", actual)
    actual = decidir(filas, actual)
    hitos["final"] = actual
    for f in hitos["mejoras"]:
        f.pop("_mejora", None)
    return actual, hitos


# ---------------------------------------------------------------------------
# Análisis de recursos y transformaciones (solo con train)
# ---------------------------------------------------------------------------
def analizar_recursos(train):
    filas = []
    for clase in CLASES + ["todas"]:
        t = train if clase == "todas" else train[train["label"] == clase]
        fila = {"clase": clase, "reseñas": len(t)}
        for r, nombre in [("huliu", "Hu&Liu"), ("sentiwordnet", "SentiWordNet"), ("emojis", "emojis")]:
            v = np.array([puntajes_sentimiento(x, r) for x in t["text"]])
            fila[f"% cubiertas {nombre}"] = round(100 * (v.sum(axis=1) > 0).mean(), 1)
            fila[f"prom. positivo {nombre}"] = round(v[:, 0].mean(), 3)
            fila[f"prom. negativo {nombre}"] = round(v[:, 1].mean(), 3)
        fila["% con negador"] = round(100 * t["text"].map(
            lambda x: any(n in normalizar(x, "limpieza+contracciones").split() for n in NEGADORES)).mean(), 1)
        filas.append(fila)
    pd.DataFrame(filas).to_csv(SALIDA / "recursos_linguisticos.tsv", sep="\t", index=False)

    vocab = pd.Series(" ".join(normalizar(t, "limpieza+contracciones+stopwords") for t in train["text"]).split()).value_counts()
    fuera = vocab[[p.isalpha() and p not in LEXICO_POS and p not in LEXICO_NEG for p in vocab.index]]
    fuera.head(40).rename("frecuencia").to_csv(SALIDA / "palabras_frecuentes_fuera_del_lexico.tsv", sep="\t")

    con_neg = train[train["text"].map(lambda x: "NEG_" in normalizar(x, "limpieza+contracciones", "marcar"))]
    ej = con_neg[con_neg["text"].str.len() < 160].head(10)
    pd.DataFrame({
        "document_id": ej["document_id"], "label": ej["label"],
        "original": ej["text"].map(lambda x: " ".join(str(x).split())),
        "conservar": ej["text"].map(lambda x: normalizar(x, "limpieza+contracciones")),
        "marcar": ej["text"].map(lambda x: normalizar(x, "limpieza+contracciones", "marcar")),
        "eliminar": ej["text"].map(lambda x: normalizar(x, "limpieza+contracciones", "eliminar")),
    }).to_csv(SALIDA / "ejemplos_negacion.tsv", sep="\t", index=False)

    contradicen = []
    for _, r in train.iterrows():
        p, n = puntajes_sentimiento(r["text"], "huliu")
        if (r["label"] == "positive" and n > p) or (r["label"] == "negative" and p > n):
            contradicen.append({"document_id": r["document_id"], "label": r["label"],
                                "lex_pos": p, "lex_neg": n, "text": r["text"]})
    pd.DataFrame(contradicen).to_csv(SALIDA / "revision_etiquetas_candidatas.tsv", sep="\t", index=False)


def analizar_transformaciones(train):
    """Compara reseñas originales vs transformadas por ortografía y repeticiones."""
    cambios_orto, cambios_rep, ejemplos = {}, {}, []
    for _, r in train.iterrows():
        tokens = normalizar(r["text"], "limpieza+contracciones").split()
        for t in tokens:
            c = corregir_palabra(t)
            if c != t:
                cambios_orto[(t, c)] = cambios_orto.get((t, c), 0) + 1
        rep = normalizar_repeticiones(tokens)
        for t in set(tokens) - set(rep):
            cambios_rep[t] = cambios_rep.get(t, 0) + 1
        a = normalizar(r["text"], "limpieza+contracciones")
        b = normalizar(r["text"], "limpieza+contracciones+repeticiones+ortografia")
        if a != b and len(a) < 220:
            ejemplos.append({"document_id": r["document_id"], "label": r["label"],
                             "original_normalizado": a, "con_repeticiones_y_ortografia": b})
    pd.DataFrame([{"original": o, "corregida": c, "veces": v} for (o, c), v in cambios_orto.items()]
                 ).sort_values("veces", ascending=False).to_csv(SALIDA / "cambios_ortografia.tsv", sep="\t", index=False)
    pd.DataFrame([{"token_original": t, "reseñas": v} for t, v in cambios_rep.items()]
                 ).sort_values("reseñas", ascending=False).to_csv(SALIDA / "cambios_repeticiones.tsv", sep="\t", index=False)
    pd.DataFrame(ejemplos).to_csv(SALIDA / "ejemplos_ortografia_repeticiones.tsv", sep="\t", index=False)
    print(f"Ortografía: {len(cambios_orto)} correcciones distintas en train; "
          f"repeticiones: {len(cambios_rep)} tokens distintos modificados")


def demostrar_fuga_remuestreo(train, dev):
    """Muestra por qué NO se balancea antes de partir los datos. Usa solo
    train+dev (test no se toca) y no participa en ninguna decisión."""
    from imblearn.over_sampling import RandomOverSampler
    cfg = dict(CONFIG_BASE, modelo="LogisticRegression", pasos="limpieza+contracciones")
    # Correcto: oversampling solo dentro de train
    pipe = construir_pipeline(dict(cfg, balanceo="oversampling")).fit(train["text"], train["label"])
    m_ok = calcular_metricas(dev["label"], pipe.predict(dev["text"]))
    # Incorrecto: oversampling de todo el conjunto y DESPUÉS partir
    pool = pd.concat([train, dev])
    Xr, yr = RandomOverSampler(random_state=SEMILLA).fit_resample(pool[["text"]], pool["label"])
    Xa, Xb, ya, yb = train_test_split(Xr["text"], yr, test_size=len(dev) / len(pool),
                                      shuffle=True, random_state=SEMILLA, stratify=yr)
    pipe = construir_pipeline(cfg).fit(Xa, ya)
    m_mal = calcular_metricas(yb, pipe.predict(Xb))
    repetidas = int(Xb.isin(set(Xa)).sum())
    pd.DataFrame([
        {"procedimiento": "correcto: partir y luego oversampling solo en train", "reseñas evaluadas": len(dev),
         "accuracy": round(m_ok["accuracy"], 4), "macro_f1": round(m_ok["macro_f1"], 4),
         "f1_negative": round(m_ok["f1_negative"], 4), "reseñas de evaluación repetidas en train": 0},
        {"procedimiento": "incorrecto: oversampling antes de partir", "reseñas evaluadas": len(Xb),
         "accuracy": round(m_mal["accuracy"], 4), "macro_f1": round(m_mal["macro_f1"], 4),
         "f1_negative": round(m_mal["f1_negative"], 4), "reseñas de evaluación repetidas en train": repetidas},
    ]).to_csv(SALIDA / "demostracion_balanceo_antes_de_particion.tsv", sep="\t", index=False)


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def _ejes_limpios(ax):
    ax.grid(axis="y", color="#e5e4e0", linewidth=0.8)
    for lado in ["top", "right"]:
        ax.spines[lado].set_visible(False)


def figura_matriz(cm, titulo, ruta):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(3.6, 3.2), dpi=200)
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > cm.max() / 2 else "#0b0b0b")
    ax.set_xticks([0, 1], ["negative", "positive"])
    ax.set_yticks([0, 1], ["negative", "positive"])
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(titulo, fontsize=9)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def figura_etapas(evolucion, ruta):
    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=200)
    for modelo in MODELOS:
        t = evolucion[evolucion["model"] == modelo]
        ax.plot(range(len(t)), t["macro_f1"], marker="o", linewidth=2, markersize=6,
                color=COLORES[modelo], label=modelo)
        ax.annotate(f"{t['macro_f1'].iloc[-1]:.3f}", (len(t) - 1, t["macro_f1"].iloc[-1]),
                    textcoords="offset points", xytext=(8, 0), va="center", fontsize=8, color="#52514e")
    etiquetas = evolucion[evolucion["model"] == MODELOS[0]]["paso"].tolist()
    ax.set_xticks(range(len(etiquetas)), [e.replace(" ", "\n", 1) for e in etiquetas], fontsize=7)
    ax.set_ylabel("macro-F1 en dev")
    _ejes_limpios(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_xlim(-0.3, len(etiquetas) - 0.4)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def figura_impacto(impacto, modelo, ruta):
    t = impacto[impacto["model"] == modelo].iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.8, 0.32 * len(t) + 1.0), dpi=200)
    colores = ["#1baf7a" if d > 0 else ("#c9c8c3" if d == 0 else "#d1495b") for d in t["delta_macro_f1"]]
    ax.barh(range(len(t)), t["delta_macro_f1"], color=colores, height=0.6)
    corto = {"normalización (sobre tokenización)": "Normalización", "desbalance": "Desbalance",
             "mejoras de sentimiento": "Sentimiento"}
    acentos = {"ortografia": "ortografía", "lematizacion": "lematización"}
    etiquetas = [f"{corto[g]} · {acentos.get(x, x)}" for g, x in zip(t["grupo"], t["tecnica"])]
    ax.set_yticks(range(len(t)), etiquetas, fontsize=7.5)
    ax.axvline(0, color="#52514e", linewidth=0.8)
    for i, d in enumerate(t["delta_macro_f1"]):
        ax.text(d + (0.002 if d >= 0 else -0.002), i, f"{d:+.3f}", va="center",
                ha="left" if d >= 0 else "right", fontsize=7, color="#3d3d3a")
    ax.set_xlabel("Cambio en macro-F1 en dev al agregar la técnica")
    ax.grid(axis="x", color="#e5e4e0", linewidth=0.8)
    for lado in ["top", "right"]:
        ax.spines[lado].set_visible(False)
    lim = max(abs(t["delta_macro_f1"]).max() * 1.35, 0.02)
    ax.set_xlim(-lim, lim)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------
def main():
    for d in (SALIDA, FIGURAS, MODELOS_DIR):
        d.mkdir(exist_ok=True)

    etiquetados = preparar_datos()
    train = etiquetados[etiquetados["split"] == "train"]
    dev = etiquetados[etiquetados["split"] == "dev"]
    test = etiquetados[etiquetados["split"] == "test"]
    analizar_recursos(train)
    analizar_transformaciones(train)
    demostrar_fuga_remuestreo(train, dev)

    reg = Registro(train["text"], train["label"].to_numpy(), dev["text"], dev["label"].to_numpy())
    ganadoras, hitos = {}, {}
    for modelo in MODELOS:
        ganadoras[modelo], hitos[modelo] = explorar_modelo(reg, modelo)

    final = max(ganadoras.values(),
                key=lambda f: (round(f["macro_f1"], 4), round(f["f1_negative"], 4)))
    tabla = pd.DataFrame(reg.filas)
    tabla["selected"] = "no"
    for f in ganadoras.values():
        tabla.loc[tabla["experiment_id"] == f["experiment_id"], "selected"] = "yes (mejor del modelo)"
    tabla.loc[tabla["experiment_id"] == final["experiment_id"], "selected"] = "yes (final)"
    columnas = ["experiment_id", "etapa", "model", "cambio_probado", "normalizacion", "representacion",
                "mejora_linguistica", "desbalance", "hiperparametros", "accuracy", "macro_precision",
                "macro_recall", "macro_f1", "f1_negative", "f1_positive", "recall_negative",
                "referencia_id", "dif_vs_referencia", "decision", "selected"]
    tabla[columnas + ["config"]].to_csv(SALIDA / "experiments.tsv", sep="\t", index=False)
    try:
        tabla[columnas].to_excel(SALIDA / "experiments.xlsx", index=False)
    except ImportError:
        print("Aviso: instala openpyxl para generar experiments.xlsx (el .tsv sí se generó).")

    # Tabla incremental (cronológica): estado aceptado al terminar cada etapa
    pasos_evol = [("línea base", "base"), ("E1 mejor paso individual", None), ("E2 combinación", "normalizacion"),
                  ("E3 desbalance", "desbalance"), ("E4 representación", "sin_mejoras"),
                  ("E5-E6 mejoras", "con_mejoras"), ("E7 hiperparámetros", "final")]
    evol = []
    for modelo in MODELOS:
        t = tabla[tabla["model"] == modelo]
        h = hitos[modelo]
        e1 = t[t["etapa"] == "E1 individual"]
        e1_mejor = e1.loc[e1["macro_f1"].idxmax()] if e1["macro_f1"].max() > h["base"]["macro_f1"] else None
        for nombre, clave in pasos_evol:
            f = (e1_mejor.to_dict() if e1_mejor is not None else h["base"]) if clave is None else h[clave]
            evol.append({"model": modelo, "paso": nombre, "experiment_id": f["experiment_id"],
                         "cambio_aceptado": f["cambio_probado"], **{k: describir(json.loads(f["config"]))[k]
                         for k in ["normalizacion", "representacion", "mejora_linguistica", "desbalance", "hiperparametros"]},
                         "accuracy": f["accuracy"], "macro_f1": f["macro_f1"], "f1_negative": f["f1_negative"]})
    evolucion = pd.DataFrame(evol)
    evolucion.to_csv(SALIDA / "evolucion_incremental.tsv", sep="\t", index=False)

    # Impacto de cada técnica (sin vs con) y efecto marginal (idea del equipo)
    impacto = []
    for modelo in MODELOS:
        t = tabla[tabla["model"] == modelo]
        h = hitos[modelo]
        for _, f in t[t["etapa"] == "E1 individual"].iloc[1:].iterrows():
            impacto.append({"model": modelo, "grupo": "normalización (sobre tokenización)",
                            "tecnica": f["cambio_probado"][2:].replace(" (solo)", ""),
                            "experiment_id": f["experiment_id"], "f1_sin": h["base"]["macro_f1"], "f1_con": f["macro_f1"],
                            "acc_sin": h["base"]["accuracy"], "acc_con": f["accuracy"],
                            "f1neg_sin": h["base"]["f1_negative"], "f1neg_con": f["f1_negative"]})
        ref = h["normalizacion"]
        for _, f in t[t["etapa"] == "E3 desbalance"].iterrows():
            impacto.append({"model": modelo, "grupo": "desbalance", "tecnica": f["cambio_probado"],
                            "experiment_id": f["experiment_id"], "f1_sin": ref["macro_f1"], "f1_con": f["macro_f1"],
                            "acc_sin": ref["accuracy"], "acc_con": f["accuracy"],
                            "f1neg_sin": ref["f1_negative"], "f1neg_con": f["f1_negative"]})
        ref = h["sin_mejoras"]
        for _, f in t[t["etapa"].isin(["E5 mejoras", "E6 dos mejores"])].iterrows():
            impacto.append({"model": modelo, "grupo": "mejoras de sentimiento",
                            "tecnica": f["cambio_probado"][2:] if f["cambio_probado"].startswith("+ ") else f["cambio_probado"],
                            "experiment_id": f["experiment_id"], "f1_sin": ref["macro_f1"], "f1_con": f["macro_f1"],
                            "acc_sin": ref["accuracy"], "acc_con": f["accuracy"],
                            "f1neg_sin": ref["f1_negative"], "f1neg_con": f["f1_negative"]})
    impacto = pd.DataFrame(impacto)
    impacto["delta_macro_f1"] = (impacto["f1_con"] - impacto["f1_sin"]).round(4)
    impacto["delta_f1_negative"] = (impacto["f1neg_con"] - impacto["f1neg_sin"]).round(4)
    impacto.to_csv(SALIDA / "impacto_tecnicas.tsv", sep="\t", index=False)

    marginal = []
    norm = tabla[tabla["etapa"].isin(["E1 individual", "E2 combinaciones"])].copy()
    norm["pasos"] = norm["config"].map(lambda c: texto_a_pasos(json.loads(c)["pasos"]))
    for modelo in MODELOS:
        t = norm[norm["model"] == modelo]
        for p in PASOS:
            con = t[t["pasos"].map(lambda s: p in s)]["macro_f1"]
            sin = t[t["pasos"].map(lambda s: p not in s)]["macro_f1"]
            marginal.append({"model": modelo, "paso": p, "promedio_con": round(con.mean(), 4),
                             "promedio_sin": round(sin.mean(), 4), "diferencia": round(con.mean() - sin.mean(), 4),
                             "n_con": len(con), "n_sin": len(sin)})
    pd.DataFrame(marginal).to_csv(SALIDA / "efecto_marginal_normalizacion.tsv", sep="\t", index=False)

    # Ablación de mejoras (formato del anexo de la práctica)
    abl = impacto[impacto["grupo"] == "mejoras de sentimiento"][
        ["model", "experiment_id", "tecnica", "f1_con", "delta_macro_f1", "f1neg_con", "delta_f1_negative"]]
    abl.columns = ["model", "experiment_id", "enhancement", "dev_macro_f1", "difference_vs_baseline",
                   "dev_f1_negative", "difference_f1_negative"]
    base_rows = pd.DataFrame([{"model": m, "experiment_id": hitos[m]["sin_mejoras"]["experiment_id"],
                               "enhancement": "baseline (sin aumento lingüístico)",
                               "dev_macro_f1": hitos[m]["sin_mejoras"]["macro_f1"], "difference_vs_baseline": 0.0,
                               "dev_f1_negative": hitos[m]["sin_mejoras"]["f1_negative"],
                               "difference_f1_negative": 0.0} for m in MODELOS])
    pd.concat([base_rows, abl]).sort_values(["model", "experiment_id"], kind="stable").to_csv(
        SALIDA / "ablacion_mejoras.tsv", sep="\t", index=False)

    # Predicciones y métricas en dev de la mejor configuración de cada modelo
    predicciones, metricas, errores, reportes = [], [], [], []
    for modelo in MODELOS:
        f = ganadoras[modelo]
        cfg = json.loads(f["config"])
        pipe = construir_pipeline(cfg).fit(train["text"], train["label"])
        pred = pipe.predict(dev["text"])
        m = calcular_metricas(dev["label"], pred)
        metricas.append({"experiment_id": f["experiment_id"], "model": modelo, "split": "dev", **m})
        predicciones.append(pd.DataFrame({
            "document_id": dev["document_id"], "gold_label": dev["label"], "predicted_label": pred,
            "split": "dev", "model": modelo, "experiment_id": f["experiment_id"]}))
        reportes.append(f"===== DEV | {f['experiment_id']} | {modelo}\n"
                        + classification_report(dev["label"], pred, labels=CLASES, digits=4, zero_division=0))
        figura_matriz(json.loads(m["confusion_matrix"]), f"Dev · {modelo} ({f['experiment_id']})",
                      FIGURAS / f"matriz_dev_{modelo}.png")
        for (_, r), p in zip(dev.iterrows(), pred):
            if p != r["label"]:
                lp, ln = puntajes_sentimiento(r["text"], "huliu")
                errores.append({
                    "model": modelo, "experiment_id": f["experiment_id"], "document_id": r["document_id"],
                    "item_name": r["item_name"], "gold_label": r["label"], "predicted_label": p,
                    "lex_pos": lp, "lex_neg": ln,
                    "tiene_negador": any(n in normalizar(r["text"], "limpieza+contracciones").split() for n in NEGADORES),
                    "review": " ".join(str(r["text"]).split()),
                    "texto_normalizado": normalizar(r["text"], cfg["pasos"], cfg["negacion"])})
    pd.DataFrame(errores).to_csv(SALIDA / "errores_dev.tsv", sep="\t", index=False)

    # Efecto de las mejoras sobre los errores en dev (modelo final)
    mf = final["model"]
    comparaciones = [("línea base", hitos[mf]["base"]), ("sin mejoras de sentimiento", hitos[mf]["sin_mejoras"]),
                     ("final", final)]
    pred_final = reg.preds.get(final["experiment_id"])
    if pred_final is None:     # la ganadora vino de GridSearch: se recalcula
        pred_final = construir_pipeline(json.loads(final["config"])).fit(
            train["text"], train["label"]).predict(dev["text"])
    cambios = []
    y = dev["label"].to_numpy()
    for nombre, f in comparaciones[:2]:
        p0 = reg.preds[f["experiment_id"]]
        for i, (_, r) in enumerate(dev.iterrows()):
            if p0[i] != pred_final[i]:
                cambios.append({"comparacion": f"{nombre} ({f['experiment_id']}) -> final ({final['experiment_id']})",
                                "document_id": r["document_id"], "gold_label": y[i], "antes": p0[i],
                                "despues": pred_final[i],
                                "efecto": "error corregido" if pred_final[i] == y[i] else
                                ("nuevo falso negativo" if y[i] == "negative" else "nuevo falso positivo"),
                                "review": " ".join(str(r["text"]).split())})
    pd.DataFrame(cambios).to_csv(SALIDA / "efecto_mejoras_en_errores.tsv", sep="\t", index=False)

    # Prueba final ÚNICA en test (se puede omitir con --sin-test mientras se desarrolla)
    cfg_final = json.loads(final["config"])
    if "--sin-test" in sys.argv:
        print(f"Configuración final: {final['experiment_id']} ({cfg_final['modelo']}) — test omitido")
        return
    train_dev = pd.concat([train, dev])
    modelo_final = construir_pipeline(cfg_final).fit(train_dev["text"], train_dev["label"])
    pred_test = modelo_final.predict(test["text"])
    m = calcular_metricas(test["label"], pred_test)
    metricas.append({"experiment_id": "test", "model": cfg_final["modelo"], "split": "test", **m})
    predicciones.append(pd.DataFrame({
        "document_id": test["document_id"], "gold_label": test["label"], "predicted_label": pred_test,
        "split": "test", "model": cfg_final["modelo"], "experiment_id": "test"}))
    reportes.append(f"===== TEST | configuración {final['experiment_id']} congelada | "
                    f"{cfg_final['modelo']} reentrenado con train+dev\n"
                    + classification_report(test["label"], pred_test, labels=CLASES, digits=4, zero_division=0))
    figura_matriz(json.loads(m["confusion_matrix"]),
                  f"Test · {cfg_final['modelo']} (config. {final['experiment_id']})", FIGURAS / "matriz_test_final.png")

    pd.concat(predicciones).to_csv(SALIDA / "classification_predictions.tsv", sep="\t", index=False)
    met = pd.DataFrame(metricas)
    met = met.round({c: 4 for c in met.columns if met[c].dtype == float})
    met.to_csv(SALIDA / "classification_metrics.tsv", sep="\t", index=False)
    (SALIDA / "classification_reports.txt").write_text("\n\n".join(reportes), encoding="utf-8")
    figura_etapas(evolucion, FIGURAS / "progreso_por_etapa.png")
    figura_impacto(impacto, final["model"], FIGURAS / "impacto_tecnicas_modelo_final.png")

    with open(MODELOS_DIR / "modelo_final.pkl", "wb") as fh:
        pickle.dump(modelo_final, fh)
    (MODELOS_DIR / "config_final.json").write_text(json.dumps(
        {"experiment_id": final["experiment_id"], "config": cfg_final,
         "descripcion": describir(cfg_final), "entrenado_con": "train+dev"},
        indent=2, ensure_ascii=False), encoding="utf-8")

    import imblearn
    import spellchecker
    (SALIDA / "entorno.txt").write_text("\n".join([
        f"python={platform.python_version()}", f"scikit-learn={sklearn.__version__}",
        f"imbalanced-learn={imblearn.__version__}", f"pyspellchecker={getattr(spellchecker, '__version__', 'n/d')}",
        f"nltk={nltk.__version__}", f"pandas={pd.__version__}", f"numpy={np.__version__}",
        f"random_state={SEMILLA} (particiones, LR, LinearSVC, SVD, oversampling, undersampling, SMOTE)",
        "particion=70/15/15 estratificada, shuffle=True", f"experimentos={len(tabla)}",
    ]) + "\n", encoding="utf-8")

    print("\nMejor de cada modelo en dev:")
    for modelo, f in ganadoras.items():
        print(f"  {modelo:<18} {f['experiment_id']}  acc={f['accuracy']:.4f}  macro-F1={f['macro_f1']:.4f}")
    print(f"Configuración final: {final['experiment_id']} ({cfg_final['modelo']})")
    print(f"TEST  accuracy={m['accuracy']:.4f}  macro-F1={m['macro_f1']:.4f}  F1 negative={m['f1_negative']:.4f}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
