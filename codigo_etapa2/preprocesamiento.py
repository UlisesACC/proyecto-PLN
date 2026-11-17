"""
preprocesamiento.py — Piezas de texto que se usan dentro de los Pipelines (Etapa 2)

1. NORMALIZACIÓN por pasos independientes y combinables. Cada paso se puede
   activar solo o junto con otros (por eso se pueden probar combinaciones):
       limpieza       quita HTML/BBCode/URLs/caracteres de control, pasa a
                      minúsculas y descarta los signos de puntuación
       contracciones  "don't"/"dont" -> "do not", "can't" -> "can not", "i'm" -> "i am"
       repeticiones   "soooo" -> "soo" y "counter counter counter" -> "counter"
       ortografia     corrige palabras mal escritas con pyspellchecker
                      (protege nombres de juegos y jerga de videojuegos)
       lematizacion   lleva cada palabra a su lema con WordNet + etiqueta POS
       stopwords      quita palabras vacías de NLTK SIN quitar los negadores
   La línea base (N0) es "tokenizacion": solo partir el texto en tokens.

2. MANEJO DE NEGACIÓN (mejora de sentimiento), tres variantes:
       conservar  los negadores se quedan como cualquier palabra (por defecto)
       marcar     se antepone NEG_ a las 3 palabras siguientes al negador
       eliminar   control: se borran los negadores (para medir cuánto importan)

3. RECURSOS DE SENTIMIENTO (mejoras), cada uno agrega 2 columnas al vector:
       huliu         nº de palabras positivas / negativas del léxico de Hu & Liu
       sentiwordnet  suma de puntajes de polaridad positiva / negativa (SentiWordNet 3.0)
       emojis        suma de polaridad de emojis (léxico del curso) y emoticones
   En los tres, una palabra dentro del alcance de una negación invierte su
   polaridad ("not good" cuenta como negativa).

4. BALANCEO DE CLASES (solo durante el entrenamiento): class_weight,
   oversampling, undersampling o SMOTE.

5. construir_pipeline(cfg): arma el Pipeline completo a partir de un diccionario.
"""

import html
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

SEMILLA = 0
CARPETA = Path(__file__).resolve().parent
RECURSOS = CARPETA / "recursos"


# ---------------------------------------------------------------------------
# Recursos de NLTK (solo se descargan la primera vez)
# ---------------------------------------------------------------------------
def asegurar_recursos_nltk():
    necesarios = {
        "punkt_tab": "tokenizers/punkt_tab",
        "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
        "wordnet": "corpora/wordnet",
        "stopwords": "corpora/stopwords",
    }
    for recurso, ruta in necesarios.items():
        try:
            nltk.data.find(ruta)
        except LookupError:
            try:
                nltk.data.find(ruta + ".zip")
            except LookupError:
                nltk.download(recurso, quiet=True)


asegurar_recursos_nltk()
_lematizador = WordNetLemmatizer()

# ---------------------------------------------------------------------------
# Listas de palabras
# ---------------------------------------------------------------------------
# Palabras que niegan. Nunca se quitan como stopwords: "good" vs "not good".
NEGADORES = {"not", "no", "never", "nothing", "nobody", "none", "nowhere",
             "neither", "nor", "cannot", "without", "n't"}

# Pedazos de contracciones negativas que trae la lista de NLTK ("don", "isn"...)
RESIDUOS_NEGATIVOS = {"don", "doesn", "didn", "isn", "aren", "wasn", "weren", "hasn",
                      "haven", "hadn", "won", "wouldn", "shouldn", "couldn", "mustn",
                      "mightn", "needn", "shan", "ain"}
STOPWORDS = {
    p for p in stopwords.words("english")
    if p not in NEGADORES and "n't" not in p and p not in RESIDUOS_NEGATIVOS
} | {"'s", "'m", "'re", "'ve", "'ll", "'d"}

# Contracciones: token separado por word_tokenize -> forma completa
CONTRACCIONES = {"n't": "not", "ca": "can", "wo": "will", "sha": "shall", "'m": "am",
                 "'re": "are", "'ve": "have", "'ll": "will", "'d": "would", "im": "i am"}
# Contracciones escritas sin apóstrofo ("dont", "cant", "isnt")
RE_SIN_APOSTROFO = re.compile(
    r"\b(do|does|did|ca|wo|is|are|was|were|have|has|had|would|should|could|ai)nt\b", re.I)

FIN_DE_ALCANCE = {".", ",", "!", "?", ";", ":", "...", "--", ")", "but"}
VENTANA_NEGACION = 3          # palabras marcadas después de un negador

RE_BBCODE = re.compile(r"\[/?[a-zA-Z0-9*]+(=[^\]]*)?\]")      # [h1], [/b], [url=...]
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_ESPACIOS = re.compile(r"\s+")
RE_ALFANUM = re.compile(r"\w", re.UNICODE)
RE_LETRAS_REPETIDAS = re.compile(r"([^\W\d_])\1{2,}")         # 3 o más letras iguales seguidas

# Palabras que el corrector ortográfico NO debe tocar: nombres de los juegos,
# nombres propios del dominio y jerga de videojuegos.
PROTEGIDAS = {
    "counter", "strike", "cs", "cs2", "csgo", "dota", "portal", "stardew", "valley",
    "baldur", "baldurs", "bg3", "guilty", "gear", "strive", "valve", "steam", "larian",
    "glados", "wheatley", "aperture", "gabe", "ubisoft", "ea", "pelican", "pierre",
    "gg", "op", "nerf", "nerfed", "nerfs", "buff", "buffed", "noob", "noobs", "lol",
    "lmao", "ngl", "tbh", "imo", "idk", "fps", "pvp", "pve", "dlc", "dlcs", "mmo", "moba",
    "rpg", "crpg", "npc", "npcs", "afk", "smurf", "smurfs", "smurfing", "grind", "grindy",
    "laggy", "meta", "rng", "mmr", "elo", "hitbox", "hitreg", "speedrun", "coop",
    "multiplayer", "singleplayer", "gameplay", "playthrough", "cutscene", "cutscenes",
    "dev", "devs", "vac", "esports", "matchmaking", "respawn", "skins", "lootbox",
    "lootboxes", "microtransactions", "hero", "heroes", "rollback", "netcode", "combo",
    "combos", "frag", "frags", "aim", "aimbot", "wallhack", "griefer", "griefers",
    # Agregadas tras revisar en TRAIN las sobrecorrecciones de la primera versión
    # (resultados/cambios_ortografia.tsv): nombres propios, productos y jerga.
    "wingull", "replayability", "replayable", "modding", "modable", "moddable", "cooldown",
    "immersive", "rpgs", "crpgs", "robo", "arcsys", "moveset", "movesets", "badguy", "oled",
    "anticheat", "perma", "vacnet", "skyrim", "corwyn", "unironically", "lobcorp", "headshot",
    "headshots", "niko", "tryhard", "tryhards", "deathmatch", "comp", "valorant", "warcraft",
    "minecraft", "factorio", "timberborn", "yoshi", "chell", "joja", "xbox", "playstation",
    "nintendo", "minmax", "blyat", "pantera", "videogame", "videogames", "yall", "yknow",
    "thats", "theres", "youre", "whats", "theyre", "ive", "youve", "hes", "shes", "lets",
}

# Emoticones de texto (no hay léxico en el curso; lista propia con polaridad 1)
EMOTICONES_POS = {":)", ":-)", ":d", ":-d", ";)", ";-)", "=)", "<3", "xd", ":p", ":3", ";d", "c:"}
EMOTICONES_NEG = {":(", ":-(", ":'(", ":/", ":-/", ":[", "d:", ">:(", ":|", "):"}
RE_EMOTICON = re.compile(r"(?<!\w)(?:[:;=][\-']?[)(dDpP3/\[|]|<3|xd|c:|d:|>:\(|\):)(?!\w)", re.I)


# ---------------------------------------------------------------------------
# Pasos de normalización
# ---------------------------------------------------------------------------
PASOS = ["limpieza", "contracciones", "repeticiones", "ortografia", "lematizacion", "stopwords"]
NOMBRE_PASO = {"limpieza": "limpieza", "contracciones": "contracciones",
               "repeticiones": "repeticiones", "ortografia": "ortografía",
               "lematizacion": "lematización", "stopwords": "stopwords"}


def pasos_a_texto(pasos):
    """Convierte un conjunto de pasos en un nombre ordenado: 'limpieza+stopwords'."""
    pasos = [p for p in PASOS if p in set(pasos)]
    return "+".join(pasos) if pasos else "tokenizacion"


def texto_a_pasos(nombre):
    return set() if nombre in ("tokenizacion", "N0", "") else set(nombre.split("+"))


def limpiar(texto):
    """Paso 'limpieza': HTML, BBCode, URLs, caracteres de control y minúsculas."""
    if not isinstance(texto, str):
        return ""
    texto = html.unescape(texto)
    texto = RE_BBCODE.sub(" ", texto)
    texto = RE_URL.sub(" ", texto)
    texto = "".join(" " if unicodedata.category(c).startswith("C") else c for c in texto)
    texto = unicodedata.normalize("NFKC", texto)
    return RE_ESPACIOS.sub(" ", texto).strip().lower()


def _pos_wordnet(etiqueta):
    """Convierte la etiqueta POS de Penn Treebank al formato de WordNet."""
    if etiqueta.startswith("J"):
        return wn.ADJ
    if etiqueta.startswith("V"):
        return wn.VERB
    if etiqueta.startswith("R"):
        return wn.ADV
    return wn.NOUN


def normalizar_repeticiones(tokens):
    """'soooo' -> 'soo' (máx. 2 letras iguales) y quita tokens idénticos seguidos."""
    salida = []
    for t in tokens:
        t = RE_LETRAS_REPETIDAS.sub(r"\1\1", t)
        if salida and t.lower() == salida[-1].lower() and RE_ALFANUM.search(t):
            continue
        salida.append(t)
    return salida


_corrector = None


def _obtener_corrector():
    global _corrector
    if _corrector is None:
        from spellchecker import SpellChecker      # pyspellchecker
        _corrector = SpellChecker(language="en", distance=2)
    return _corrector


@lru_cache(maxsize=None)
def corregir_palabra(palabra):
    """Devuelve la palabra corregida, o la misma si no hace falta / no es seguro."""
    p = palabra.lower()
    if (len(p) < 4 or not p.isalpha() or p in PROTEGIDAS or p in NEGADORES
            or p in LEXICO_POS or p in LEXICO_NEG):
        return palabra
    corrector = _obtener_corrector()
    if p in corrector:                 # ya existe en el diccionario
        return palabra
    sugerida = corrector.correction(p)
    return sugerida if sugerida else palabra


def marcar_negacion(tokens):
    """Antepone NEG_ a las palabras que siguen a un negador, hasta un signo de
    puntuación, la palabra 'but' o VENTANA_NEGACION palabras.
    Ejemplo: ["not", "good", "game", "."] -> ["not", "NEG_good", "NEG_game", "."]"""
    salida, restantes = [], 0
    for tok in tokens:
        base = tok.lower()
        if base in FIN_DE_ALCANCE:
            restantes = 0
            salida.append(tok)
        elif base in NEGADORES:
            restantes = VENTANA_NEGACION
            salida.append(tok)
        elif restantes > 0 and RE_ALFANUM.search(tok):
            salida.append("NEG_" + tok)
            restantes -= 1
        else:
            salida.append(tok)
    return salida


@lru_cache(maxsize=None)
def normalizar(texto, pasos="tokenizacion", negacion="conservar"):
    """Normaliza UNA reseña y devuelve sus tokens separados por espacios.
    Orden fijo de aplicación: limpieza -> tokenización -> contracciones ->
    repeticiones -> ortografía -> lematización -> negación -> stopwords ->
    quitar signos (si hay limpieza). Se guarda en caché (lru_cache): no aprende
    nada del corpus, así que no hay fuga de datos."""
    activos = texto_a_pasos(pasos)
    texto = texto if isinstance(texto, str) else ""

    if "limpieza" in activos:
        texto = limpiar(texto)
    if "contracciones" in activos:
        texto = RE_SIN_APOSTROFO.sub(lambda m: m.group(1) + "n't", texto)
    tokens = word_tokenize(texto)
    if "contracciones" in activos:
        nuevos = []
        for t in tokens:
            nuevos.extend(CONTRACCIONES.get(t.lower(), t).split())
        tokens = nuevos
    if "repeticiones" in activos:
        tokens = normalizar_repeticiones(tokens)
    if "ortografia" in activos:
        tokens = [corregir_palabra(t) if t.isalpha() else t for t in tokens]
    if "lematizacion" in activos:
        tokens = [t if not RE_ALFANUM.search(t) or t.lower() in NEGADORES
                  else _lematizador.lemmatize(t, _pos_wordnet(tag))
                  for t, tag in pos_tag(tokens)]

    if negacion == "marcar":
        tokens = marcar_negacion(tokens)
    elif negacion == "eliminar":
        tokens = [t for t in tokens if t.lower() not in NEGADORES]

    resultado = []
    for t in tokens:
        if "limpieza" in activos and not RE_ALFANUM.search(t):
            continue
        base = (t[4:] if t.startswith("NEG_") else t).lower()
        if "stopwords" in activos and base in STOPWORDS:
            continue
        resultado.append(t)
    return " ".join(resultado)


class Normalizador(BaseEstimator, TransformerMixin):
    """Paso de Pipeline: recibe textos crudos y devuelve textos normalizados."""

    def __init__(self, pasos="tokenizacion", negacion="conservar"):
        self.pasos = pasos
        self.negacion = negacion

    def fit(self, X, y=None):
        return self          # regla fija: no aprende nada

    def transform(self, X):
        return [normalizar(str(t), self.pasos, self.negacion) for t in X]


# ---------------------------------------------------------------------------
# Recursos de sentimiento
# ---------------------------------------------------------------------------
def _leer_lista(nombre):
    with open(RECURSOS / nombre, encoding="latin-1") as f:
        return {l.strip().lower() for l in f if l.strip() and not l.startswith(";")}


LEXICO_POS = _leer_lista("positive-words.txt")            # Hu & Liu (2004)
LEXICO_NEG = _leer_lista("negative-words.txt")
_swn = pd.read_csv(RECURSOS / "sentiwordnet_compacto.tsv", sep="\t", keep_default_na=False)
SWN = {p: (a, b) for p, a, b in zip(_swn["palabra"], _swn["positivo"], _swn["negativo"])}
_emo = pd.read_csv(RECURSOS / "emoji_lexicon.tsv", sep="\t", keep_default_na=False)
EMOJIS = {e: (p, n) for e, p, n in zip(_emo["emoji"], _emo["positive"], _emo["negative"])}


def construir_sentiwordnet_compacto(ruta_original, ruta_salida=RECURSOS / "sentiwordnet_compacto.tsv"):
    """Reproduce recursos/sentiwordnet_compacto.tsv a partir de SentiWordNet_3.0.0.txt
    (Lexicons.zip del curso): por palabra, promedio de los puntajes de sus sentidos
    ponderado por 1/número de sentido (los sentidos más comunes pesan más)."""
    acumulado = {}
    with open(ruta_original, encoding="utf-8") as f:
        for linea in f:
            if linea.startswith("#") or not linea.strip():
                continue
            c = linea.rstrip("\n").split("\t")
            if len(c) < 5 or not c[2]:
                continue
            pos, neg = float(c[2]), float(c[3])
            for termino in c[4].split():
                palabra, sentido = termino.rsplit("#", 1)
                palabra = palabra.lower()
                if "_" in palabra:
                    continue
                peso = 1.0 / int(sentido)
                a = acumulado.setdefault(palabra, [0.0, 0.0, 0.0])
                a[0] += pos * peso
                a[1] += neg * peso
                a[2] += peso
    filas = [(p, round(a[0] / a[2], 4), round(a[1] / a[2], 4))
             for p, a in acumulado.items() if a[0] > 0 or a[1] > 0]
    pd.DataFrame(filas, columns=["palabra", "positivo", "negativo"]).sort_values(
        "palabra").to_csv(ruta_salida, sep="\t", index=False)


def _alcance_negado(tokens):
    """Lista de True/False: ¿la palabra i está dentro del alcance de una negación?"""
    marcas, restantes = [], 0
    for t in tokens:
        if t in FIN_DE_ALCANCE:
            restantes = 0
            marcas.append(False)
        elif t in NEGADORES:
            restantes = VENTANA_NEGACION
            marcas.append(False)
        elif restantes > 0 and RE_ALFANUM.search(t):
            marcas.append(True)
            restantes -= 1
        else:
            marcas.append(False)
    return marcas


@lru_cache(maxsize=None)
def puntajes_sentimiento(texto, recurso):
    """Devuelve (positivo, negativo) de una reseña según el recurso elegido.
    Dentro del alcance de una negación, positivo y negativo se intercambian."""
    limpio = RE_SIN_APOSTROFO.sub(lambda m: m.group(1) + "n't", limpiar(texto))
    if recurso == "emojis":
        pos = neg = 0.0
        for ch in limpio:
            if ch in EMOJIS:
                pos += EMOJIS[ch][0]
                neg += EMOJIS[ch][1]
        for e in RE_EMOTICON.findall(limpio):
            e = e.lower()
            pos += e in EMOTICONES_POS
            neg += e in EMOTICONES_NEG
        return pos, neg

    tokens = [CONTRACCIONES.get(t, t) for t in word_tokenize(limpio)]
    negadas = _alcance_negado(tokens)
    pos = neg = 0.0
    for t, negada in zip(tokens, negadas):
        if t in NEGADORES:
            continue
        if recurso == "huliu":
            p, n = float(t in LEXICO_POS), float(t in LEXICO_NEG)
        else:                                       # sentiwordnet
            p, n = SWN.get(t, (0.0, 0.0))
        if negada:
            p, n = n, p
        pos += p
        neg += n
    return pos, neg


class RasgosSentimiento(BaseEstimator, TransformerMixin):
    """Paso de Pipeline: 2 columnas [positivo, negativo] por reseña.
    Se aplica log(1 + x) para que reseñas muy largas no dominen y para que los
    valores queden en una escala parecida a la de las columnas de palabras.
    Los valores nunca son negativos (MultinomialNB los acepta)."""

    def __init__(self, recurso="huliu"):
        self.recurso = recurso

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log1p(np.array([puntajes_sentimiento(str(t), self.recurso) for t in X], dtype=float))


# ---------------------------------------------------------------------------
# Balanceo de clases (solo al entrenar)
# ---------------------------------------------------------------------------
class ClasificadorBalanceado(BaseEstimator, ClassifierMixin):
    """Envuelve un clasificador y, SOLO dentro de fit (es decir, solo con los
    datos de entrenamiento), aplica el método de balanceo elegido:
        None            sin balanceo
        class_weight    pesos 'balanced' (equivocarse con la clase pequeña cuesta más)
        oversampling    duplica al azar reseñas negativas hasta igualar a las positivas
        undersampling   quita al azar reseñas positivas hasta igualar a las negativas
        smote           crea reseñas negativas sintéticas interpolando vecinos (SMOTE)
    predict nunca remuestrea: dev y test se quedan con su distribución real."""

    def __init__(self, estimador=None, metodo=None, random_state=SEMILLA):
        self.estimador = estimador
        self.metodo = metodo
        self.random_state = random_state

    def fit(self, X, y):
        est = clone(self.estimador)
        if self.metodo in (None, "ninguno"):
            est.fit(X, y)
        elif self.metodo == "class_weight":
            est.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
        else:
            from imblearn.over_sampling import SMOTE, RandomOverSampler
            from imblearn.under_sampling import RandomUnderSampler
            muestreador = {"oversampling": RandomOverSampler(random_state=self.random_state),
                           "undersampling": RandomUnderSampler(random_state=self.random_state),
                           "smote": SMOTE(random_state=self.random_state, k_neighbors=5)}[self.metodo]
            X_res, y_res = muestreador.fit_resample(X, y)
            est.fit(X_res, y_res)
        self.estimador_ = est
        self.classes_ = est.classes_
        return self

    def predict(self, X):
        return self.estimador_.predict(X)

    def decision_function(self, X):
        return self.estimador_.decision_function(X)


# ---------------------------------------------------------------------------
# Armado del Pipeline completo
# ---------------------------------------------------------------------------
def crear_clasificador(modelo, hiper=None):
    hiper = dict(hiper or {})
    if modelo == "MultinomialNB":
        return MultinomialNB(alpha=hiper.get("alpha", 1.0))
    if modelo == "LogisticRegression":
        return LogisticRegression(C=hiper.get("C", 1.0), max_iter=5000, random_state=SEMILLA)
    if modelo == "LinearSVC":
        return LinearSVC(C=hiper.get("C", 1.0), max_iter=20000, random_state=SEMILLA)
    raise ValueError(f"Modelo desconocido: {modelo}")


def crear_vectorizador(representacion, ngramas):
    # Los textos ya vienen normalizados y separados por espacios: cada token es
    # cualquier secuencia sin espacios (\S+) y no se vuelve a pasar a minúsculas.
    opciones = dict(ngram_range=tuple(ngramas), token_pattern=r"\S+", lowercase=False)
    if representacion == "tfidf":
        return TfidfVectorizer(**opciones)
    if representacion == "binaria":
        return CountVectorizer(binary=True, **opciones)
    if representacion == "frecuencia":
        return CountVectorizer(**opciones)
    raise ValueError(f"Representación desconocida: {representacion}")


CONFIG_BASE = {
    "pasos": "tokenizacion", "negacion": "conservar", "representacion": "frecuencia",
    "ngramas": (1, 1), "svd": None, "recursos": [], "balanceo": None, "hiper": {},
}


def construir_pipeline(cfg):
    """cfg ejemplo:
    {"pasos": "limpieza+contracciones+stopwords", "negacion": "marcar",
     "representacion": "tfidf", "ngramas": (1, 2), "svd": None,
     "recursos": ["huliu"], "balanceo": "class_weight",
     "modelo": "LogisticRegression", "hiper": {"C": 10.0}}"""
    cfg = {**CONFIG_BASE, **cfg}
    pasos_texto = [
        ("normalizacion", Normalizador(cfg["pasos"], cfg["negacion"])),
        ("vectorizador", crear_vectorizador(cfg["representacion"], cfg["ngramas"])),
    ]
    if cfg["svd"]:
        # SVD solo con LR y LinearSVC: produce valores negativos que MultinomialNB no acepta
        pasos_texto.append(("svd", TruncatedSVD(n_components=cfg["svd"], random_state=SEMILLA)))
    rama_texto = Pipeline(pasos_texto)
    if cfg["recursos"]:
        ramas = [("texto", rama_texto)] + [(r, RasgosSentimiento(r)) for r in cfg["recursos"]]
        rasgos = FeatureUnion(ramas)
    else:
        rasgos = rama_texto
    return Pipeline([
        ("rasgos", rasgos),
        ("clasificador", ClasificadorBalanceado(crear_clasificador(cfg["modelo"], cfg["hiper"]),
                                                cfg["balanceo"])),
    ])


def describir(cfg):
    """Texto corto y legible de una configuración (para tablas y reportes)."""
    cfg = {**CONFIG_BASE, **cfg}
    ng = {(1, 1): "unigramas", (2, 2): "bigramas", (1, 2): "uni+bigramas"}[tuple(cfg["ngramas"])]
    pasos = texto_a_pasos(cfg["pasos"])
    norm = "tokenización" if not pasos else "tok + " + " + ".join(
        NOMBRE_PASO[p] for p in PASOS if p in pasos)
    mejoras = []
    if cfg["negacion"] == "marcar":
        mejoras.append("negación marcada (NEG_)")
    elif cfg["negacion"] == "eliminar":
        mejoras.append("negadores eliminados (control)")
    nombres = {"huliu": "léxico Hu&Liu", "sentiwordnet": "SentiWordNet", "emojis": "emojis/emoticones"}
    mejoras += [nombres[r] for r in cfg["recursos"]]
    defecto = "alpha=1.0" if cfg["modelo"] == "MultinomialNB" else "C=1.0"
    return {
        "normalizacion": norm,
        "representacion": f"{cfg['representacion']} {ng}" + (f" + SVD({cfg['svd']})" if cfg["svd"] else ""),
        "mejora_linguistica": " + ".join(mejoras) if mejoras else "ninguna",
        "desbalance": cfg["balanceo"] or "ninguno",
        "hiperparametros": ", ".join(f"{k}={v}" for k, v in cfg["hiper"].items()) or defecto,
    }
