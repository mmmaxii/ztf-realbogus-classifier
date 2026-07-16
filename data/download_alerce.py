"""
Descarga stamps de ZTF etiquetadas real/bogus usando el broker ALeRCE.

Las etiquetas vienen del `stamp_classifier` de ALeRCE (Carrasco-Davis et al. 2021),
que es el unico clasificador de ALeRCE con clase `bogus`. Sus 5 clases son
SN, AGN, VS, asteroid y bogus; las mapeamos a binario:

    bogus                       -> 0 (bogus)
    SN, AGN, VS, asteroid       -> 1 (real)

El `lc_classifier` (light curve) NO sirve para esto: solo clasifica objetos que ya
pasaron el filtro de bogus, asi que no tiene ejemplos negativos.

Detalle importante: el stamp_classifier etiqueta la PRIMERA deteccion de cada objeto.
`get_stamps(candid=None)` justamente devuelve la primera deteccion, asi que la etiqueta
y la imagen corresponden al mismo evento. No pasar un candid arbitrario.

Ver docs/alerce_api.md para las firmas verificadas de la API y sus gotchas.

Uso:
    python data/download_alerce.py --n-bogus 5            # smoke test rapido
    python data/download_alerce.py --n-bogus 500          # dataset real, balanceado
    python data/download_alerce.py --n-bogus 500 --resume # retomar una corrida cortada
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np
from alerce.core import Alerce

# Clases del stamp_classifier y su mapeo a binario.
BOGUS_CLASSES = ["bogus"]
REAL_CLASSES = ["SN", "AGN", "VS", "asteroid"]

LABEL_BOGUS = 0
LABEL_REAL = 1

CLASSIFIER = "stamp_classifier"
CLASSIFIER_VERSION = "stamp_classifier_1.0.4"

# Orden de los canales que devuelve get_stamps(format="numpy").
CHANNEL_NAMES = ["science", "template", "difference"]

METADATA_COLUMNS = [
    "oid",
    "label",  # 0 = bogus, 1 = real
    "stamp_class",  # clase original del stamp_classifier
    "probability",  # confianza de ALeRCE en esa clase
    "shape",
    "nan_frac",  # fraccion de pixeles NaN (recorte en el borde del CCD)
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def fetch_oids(client, class_name, n_objects, min_prob, page_size=100):
    """Devuelve hasta `n_objects` oids clasificados como `class_name` con probabilidad >= min_prob.

    Pagina hasta juntar los que se piden. `probability` funciona como cota inferior.
    Ojo: query_objects toma los filtros por **kwargs, asi que un nombre mal escrito no
    da error, simplemente se ignora y devuelve datos sin filtrar. Por eso validamos
    la respuesta abajo en vez de confiar.
    """
    collected = []
    page = 1

    while len(collected) < n_objects:
        try:
            df = client.query_objects(
                survey="ztf",
                classifier=CLASSIFIER,
                classifier_version=CLASSIFIER_VERSION,
                class_name=class_name,
                probability=min_prob,
                page=page,
                page_size=min(page_size, n_objects - len(collected)),
                format="pandas",
            )
        except Exception as exc:
            logger.error("Fallo la query de '%s' (pagina %d): %s", class_name, page, exc)
            break

        if df is None or len(df) == 0:
            break  # no hay mas resultados

        # Verificamos que el filtro se haya aplicado de verdad (ver docstring).
        wrong_class = df[df["class"] != class_name]
        if len(wrong_class) > 0:
            raise RuntimeError(
                f"query_objects devolvio clases inesperadas para class_name='{class_name}': "
                f"{sorted(wrong_class['class'].unique())}. El filtro no se aplico; "
                f"revisar los nombres de kwargs contra docs/alerce_api.md."
            )

        collected.extend(df[["oid", "class", "probability"]].to_dict("records"))
        page += 1

    return collected[:n_objects]


def download_stamps(client, oid, max_retries=3, backoff=2.0):
    """Descarga las 3 stamps de la primera deteccion de `oid` como un array (3, 63, 63).

    Reintenta con backoff exponencial: es una API publica y una descarga larga se
    cae a la mitad sin esto. Devuelve None si el objeto no tiene stamps disponibles.
    """
    for attempt in range(max_retries):
        try:
            # candid=None -> primera deteccion, que es la que el stamp_classifier etiqueto.
            stamps = client.get_stamps(oid=oid, survey="ztf", format="numpy")
        except Exception as exc:
            if attempt == max_retries - 1:
                logger.warning("  %s: fallo tras %d intentos (%s)", oid, max_retries, exc)
                return None
            time.sleep(backoff * (2**attempt))
            continue

        if stamps is None or len(stamps) != 3:
            logger.warning("  %s: se esperaban 3 stamps, llegaron %s", oid, stamps and len(stamps))
            return None

        # Los stamps vienen como '>f4' (big-endian, herencia de FITS) y PyTorch no
        # soporta big-endian: torch.from_numpy() falla. Casteamos a float32 nativo.
        return np.stack(stamps).astype(np.float32)

    return None


def load_done_oids(metadata_path, raw_dir):
    """Lee los oids ya descargados para poder retomar una corrida cortada (--resume)."""
    if not metadata_path.exists():
        return set()

    done = set()
    with metadata_path.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            npy_path = raw_dir / str(row["label"]) / f"{row['oid']}.npy"
            if npy_path.exists():  # el .npy tiene que existir de verdad, no solo la fila
                done.add(row["oid"])
    return done


def main():
    parser = argparse.ArgumentParser(
        description="Descarga stamps ZTF etiquetadas real/bogus desde ALeRCE.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-bogus",
        type=int,
        default=500,
        help="Cantidad de objetos bogus. Se descarga la misma cantidad total de "
        "objetos 'real', repartida entre las 4 clases reales, para que el "
        "dataset binario quede balanceado 1:1.",
    )
    parser.add_argument(
        "--min-prob",
        type=float,
        default=0.7,
        help="Probabilidad minima de ALeRCE. Mas alto = etiquetas mas limpias, menos objetos.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Pausa entre descargas, en segundos. Es una API publica y gratis: no la castigues.",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Saltea los objetos ya descargados."
    )
    args = parser.parse_args()

    # Las probabilidades de las 5 clases suman 1, asi que con un umbral > 0.5 cada objeto
    # puede calificar en una sola clase. Por debajo, el mismo oid podria entrar como bogus
    # Y como real a la vez, con etiquetas contradictorias.
    if args.min_prob <= 0.5:
        parser.error(
            f"--min-prob debe ser > 0.5 (se paso {args.min_prob}); si no, un mismo objeto "
            "puede quedar etiquetado en dos clases a la vez."
        )

    # bogus: n_bogus objetos. real: n_bogus repartidos entre las 4 clases reales.
    # Asi el binario queda ~1:1 en vez de 4:1 aplastando a bogus por construccion.
    n_per_real_class = max(1, round(args.n_bogus / len(REAL_CLASSES)))
    targets = [(c, LABEL_BOGUS, args.n_bogus) for c in BOGUS_CLASSES]
    targets += [(c, LABEL_REAL, n_per_real_class) for c in REAL_CLASSES]

    raw_dir = args.out_dir
    metadata_path = raw_dir / "metadata.csv"
    failures_path = raw_dir / "failures.log"
    for label in (LABEL_BOGUS, LABEL_REAL):
        (raw_dir / str(label)).mkdir(parents=True, exist_ok=True)

    done = load_done_oids(metadata_path, raw_dir) if args.resume else set()
    if done:
        logger.info("Modo --resume: %d objetos ya descargados, se saltean.", len(done))

    client = Alerce()

    # 1. Juntar los oids de cada clase antes de bajar nada, para reportar el plan.
    logger.info(
        "Buscando objetos en ALeRCE (%s, prob >= %.2f)...", CLASSIFIER_VERSION, args.min_prob
    )
    plan = []
    for class_name, label, n_objects in targets:
        objs = fetch_oids(client, class_name, n_objects, args.min_prob)
        logger.info("  %-9s -> %4d objetos (pedidos: %d)", class_name, len(objs), n_objects)
        if len(objs) < n_objects:
            logger.warning(
                "  %-9s: ALeRCE devolvio menos de lo pedido; bajar --min-prob daria mas.",
                class_name,
            )
        plan.extend((o, label) for o in objs)

    pending = [(o, lbl) for o, lbl in plan if o["oid"] not in done]
    logger.info("Total a descargar: %d stamps (%d ya estaban).", len(pending), len(plan) - len(pending))

    # 2. Descargar. El CSV se escribe incrementalmente para que una corrida cortada
    #    no pierda lo ya bajado.
    is_new_csv = not metadata_path.exists()
    n_ok, n_fail = 0, 0

    with metadata_path.open("a", newline="", encoding="utf-8") as fh, failures_path.open(
        "a", encoding="utf-8"
    ) as fail_fh:
        writer = csv.DictWriter(fh, fieldnames=METADATA_COLUMNS)
        if is_new_csv:
            writer.writeheader()

        for i, (obj, label) in enumerate(pending, start=1):
            oid = obj["oid"]
            arr = download_stamps(client, oid)

            if arr is None:
                n_fail += 1
                fail_fh.write(f"{oid}\t{obj['class']}\tno se pudieron bajar las stamps\n")
                fail_fh.flush()
                continue

            np.save(raw_dir / str(label) / f"{oid}.npy", arr)
            writer.writerow(
                {
                    "oid": oid,
                    "label": label,
                    "stamp_class": obj["class"],
                    "probability": round(float(obj["probability"]), 4),
                    "shape": "x".join(str(d) for d in arr.shape),
                    "nan_frac": round(float(np.isnan(arr).mean()), 4),
                }
            )
            fh.flush()
            n_ok += 1

            if i % 25 == 0 or i == len(pending):
                logger.info("  %d/%d (ok: %d, fallidos: %d)", i, len(pending), n_ok, n_fail)

            time.sleep(args.sleep)

    logger.info("Listo. %d descargados, %d fallidos.", n_ok, n_fail)
    logger.info("Datos en %s | metadata en %s", raw_dir, metadata_path)
    if n_fail:
        logger.info("Objetos fallidos listados en %s", failures_path)

    # Exito = no fallo nada. Una corrida con --resume donde ya estaba todo bajado
    # descarga 0 objetos y eso es correcto, no un error.
    if pending and n_ok == 0:
        logger.error("No se pudo descargar ningun objeto de los %d pendientes.", len(pending))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
