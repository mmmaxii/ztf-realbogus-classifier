# API de ALeRCE — notas verificadas

Todo lo de este documento fue **verificado contra la API viva** con `alerce==2.3.0` (julio 2026),
por introspección (`inspect.signature`) y queries reales. No es copia de la documentación.

Para re-verificar si algo cambia:

```bash
.venv/Scripts/python.exe -c "from alerce.core import Alerce; import inspect; print(inspect.signature(Alerce.get_stamps))"
```

## De dónde salen las etiquetas real/bogus

ALeRCE corre varios clasificadores. La lista viva se obtiene con `client.query_classifiers(survey='ztf')`.

**El `lc_classifier` (light curve) NO sirve para este proyecto.** Sus 15 clases son
`SNIa, SNIbc, SNII, SLSN, QSO, AGN, Blazar, CV/Nova, YSO, LPV, E, DSCT, RRL, CEP, Periodic-Other`
— **ninguna es bogus**, porque solo clasifica objetos que ya pasaron el filtro de bogus.
Consultando por ahí no existen ejemplos negativos y el dataset no se puede armar.

**El `stamp_classifier` sí.** Es el que usamos:

| classifier_name | classifier_version | clases |
|---|---|---|
| `stamp_classifier` | `stamp_classifier_1.0.4` | `SN, AGN, VS, asteroid, bogus` ← **el que usamos** |
| `stamp_classifier` | `stamp_classifier_1.0.0` | `SN, AGN, VS, asteroid, bogus` |
| `stamp_classifier_2025_beta` | `beta` | `SN, AGN, VS, asteroid, bogus, satellite` |

Mapeo binario del proyecto: `bogus` → **0**, las otras cuatro → **1** (real).

Existe un `stamp_classifier_2025_beta` que agrega la clase `satellite`. No lo usamos por ser beta,
pero es interesante para el README: los satélites son una fuente creciente de falsos positivos.

## Firmas reales (verificadas por introspección)

```python
query_objects(self, format='pandas', index=None, sort=None, survey: str | None = None, **kwargs)
get_stamps(self, oid, candid=None, measurement_id=None, include_variance_and_psf=False, format='HDUList', survey=None)
query_probabilities(self, oid, format='json', survey: str | None = None, index=None, sort=None)
query_classifiers(self, format='json', survey: str | None = None, **kwargs)
query_classes(self, classifier_name, classifier_version, format='json', survey: str | None = None, **kwargs)
query_detections(self, oid: str | int, format: str = 'json', survey: str | None = None, index=None, sort=None)
```

`survey='ztf'` debe pasarse explícito en todos los métodos: el default está deprecado y emite
`DeprecationWarning`.

### `query_objects` — cuidado con los nombres de parámetros

Los filtros van por `**kwargs`, así que **un parámetro mal escrito no da error: se ignora en
silencio y te devuelve datos sin filtrar**. Los kwargs válidos para ZTF son:

`classifier`, `class_name`, `ndet`, `probability`, `firstmjd`, `lastmjd`, `ra`, `dec`, `radius`,
`page`, `page_size`, `count`, `order_by`, `order_mode`.

Es **`classifier` y `class_name`** — NO `classifier_name` (ese es solo de `query_classes`).

`probability=0.7` funciona como **cota inferior** (verificado: devuelve 0.707–0.770, ninguno por debajo).

**`classifier_version` funciona aunque no este en la lista documentada de kwargs — y hay que pasarlo.**
Verificado pidiendo la misma clase con las dos versiones:

```
v1.0.4:  [aaaafmw, aaaagix, aaaahbe, aaaalwz, aaaamms, ...]
v1.0.0:  [aaaajjr, aaaajpx, aaaakgr, aaaarem, aaaazmk, ...]   <- conjunto distinto
sin ver: [aaaafmw, aaaagix, aaaahbe, aaaajjr, aaaajpx, ...]   <- MEZCLA de las dos
```

Sin `classifier_version`, la API devuelve objetos etiquetados por **ambas versiones mezcladas**, o sea
el dataset tendria etiquetas de dos clasificadores distintos. Por eso el script fija
`stamp_classifier_1.0.4`.

Uso confirmado:

```python
df = client.query_objects(
    survey="ztf",
    classifier="stamp_classifier",
    classifier_version="stamp_classifier_1.0.4",
    class_name="bogus",
    probability=0.7,
    page_size=100,
    format="pandas",
)
```

**`probability` debe ser > 0.5.** Las probabilidades de las 5 clases suman 1, asi que con un umbral
por encima de 0.5 cada objeto puede calificar en una sola clase. Por debajo, el mismo `oid` podria
aparecer como bogus y como real a la vez, con etiquetas contradictorias. El script valida esto.

Devuelve un DataFrame de 23 columnas con índice por defecto (no `oid`). Las relevantes:
`oid`, `class`, `classifier`, `probability`, `ndet`, `firstmjd`, `lastmjd`, `meanra`, `meandec`.

### `get_stamps` — dos gotchas importantes

```python
stamps = client.get_stamps(oid=oid, survey="ztf", format="numpy")
```

- `format='numpy'` **sí existe** (las opciones son `'HDUList' | 'numpy'`). Devuelve una **lista
  de 3 arrays** `(63, 63)` en orden **science, template, difference**.
- `candid=None` (default) → **usa la primera detección**. Esto es exactamente lo que necesitamos:
  el `stamp_classifier` etiqueta la primera detección, así que con el default la etiqueta y la
  imagen corresponden al mismo evento. Pasar un `candid` arbitrario rompería esa correspondencia.

**Gotcha 1 — dtype big-endian.** Los arrays vienen como `>f4` (big-endian float32, herencia del
formato FITS). **PyTorch no soporta big-endian**: `torch.from_numpy()` falla con
`ValueError: given numpy array has byte order different from the native byte order`.
Hay que castear al guardar:

```python
arr = np.stack(stamps).astype(np.float32)  # >f4 -> float32 nativo
```

**Gotcha 2 — NaNs por recorte en el borde del CCD.** Una minoría de stamps traen NaNs, pero
cuando los traen son muchos (se observaron 144, 473, 1439 y 1792 de 3969 píxeles = hasta 45%).
Afectan **los 3 canales por igual**, lo que indica que la stamp fue recortada contra el borde
del detector y rellenada con NaN — no son píxeles muertos aislados. Se registra `nan_frac` en el
metadata para decidir en Fase 2 si se descartan o se imputan.

### Consistencia de tamaño

Verificado en 15 objetos de las 5 clases: **todas las stamps son (63, 63)**, sin excepción.
El input de la CNN a 63x63 es seguro. Aun así el script registra el shape en el metadata por si
aparece alguna distinta en una descarga grande.
