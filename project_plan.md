# ZTF Real/Bogus Classifier - Project Plan

## Decisiones Iniciales

**Nombre del repo:** ztf-realbogus-classifier
(simple, describe exactamente qué hace, se entiende en 2 segundos en un CV o LinkedIn — nada de nombres creativos que en una entrevista tenés que explicar)

**Modelo:** Arrancás con una CNN custom pequeña (no ResNet, no transfer learning como base). Razones:

* Las stamps son 63x63px — chiquitas. Un ResNet18 pre-entrenado en ImageNet está pensado para imágenes de objetos naturales a 224x224; es overkill y en la práctica no mejora sobre una CNN chica bien diseñada para este dominio (esto está documentado en los papers de ALeRCE y Zoo/Bogus classifiers, así que no es una intuición tuya nomás, tenés literatura que la respalda).
* Una CNN custom bien armada + bien explicada en el README vale más en la entrevista que un "usé un modelo pre-entrenado" — te preguntan "por qué esta arquitectura" y tenés respuesta técnica real, no "porque es el default de torchvision".
* Después, como bonus si te sobra tiempo, hacés la comparación contra ResNet18 fine-tuneado y mostrás con números que tu CNN chica gana o empata con muchas menos parámetros. Eso es una sección de "Model Comparison" que impresiona mucho más que partir directo con transfer learning.

**Arquitectura concreta de la CNN:**
* Input: 3 canales (science, template, difference), 63x63
* 3 bloques Conv2d + BatchNorm + ReLU + MaxPool (16→32→64 filtros)
* Flatten → FC(128) → Dropout(0.3) → FC(2)
* Loss: CrossEntropyLoss con class weights (por el desbalance real/bogus)
* Optimizer: Adam, lr=1e-3 con ReduceLROnPlateau

**Stack completo:**
* PyTorch + torchvision (transforms)
* astropy para leer FITS si los stamps vienen en ese formato
* FastAPI para el endpoint de inferencia
* Docker + GitHub Actions (mismo patrón que ya tenés en el proyecto de Seeds — consistencia entre portfolios se nota)
* MLflow opcional para trackear experimentos (le suma "sabe hacer MLOps" sin ser el foco)

**Estructura de carpetas:**
```text
ztf-realbogus-classifier/
├── data/              # scripts de descarga, no los datos crudos
├── notebooks/
│   └── 01_eda_stamps.ipynb
├── src/
│   ├── dataset.py      # Dataset + DataLoader custom
│   ├── model.py         # CNN
│   ├── train.py
│   ├── evaluate.py      # métricas + ROC/AUC + comparación ALeRCE
│   └── gradcam.py
├── api/
│   └── main.py          # FastAPI
├── Dockerfile
├── .github/workflows/ci.yml
├── requirements.txt
└── README.md
```

## Análisis (Pros y Contras)

### 👍 PROS (Por qué este proyecto brilla)
1. **Justificación Técnica (El mayor punto fuerte):** Explicar por qué no usas un ResNet18 (imágenes 63x63 vs 224x224, sobreparametrización) demuestra que entiendes lo que haces y no solo copias tutoriales. Esa discusión en un README o en una entrevista vale oro puro.
2. **End-to-End Real (MLOps + Ingeniería):** No te quedas en el Jupyter Notebook. Tienes el entrenamiento (`train.py`), la API de inferencia (`FastAPI`), la contenedorización (`Docker`) y CI/CD (`GitHub actions`). Esto grita "estoy listo para la industria y sé poner modelos en producción".
3. **Interpretabilidad (`gradcam.py`):** En ciencia (y finanzas/medicina), la caja negra no sirve. Incluir GradCAM para mostrar dónde está mirando la CNN para decir que algo es "Real" o "Bogus" demuestra un nivel de madurez avanzado.
4. **Desbalance de Clases:** Mencionas usar `class weights` y ROC/AUC. El mundo real está desbalanceado (hay mucha más basura "bogus" que eventos reales "real"). Abordar esto frontalmente es un gran plus.
5. **Nombre Perfecto:** `ztf-realbogus-classifier` es directo, profesional y súper "Googleable" para reclutadores técnicos.

### ⚠️ CONTRAS / RETOS A TENER EN CUENTA (Para que no te trabes)
1. **Obtención y Limpieza de Datos:** 
   * *El reto:* Las alertas de ZTF vienen originalmente en formato AVRO o desde APIs de brokers como ALeRCE. Bajar, extraer los *stamps* (comprimidos o FITS) y alinearlos (Science, Template, Difference) puede ser tedioso.
   * *Solución:* Asegúrate de tener una fuente clara (ej. descargar un dataset ya curado o armar un script robusto que lea los AVROs y genere un `.h5` o `.npy` consolidado para el Dataloader). No leas FITS o AVROs "al vuelo" durante el entrenamiento porque será un cuello de botella por I/O.
2. **Cuidado con los NaNs/Infinitos:** Los *stamps* astronómicos de diferencia a veces tienen pixeles muertos, NaNs o valores extremos (fondo negativo). Tendrás que aplicar un buen paso de normalización y reemplazo de NaNs en tu pipeline de datos.
3. **Riesgo de Sobreajuste (Overfitting):** Tu CNN es pequeña (excelente decisión), pero si tu dataset de entrenamiento no tiene mucha variabilidad o es pequeño, igual puedes sobreajustar. Mantén el `Dropout(0.3)` y evalúa agregar *Data Augmentation* ligero (rotaciones de 90 grados, flips, ya que en el espacio no hay "arriba" ni "abajo").
