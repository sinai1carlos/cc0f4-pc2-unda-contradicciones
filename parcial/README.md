# CC0F4 - Examen parcial: recuperación, RAG y una herramienta con contratos sobre el Código Civil peruano

**Autor:** Carlos Sinai Unda Miguel


## 1. Problema y continuidad

El proyecto busca detectar contradicciones entre normas. Para eso hacen falta dos piezas: encontrar los artículos que regulan el mismo supuesto y decidir cómo se relacionan.

| Entrega | Qué se construyó | Qué quedó pendiente |
|---|---|---|
| PC1 | Clasificador con Qwen2.5-0.5B sobre pares de normas ya emparejados, salida JSON validada | El par llegaba hecho a mano; 9 casos sintéticos |
| PC2 | Recuperación densa (E5 + FAISS) con consulta estructurada por el LLM | Corpus sintético de 10 documentos; 4 de 6 JSON inválidos por una tilde en el enum (`prohibición_permiso` frente a `prohibicion_permiso`) |
| Parcial | Recuperación A/B/C sobre el Código Civil real, RAG con citas evaluado a mano y una herramienta de cómputo de plazos con contratos | Ver secciones 9 y 10 |

Cambios por las observaciones de la PC2:

- **Corpus real** en lugar de documentos sintéticos (sección 3).
- **La tilde.** JSON Schema compara las cadenas de un `enum` carácter por carácter, y "ó" (U+00F3) y "o" (U+006F) son caracteres distintos. Un solo carácter hacía inválido todo el objeto, la consulta estructurada no pasaba la validación y la búsqueda no llegaba a ejecutarse. 
- **Reproducibilidad.** `requirements.txt`, `Dockerfile`.

## 2. Ejecución

```bash
docker build -t cc0f4-parcial .
docker run --rm -p 8888:8888 -v "$PWD":/workspace/parcial cc0f4-parcial
```

También corre en la imagen del curso (`cc0f4:2026-2`) con `pip install -r requirements.txt`. Abrir y ejecutar en orden, en CPU:

| Notebook | Experimento | Tiempo aproximado |
|---|---|---|
| `notebooks/03_herramienta.ipynb` | 3, herramienta | segundos |
| `notebooks/01_recuperacion.ipynb` | 1, recuperación A/B/C | 2 a 3 min para indexar + 1 min por condición C |
| `notebooks/02_rag.ipynb` | 2, closed-book frente a RAG | unos 10 min para las 24 generaciones |


```text
parcial/
├── README.md, Dockerfile, requirements.txt
├── data/        corpus, consultas, qrels, referencias, feriados, casos de la herramienta (README propio)
├── prompts/     rag_sistema.txt, rag_usuario.txt, closed_book_usuario.txt
├── src/         config, corpus, recuperacion, metricas, rag, herramienta_plazos
├── notebooks/   01_recuperacion, 02_rag, 03_herramienta (ejecutados)
├── results/     métricas, rankings, respuestas crudas, auditoría manual, trazas
└── slides/      Examen_Parcial_CC0F4.pdf
```

## 3. Corpus, consultas y qrels

**Corpus.** Código Civil peruano (D.L. 295), versión `corpus_cc_peru_v2.json`: 2135 artículos, de los que se indexan los **2115 vigentes**. Cada artículo es un passage con identificador estable (`cc_art_183`). Longitud media 43 palabras, mediana 33, máximo 500. Siete artículos con texto dudoso respecto de la reforma del D.Leg. 1384 (43, 44, 241, 274, 334, 347, 694) se indexan pero no se usan como evidencia relevante. Procedencia, versión y criterios en `data/README.md`.

El enunciado pone como ejemplo "100 pasajes de al menos 15 documentos". Aquí la fuente es una sola norma, pero cada artículo es una disposición independiente con su propio número, repartida en el Título Preliminar y 10 libros, y el corpus tiene 2115 passages. Eso cumple el propósito del mínimo: evitar coincidencias fáciles y rankings inflados.

**Consultas.** 20 consultas escritas antes de ejecutar cualquier sistema, con 31 pares relevantes:

| tipo | n | qué prueba | ejemplo |
|---|---|---|---|
| directa | 12 | vocabulario del Código | ¿En cuántos años prescribe la acción de nulidad del acto jurídico? |
| parafraseada | 5 | lenguaje cotidiano | Vivo en una casa desde hace diez años como si fuera mía, ¿puedo convertirme en propietario? |
| contradiccion | 3 | regla general frente a excepción | Si el último día de un plazo de caducidad cae en día inhábil, ¿el plazo se extiende? (arts. 183 y 2007) |

Cada consulta guarda en `justificacion_qrels` el inciso que justifica su relevancia. Los distractores (artículos con el mismo vocabulario que no responden, como el 2002 frente al 2007) no están en los qrels.

## 4. Arquitectura

```text
pregunta del usuario
  -> A: E5 (query:/passage:, L2) + FAISS IndexFlatIP ............ top-20
  -> B: A + BM25 (k1 1.5, b 0.75, stopwords del Cuaderno 7) + RRF (k = 60)
  -> C: B + Cross-Encoder mmarco-mMiniLMv2 sobre los 20 candidatos
  -> top-3 de C como EVIDENCIA [cc_art_N]
  -> Qwen2.5-0.5B-Instruct, greedy, 220 tokens
  -> JSON {respuesta, citas, abstencion} -> JSON Schema -> citas dentro del contexto
  -> auditoría manual de corrección, citas y respaldo

herramienta independiente (sin LLM ni agente)
  plazos.computar_vencimiento -> contrato de entrada -> dominio -> reintentos/timeout -> contrato de salida
```

Unidad de recuperación: el artículo. En la PC2 el tamaño de chunk no cambió Recall@3, y el artículo es la unidad que se cita en derecho, así que el ranking de chunks coincide con el ranking de passages de los qrels.

## 5. Matriz de trazabilidad

| Semana | Recurso | Concepto | Dónde se aplica | Evidencia | Conclusión |
|---|---|---|---|---|---|
| 1 | Cuaderno 1, Lectura 2 | tokenización, atención, máscara causal | `02_rag.ipynb`, sección 1 | "caducidad" se divide en `cad`, `uc`, `idad`; Qwen tiene 24 capas | La evidencia va antes de la pregunta, porque cada token solo ve los anteriores |
| 2 | Cuaderno 2, Laboratorio 2, Lectura 3 | logits, softmax, greedy | `02_rag.ipynb`, sección 4 | Con evidencia, el prompt pasa de 179 a 602 tokens y la latencia de 14.9 s a 34.5 s | Greedy hace que el mismo prompt dé la misma salida |
| 3 | Cuaderno 3 | JSON Schema y validación | `src/rag.py` | Schema válido: 8/12 sin evidencia y 12/12 con evidencia | Cumplir el schema no significa responder bien |
| 4 | Cuaderno 4, Laboratorio 4 | embeddings E5, coseno, FAISS | `01_recuperacion.ipynb`, secciones 2 y 3 | El score de FAISS coincide con el coseno calculado a mano; Recall@3 de A = 0.658 | La búsqueda densa de la PC2 es la línea base |
| 5 | Cuaderno 5, Lectura 5 | BM25, RRF, Cross-Encoder | `src/recuperacion.py` | En C13, BM25 baja el art. 333 del puesto 3 al 13 | Agregar BM25 trae distractores con las mismas palabras |
| 6 | Cuaderno 6, Lectura 6 | contratos, reintentos, timeout | `src/herramienta_plazos.py` | 13 de 13 casos correctos; H09 se recupera en el segundo intento | Solo se reintenta lo que puede cambiar entre intentos |
| 7 | Cuaderno 7, Lectura 7 | Recall@k, MRR, nDCG, bootstrap, faithfulness | `src/metricas.py`, `results/` | nDCG@5 de C frente a B: +0.084, IC [0.001, 0.179]; 3 de 12 afirmaciones respaldadas | Un mejor ranking no garantiza una mejor respuesta |

## 6. Experimento 1: recuperación y reranking

**Pregunta.** ¿Mejora la recuperación al agregar BM25 con RRF a la búsqueda densa (B frente a A) y luego un Cross-Encoder (C frente a B)?

**Línea base:** A, la búsqueda densa de la PC2. **Fijos:** corpus, 20 consultas, qrels y 20 candidatos por consulta. **Modificación:** B agrega BM25 y fusiona con RRF; C reordena los 20 candidatos de B con un Cross-Encoder. **Métricas:** Recall@1/3/5, MRR, nDCG@5 y latencia.

| Condición | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@5 | Latencia media |
|---|---|---|---|---|---|---|
| A: denso | 0.442 | **0.658** | **0.700** | 0.688 | 0.642 | 60 ms |
| B: + BM25 + RRF | 0.392 | 0.558 | 0.650 | 0.629 | 0.582 | 52 ms |
| C: + Cross-Encoder | **0.542** | 0.633 | 0.675 | **0.750** | **0.666** | 3123 ms |

Latencia por etapa: búsqueda densa unos 55 ms, BM25 unos 4 ms, RRF 0.1 ms y Cross-Encoder 3061 ms.

Bootstrap pareado (5000 remuestreos, `results/exp1_bootstrap.csv`): la única diferencia cuyo intervalo no cruza cero es nDCG@5 de C frente a B, +0.084, IC 95 % [0.001, 0.179]. B queda por debajo de A en todas las métricas, pero sus intervalos cruzan cero.

**Error.** En C13 (separación de hecho), el art. 333 está en el puesto 3 en A. En B cae al 13, porque BM25 premia los arts. 2081 y 2082 de Derecho Internacional Privado, que repiten "divorcio y separación de cuerpos" pero regulan otra cosa. C lo devuelve al puesto 3. Tres consultas parafraseadas (C07, C12 y C19) no tienen ningún artículo relevante entre los 20 candidatos en ninguna condición, así que el reranker no puede corregirlas.

**Limitación.** Con 20 consultas, una sola consulta mueve hasta 5 puntos un promedio.

**Conclusión.** BM25 con RRF no mejoró la búsqueda densa. El Cross-Encoder mejoró el orden, pero multiplica la latencia por unas 60 veces y no recupera artículos que no estén entre los candidatos.

## 7. Experimento 2: RAG y evidencia

**Pregunta.** ¿Mejora la respuesta de Qwen2.5-0.5B cuando recibe los 3 primeros artículos de la condición C?

**Línea base:** closed-book, sin evidencia. **Fijos:** modelo, prompt de sistema, decoding greedy, 220 tokens y las mismas 12 preguntas: 10 con respuesta en el Código y 2 sin ella (N01, plazo para apelar una sentencia civil; N02, remuneración mínima vital). **Modificación:** evidencia en el prompt. **Métricas:** cumplimiento del schema, corrección, citas, afirmaciones respaldadas por el contexto (faithfulness) y abstención. La auditoría manual está en `results/auditoria_manual_rag.csv`.

| | closed-book | RAG |
|---|---|---|
| Cumple el schema | 8/12 | 12/12 |
| Correctas (10 preguntas) | 2/10 | 1.5/10 |
| Respuestas que citan un artículo | 0/10 | 0/10 |
| Afirmaciones respaldadas | 0/12 | 3/12 |
| Abstención en N01 y N02 | 0/2 | 1/2 |
| Latencia media | 14.9 s | 34.5 s |

La evidencia completa llegó al contexto en 7 de las 10 preguntas. Ninguna respuesta citó un artículo, así que no hubo citas que verificar.

**Respuesta no sustentada.** En C06, el art. 2001 llegó primero y dice que la acción de nulidad prescribe a los diez años. El modelo respondió "tres": tomó el inciso 3, que trata otra acción. Sin evidencia había respondido bien.

**Otros errores.** En C05, con los arts. 183 y 2007 en el contexto, aplicó la regla general y no la excepción de la caducidad. En N01 respondió "cinco días" a partir del art. 648, que trata otra apelación, en vez de abstenerse.

**Limitación.** Un solo modelo de 0.5B en CPU, y la auditoría la hizo una sola persona.

**Conclusión.** La recuperación entregó la evidencia, pero el modelo no la usó: el RAG mejoró la forma de la salida, pero no la corrección ni las citas. El cuello de botella está en el generador.

## 8. Experimento 3: herramienta con contratos

**Pregunta.** ¿La herramienta distingue cada tipo de error y reintenta solo cuando corresponde?

**Herramienta.** `plazos.computar_vencimiento` aplica el art. 183 (cómputo de plazos y traslado al día hábil siguiente) y la excepción del art. 2007 (la caducidad no se traslada). **Línea base:** 13 casos con el resultado esperado calculado a mano. **Modificación:** fallos inyectados. **Contratos:** la entrada se valida con Pydantic, el dominio exige que el vencimiento caiga dentro del calendario 2026 y la salida se valida con `PlazoOutput`. **Política:** hasta 3 intentos, 200 ms por intento y 800 ms en total.

| Caso | Situación | Resultado | Intentos |
|---|---|---|---|
| H01 | 30 días desde el 1/10 | 2/11 (sábado y feriado trasladados) | 1 |
| H03 | el mismo plazo como caducidad | 31/10, sin traslado (art. 2007) | 1 |
| H05 | cantidad negativa | `validation` | 0 |
| H08 | vencimiento en 2027 | `domain`: fuera del calendario | 1 |
| H09 | falla transitoria una vez | ok en el segundo intento | 2 |
| H11 | fuente lenta | `timeout` en los 3 intentos (753 ms) | 3 |
| H12 | salida que no cumple el contrato | `output_contract` | 1 |

Los 13 casos dieron el resultado esperado (`results/exp3_resumen.csv`).

**Reintentos.** Solo se reintentan `transient` y `timeout`, porque la fuente puede responder en otro intento. Un error de validación o de dominio se repetiría con los mismos datos, y un error de contrato de salida es un defecto del código que reintentar ocultaría.

**Error.** En H08 los datos son válidos, pero el vencimiento cae en 2027 y no hay feriados cargados para ese año. La herramienta devuelve un error en lugar de inventar una fecha.

**Limitación.** Las fallas son simuladas y el calendario cubre solo 2026.

**Conclusión.** La herramienta clasificó los 13 casos y reintentó solo los errores que pueden cambiar entre intentos.

## 9. Resultados

| Exp. | Línea base | Modificación | Resultado | Conclusión |
|---|---|---|---|---|
| 1 | A denso | B: + BM25 + RRF; C: + Cross-Encoder | B no mejora; C mejora nDCG@5 en +0.084 | El reranker mejora el orden, no la cobertura |
| 2 | closed-book | top-3 de C en el prompt | 1.5 frente a 2 correctas de 10; 0 citas | El modelo de 0.5B no usa la evidencia |
| 3 | 13 casos esperados | fallos inyectados | 13 de 13 | Reintento solo en `transient` y `timeout` |

## 10. Limitaciones

- **20 consultas y qrels.** No existe un conjunto de consultas anotadas por terceros para el Código Civil peruano. Cada qrel tiene su justificación escrita y se fijó antes de ejecutar, pero nadie más lo validó. Con 20 consultas, ningún intervalo bootstrap prueba una generalización.
- **Una sola fuente normativa**, extraída de PDF, con 7 artículos de texto dudoso. 4 artículos largos (104, 274, 277 y 1138) se truncan a 512 tokens en E5.
- **Un generador de 0.5B en CPU.** Los resultados del experimento 2 describen este modelo, no el RAG en general. La auditoría manual la hizo una sola persona.
- **Latencia en CPU** con las etapas en secuencia; no es un benchmark de rendimiento.
- **Herramienta** con fallas simuladas, calendario de un año y un supuesto propio de día hábil.

## 11. Conclusiones

Bajo este corpus, estas 20 consultas y estos modelos:

1. La recuperación densa sola dio el mayor Recall@3 (0.658), y BM25 con RRF no la mejoró en ninguna métrica.
2. El Cross-Encoder ordenó mejor (MRR 0.750, nDCG@5 0.666) a un costo de 3 segundos por consulta, y no recupera lo que no está entre los candidatos.
3. Las consultas en lenguaje cotidiano son el punto débil: tres de cinco no encuentran ningún artículo relevante en ninguna condición.
4. Con Qwen2.5-0.5B, recibir la evidencia no mejoró la corrección y el modelo no citó en ningún caso.
5. La herramienta de plazos clasificó los 13 casos y reintentó solo cuando correspondía.

Para la etapa de workflows y agentes, lo primero que conviene cambiar es el generador, antes que el recuperador.

## 12. Fuentes

**Técnicas (método):**
- Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020. Marco del experimento 2.
- Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation.* arXiv:2309.15217. Separación entre faithfulness y corrección; aquí se audita a mano en lugar de usar un juez automático.
- Qwen Team (2024). *Qwen2.5 Technical Report.* arXiv:2412.15115.

**Dominio:**
- Código Civil peruano, Decreto Legislativo 295 (1984), texto de la versión del corpus descrita en `data/README.md`.

**Material del curso** cuadernos 1 a 7, laboratorios 2 y 4, y lecturas 2, 3, 5, 6 y 7, citados en la matriz de la sección 5.
