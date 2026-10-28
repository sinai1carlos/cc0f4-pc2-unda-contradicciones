# CC0F4 - PC2: De una consulta estructurada a recuperación densa verificable

Todo el experimento vive en `notebook/PC2_unda.ipynb`. Se ejecuta en CPU dentro de un contenedor Docker con JupyterLab: abrir el notebook y correr las celdas en orden. Los resultados se escriben en `results/`.

La segmentación (`build_chunks` con `target_words` y `overlap_passages`) y la definición de Recall@k sobre `passage_id` . Todos los resultados de este README corresponden a una corrida con los modelos reales (`RUN_REAL_RETRIEVAL = True`, `RUN_REAL_LLM = True`).

```text
cc0f4-pc2-unda-contradicciones/
├── README.md
├── requirements.txt
├── notebook/PC2_unda.ipynb
├── data/
│   ├── corpus.jsonl                  (10 documentos, 76 passages con passage_id)
│   ├── queries.jsonl                 (necesidades Q0 con la evidencia de cada qrel)
│   ├── qrels.json                    (query_id -> passage_id relevantes)
│   └── query_schema.json             (JSON Schema de la consulta estructurada)
├── prompts/estructurar_consulta.txt
├── results/                          (lo genera el notebook)
└── slides/PC2_unda.pdf
```

## 1. Problema de recuperación

En la PC1 el clasificador NLI recibía el par (Norma A, Norma B) ya emparejado. En un sistema real nadie entrega ese par: dada una norma o una pregunta, primero hay que **encontrar las otras disposiciones que regulan el mismo supuesto**, porque solo entre ellas puede haber contradicción. Esta práctica implementa y evalúa únicamente ese paso de recuperación.

Pregunta auxiliar del enunciado: *¿qué evidencia textual necesitaría recuperar el sistema antes de decidir si hay contradicción?* Respuesta: los artículos que regulan el mismo supuesto que la necesidad del usuario, idealmente los dos lados del conflicto (Norma A y Norma B).

La salida recuperada no se pasa todavía al LLM ni al clasificador NLI.

## 2. Corpus y consultas

**Corpus:** 10 documentos normativos **sintéticos** (reglamentos, directivas y ordenanzas municipales ficticias, entre 177 y 380 palabras). Cada `passage` es un artículo completo, 76 en total, de 15 a 54 palabras, identificado como `D05-A06` (documento D05, artículo 6). Como en la PC1, no citan normas reales para no atribuir texto inventado a dispositivos existentes. Cada documento incorpora literalmente normas de los 9 pares , más artículos distractores con vocabulario parecido (otros plazos en días hábiles, otras vigencias de dos años, otras publicaciones en el portal). 

**Consultas (Q0):** 6 necesidades de información redactadas por el autor en lenguaje natural, una por cada caso 1 a 6 de la PC1 (los casos con relación `contradiction` o `entailment`, que son los que tienen "otra norma" que encontrar).

**qrels** (`data/qrels.json`): para cada necesidad, los `passage_id` de la Norma A y la Norma B de ese caso, siempre en documentos distintos. Como en el Cuaderno 5, la relevancia se define sobre `passage_id` y no depende del `chunk_id`, así que es la misma en las tres condiciones de segmentación.

| query_id | Necesidad (Q0) | qrels |
|---|---|---|
| N1 | ¿Hasta qué fecha tengo para presentar mi declaración jurada del mes? | D01-A04 (10 días hábiles), D02-A04 (30 días calendario) |
| N2 | ¿Puedo entrar con mi auto particular a la zona reservada? | D03-A04 (prohíbe), D04-A04 (permite) |
| N3 | ¿Cuántos días tengo para presentar la apelación? | D05-A06 (15 días hábiles), D06-A07 (5 días hábiles) |
| N4 | ¿Tengo que renovar el permiso de operación cada año? | D04-A02 (cada año), D07-A04 (dos años) |
| N5 | ¿Qué entidades deben publicar su presupuesto anual? | D08-A02 (toda entidad), D09-A04 (sector salud) |
| N6 | ¿Cuántos días de vacaciones me corresponden por un año? | D06-A04, D10-A04 |

Texto completo de cada Q0 en `data/queries.jsonl`.

## 3. Consulta estructurada y JSON Schema

Un LLM transforma cada Q0 en una salida pequeña y validable:

```json
{"query": "plazo para interponer recurso de apelación", "intent": "plazo", "abstain": false}
```

`data/query_schema.json`: `query` string de 3 a 200 caracteres; `intent` enum {`plazo`, `prohibicion_permiso`, `vigencia`, `obligacion`, `derecho`}, que son los tipos de supuesto que generaban contradicciones en la PC1; `abstain` booleano; los tres obligatorios y `additionalProperties: false`.

- **Modelo:** `Qwen/Qwen2.5-0.5B-Instruct`, el mismo de la PC1, en CPU con `transformers`.
- **Decoding:** greedy, `max_new_tokens = 96`, para que Q1 sea determinista.
- **Prompt:** `prompts/estructurar_consulta.txt`, con INSTRUCCIÓN, REGLAS, FORMATO, CONTEXTO (marcado como información auxiliar, no instrucción) y ENTRADA separados.

Validación en tres niveles, igual que en la PC1:

1. **parseable:** se extrae un objeto JSON del texto (aunque venga envuelto en \`\`\`json);
2. **schema-valid:** cumple el JSON Schema;
3. **semánticamente adecuada:** `intent` coincide con el gold y `abstain = false`. Si la `query` conserva la necesidad no lo decide el schema: lo mide la recuperación del Experimento 1.

## 4. Modelo de embeddings

`intfloat/multilingual-e5-small` (384 dimensiones, multilingüe). E5 se entrenó con prefijos asimétricos, así que la consulta se codifica como `query: <texto>` y cada fragmento como `passage: <texto>`. Los vectores se normalizan a norma L2 = 1. El notebook mide además los tokens por fragmento, porque E5 trunca a 512 tokens: un fragmento truncado se indexaría sin su parte final.

## 5. Segmentación

`build_chunks`, la política del curso (Cuaderno 5, sección 2): agrupa passages completos de un mismo documento hasta no superar `target_words`, y el siguiente fragmento empieza `overlap_passages = 1` passage antes del final del anterior. Un passage nunca se parte ni un fragmento cruza documentos. El notebook verifica que todo passage queda en algún fragmento en las tres condiciones.

| Condición | target_words | overlap_passages | Fragmentos | Palabras (media) | Fragmentos por documento | Tokens máx. (E5) |
|---|---|---|---|---|---|---|
| small | 100 | 1 | 49 | 80.2 | 4.9 | 146 |
| baseline | 180 | 1 | 20 | 144.4 | 2.0 | 248 |
| large | 320 | 1 | 11 | 234.5 | 1.1 | 412 |

Ningún fragmento supera los 512 tokens de E5, así que no hubo truncación en ninguna condición: el Experimento 2 cambia solo la granularidad.

El enunciado de la PC2 fija small = 100 (el laboratorio de la Semana 4 usaba 80).

## 6. Índice y similitud

`faiss.IndexFlatIP`: búsqueda **exacta** por producto interno, la consulta se compara contra todos los vectores. Con vectores normalizados:

```text
cos(q, p) = (q · p) / (||q|| ||p||) = q · p      si ||q|| = ||p|| = 1
```

El notebook lo comprueba como en el Cuaderno 4: para N1, el score del índice (0.8957) coincide con el coseno calculado a mano (0.8957), ambos vectores tienen norma 1.0000, y el ordenamiento de FAISS coincide con la búsqueda exacta en NumPy. Se construye un índice por condición de segmentación; el ranking se pide completo y se desempata por `chunk_id`. Si algún fragmento supera los 512 tokens de E5, la celda falla en vez de truncar en silencio.

**Recall@k** (Cuaderno 5, sección 8) = passages gold cubiertos por los fragmentos del top-k / passages gold totales. Como las dos evidencias de cada consulta están en documentos distintos y un fragmento no cruza documentos, **Recall@1 no puede superar 0.5**. También se registra el rango del primer fragmento que cubre cada evidencia.

## 7. Experimento 1 - Consulta

**Q0** (necesidad original) **vs Q1** (`query` producida por el LLM). Fijos: corpus, segmentación baseline (`target_words = 180`, `overlap_passages = 1`), E5, normalización, `IndexFlatIP`, k.

**Hipótesis:** Q1 conserva el Recall@3 de Q0 cuando es schema-valid, porque elimina palabras conversacionales sin perder los términos normativos. **Resultado: no se cumple** en el único caso válido que cambió la recuperación (N3).

Si Q1 no es schema-valid o el modelo se abstiene, se reportan dos lecturas: **estricta** (la búsqueda no se ejecuta y cuenta como Recall = 0) y **con respaldo** (se busca con Q0). La estricta muestra el costo real de un LLM que falla el formato.

| Métrica | Q0 | Q1 estricta | Q1 con respaldo |
|---|---|---|---|
| json_parseable_rate | - | 1.00 | - |
| schema_valid_rate | - | 0.33 | - |
| intent_accuracy | - | 0.00 | - |
| mean Recall@1 | 0.50 | 0.17 | 0.50 |
| mean Recall@3 | **1.00** | 0.25 | 0.92 |
| mean Recall@5 | 1.00 | 0.33 | 1.00 |

Por consulta:

| query_id | Q1 generada | schema_valid | intent (gold) | Recall@3 Q0 | Recall@3 Q1 | Efecto |
|---|---|---|---|---|---|---|
| N1 | no válida | no | `prohibición_permiso` (plazo) | 1.0 | 0.0 | degrada (sin búsqueda) |
| N2 | no válida | no | `prohibición_permiso` (prohibicion_permiso) | 1.0 | 0.0 | degrada (sin búsqueda) |
| N3 | `apela` | sí | `vigencia` (plazo) | 1.0 | 0.5 | degrada |
| N4 | no válida | no | `prohibición_permiso` (vigencia) | 1.0 | 0.0 | degrada (sin búsqueda) |
| N5 | no válida | no | `prohibición_permiso` (obligacion) | 1.0 | 0.0 | degrada (sin búsqueda) |
| N6 | `cuántos días de vacaciones` | sí | `vigencia` (derecho) | 1.0 | 1.0 | conserva |

Lectura:

- **Las 6 salidas fueron parseables, pero solo 2 cumplieron el schema.** Las otras 4 fallaron por una tilde: el modelo escribió `prohibición_permiso` y el enum admite `prohibicion_permiso`. Para JSON Schema son cadenas distintas. Incluso en N2, donde la categoría era la correcta, la salida es inválida.
- **Ningún intent fue correcto.** El modelo colapsó a dos etiquetas (`prohibición_permiso` y `vigencia`) sin importar la pregunta, el mismo patrón que en la PC1, donde respondía casi siempre `neutral`.
- **4 de los 5 casos "degrada" no son fallos de recuperación:** son consultas que no llegaron a buscar porque el JSON no pasó el schema. Con respaldo a Q0, el Recall@3 medio queda en 0.92.
- **Efecto techo:** Q0 ya alcanza Recall@3 = 1.0 y Recall@1 = 0.5 (el máximo posible) en las 6 consultas, así que Q1 solo podía conservar o degradar.

Fuente: `results/exp1_consultas.jsonl`, `results/exp1_recuperacion.csv`, `results/resumen.json`.

**Caso analizado (N3, degrada):**

| | Consulta enviada al encoder | Rango Norma A (D05-A06) | Rango Norma B (D06-A07) | Recall@3 |
|---|---|---|---|---|
| Q0 | `query: Me notificaron una resolución y quiero apelarla, ¿cuántos días tengo para presentar la apelación?` | 1 | 2 | 1.0 |
| Q1 | `query: apela` | 4 | 1 | 0.5 |

La reescritura redujo la necesidad a una sola palabra truncada. Se perdieron "plazo", "días" y "resolución", y la Norma A (15 días hábiles) cayó del top-3. El análisis completo está en la sección 10. En N6, en cambio, la query más corta mantuvo el término clave ("días de vacaciones") y el top-3 fue idéntico al de Q0.

## 8. Experimento 2 - Chunking

`target_words` 100 / 180 / 320. Fijos: corpus, consultas **Q0** (para aislar el chunking de la variabilidad del LLM), qrels, `overlap_passages = 1`, E5, normalización L2, producto interno, `IndexFlatIP`, k. Métrica principal: **mean Recall@3**.

**Hipótesis:** al aumentar `target_words` sube Recall@3, porque con fragmentos más grandes hay menos candidatos y cada uno cubre más passages, aunque la señal de la evidencia se diluya. **Resultado: no se cumple**; el tamaño no cambió Recall@k.

| Condición | Fragmentos | mean Recall@1 | mean Recall@3 | mean Recall@5 |
|---|---|---|---|---|
| small (100) | 49 | 0.50 | **1.00** | 1.00 |
| baseline (180) | 20 | 0.50 | **1.00** | 1.00 |
| large (320) | 11 | 0.50 | **1.00** | 1.00 |

Las 6 consultas obtienen Recall@3 = 1.0 en las tres condiciones, y Recall@1 = 0.5 (el techo) en todas. Bajo este corpus, el tamaño de fragmento no cambió la recuperación.

Explicaciones posibles, no verificadas en este experimento:

- Las consultas comparten términos clave con sus artículos relevantes (apelación, vacaciones, permiso de operación, presupuesto anual), y cada documento trata un tema distinto, así que ubicar el documento correcto es fácil para E5 con cualquier granularidad.
- Con 2 evidencias por consulta, Recall@3 solo puede valer 0, 0.5 o 1: la métrica es gruesa y no registra cambios de posición dentro del top-3.
- Con 20 u 11 fragmentos, el top-3 ya cubre entre el 15 % y el 27 % del corpus.

Fuente: `results/exp2_recall.csv`, `results/exp2_por_consulta.csv`, figura `results/exp2_recall.png`.

**Caso analizado (N3, no cambia):** N3 es la consulta con el distractor más directo. El art. 3 de D05 (descargos) repite la frase "dentro de los cinco días hábiles siguientes a la notificación" de la Norma B. Aun con fragmentos de 100 palabras, donde ese artículo forma su propio fragmento y compite directamente, las dos normas quedaron en el top-3 en las tres condiciones (Recall@3 = 1.0).

## 9. Resultados

| Experimento | Variable modificada | Variable fija | Métrica | Resultado | Conclusión |
|---|---|---|---|---|---|
| Consulta | Q0 original / Q1 estructurada | corpus, target_words 180, overlap_passages 1, E5, IndexFlatIP, k | schema_valid_rate + Recall@k | schema_valid_rate = 0.33; intent_accuracy = 0.00; mean Recall@3: Q0 = 1.00, Q1 estricta = 0.25, Q1 con respaldo = 0.92 | La estructuración con Qwen 0.5B no mejoró la recuperación: falló sobre todo en validez del JSON (4 de 6) y, en el caso válido que cambió (N3), degradó Recall@3 de 1.0 a 0.5 |
| Chunking | target_words 100 / 180 / 320 | corpus, Q0, qrels, overlap_passages 1, E5, L2, IP, IndexFlatIP, k | Recall@1/3/5 | mean Recall@3 = 1.00 en las tres condiciones | Sin diferencia observable: el benchmark es demasiado fácil para detectar el efecto del tamaño |

## 10. Un error de recuperación

Es el único error de recuperación de la corrida (`results/errores.md`): N3 buscado con Q1, `target_words = 180`.

- **Consulta enviada al encoder:** `query: apela`
- **Evidencia relevante:** D05-A06 ("El plazo para interponer recurso de apelación es de quince días hábiles") y D06-A07 ("El recurso de apelación debe interponerse dentro de los cinco días hábiles siguientes a la notificación").

| Rango | Fragmento | Score | Passages | ¿Contiene evidencia? |
|---|---|---|---|---|
| 1 | D06-C002 | 0.827 | D06-A05 a A08 | sí, D06-A07 |
| 2 | D09-C001 | 0.817 | D09-A01 a A06 (gestión de salud) | no |
| 3 | D08-C001 | 0.813 | D08-A01 a A05 (transparencia) | no |
| 4 | D05-C002 | 0.812 | D05-A05 a A08 | sí, D05-A06 |
| 5 | D06-C001 | 0.811 | D06-A01 a A05 | no |

Recall@1 = 0.5, Recall@3 = 0.5, Recall@5 = 1.0.

Análisis en los términos del Cuaderno 4:

- **Granularidad:** no es la causa. Con el mismo `target_words = 180` y la consulta original, las dos normas quedan en los puestos 1 y 2.
- **Contexto incluido:** el problema está en la consulta, no en el fragmento. "apela" no contiene ni el supuesto (plazo, días) ni el acto (resolución, recurso).
- **Información irrelevante:** los puestos 2 y 3 son una directiva de salud y unos lineamientos de transparencia, sin relación con apelaciones. Ocupan el lugar de la Norma A.
- **Posición de la evidencia:** la Norma A queda cuarta, a 0.001 del tercer puesto. Los cinco scores están entre 0.811 y 0.827, un rango de 0.016: con una sola palabra, el encoder casi no distingue entre fragmentos y el orden lo deciden diferencias mínimas.

El corpus incluye además distractores pensados para provocar errores interpretables. Por ejemplo, para N3 (plazo de apelación) el art. 3 de D05 dice "dentro de los cinco días hábiles siguientes a la notificación" igual que la Norma B, pero regula los descargos, no la apelación; y para N4 (vigencia del permiso de operación) el art. 3 de D07 fija "una vigencia de dos años" para otro certificado.

## 11. Limitación

- **Corpus pequeño y sintético, qrels del propio autor.** Una consulta equivale a 16.7 puntos de Recall promedio; las diferencias de una consulta no deben leerse como tendencia.
- **El tamaño `large` es casi recuperación por documento.** Los documentos tienen entre 177 y 380 palabras, así que con `target_words = 320` hay 1.1 fragmentos por documento. Además, top-5 sobre 11 fragmentos es el 45 % del corpus, por lo que Recall@5 en `large` es poco exigente; por eso la discusión se basa en Recall@3.
- **Recall@k mide si la evidencia aparece, no si el fragmento es útil.** Un fragmento grande puede contener la evidencia junto con mucho texto irrelevante que después confundiría al clasificador NLI.
- **Un solo LLM pequeño con greedy y un solo prompt.** Los resultados de Q1 no se generalizan a otros modelos ni a sampling. El fallo por tilde sugiere dos mejoras para una siguiente iteración, que no se aplicaron para no modificar el experimento después de ver resultados: normalizar tildes antes de validar, o incluir en el prompt un ejemplo JSON con el valor exacto del enum.
- **Efecto techo.** Q0 alcanzó Recall@3 = 1.0 en todas las consultas y condiciones. Con un benchmark así, ninguno de los dos experimentos podía mostrar una mejora, solo conservar o degradar.
- **La similitud no distingue contradicción de concordancia.** En la tesis observé que los pares contradictorios suelen tener coseno más alto que los neutrales: para recuperar candidatos eso ayuda (ambos lados del conflicto comparten el supuesto), pero confirma que la decisión final no puede tomarla el recuperador sino el componente NLI.

## 12. Conclusión

Bajo este corpus, estas 6 consultas, estos qrels y `multilingual-e5-small`, se observó que la recuperación densa con la pregunta original (Q0) alcanza Recall@3 = 1.0 en todas las consultas. En el Experimento 1, cambiando solo el texto enviado al encoder, la consulta estructurada por Qwen2.5-0.5B no mejoró la recuperación: 4 de 6 JSON fallaron el schema por una tilde, ningún intent fue correcto, y en N3 la query "apela" bajó Recall@3 de 1.0 a 0.5. En el Experimento 2, cambiando solo `target_words` (100, 180, 320) con overlap y modelo fijos, Recall@3 fue 1.0 en las tres condiciones. Esto no permite concluir que el tamaño de fragmento no importe: el benchmark es pequeño y fácil, y la métrica está saturada. Tampoco se puede generalizar el resultado del LLM a modelos más grandes ni a otros prompts.

## 13. Fuentes

- Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020. arXiv:2005.11401. Marco RAG del proyecto completo, del que esta práctica implementa solo la recuperación.
- Qwen Team (2024). *Qwen2.5 Technical Report.* arXiv:2412.15115.