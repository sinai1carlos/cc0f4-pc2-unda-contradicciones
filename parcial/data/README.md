# Datos del examen parcial

## Corpus

`corpus_cc_peru_v2.json`: los 2135 artículos del Código Civil peruano (Decreto Legislativo 295). Cada artículo tiene un identificador estable (`id`, por ejemplo `cc_art_183`), su número, libro, sumilla cuando la tiene y el campo `vigente`.

- Los experimentos indexan solo los 2115 artículos vigentes. Los 20 derogados (1906 a 1922, 1975, 1976 y 2064) quedan fuera.
- Siete artículos tienen `revisar_spij: true` (43, 44, 241, 274, 334, 347 y 694) porque su texto parece mezclar redacciones anteriores y posteriores al D.Leg. 1384. Se indexan como el resto, pero ninguno se usa como evidencia relevante en los qrels ni en las respuestas de referencia.
- El Código Civil es una norma pública. No se modificó el texto de ningún artículo.

## Consultas y qrels

`consultas.jsonl`: 20 consultas escritas antes de ejecutar los sistemas. Cada una tiene un `tipo`:

| tipo | consultas | qué prueba |
|---|---|---|
| directa | 12 | usa el vocabulario del Código |
| parafraseada | 5 | usa lenguaje cotidiano, sin los términos del artículo |
| contradiccion | 3 | dos artículos regulan el mismo supuesto con reglas distintas (regla general frente a excepción) |

`qrels.json`: para cada consulta, los `id` de los artículos que la responden (31 pares en total, relevancia binaria). Cada consulta lleva en `justificacion_qrels` el inciso o la frase del artículo que justifica la anotación. Los qrels se definieron leyendo el texto de los artículos, no a partir de resultados de ningún sistema. Los artículos con vocabulario parecido que no responden la consulta (por ejemplo, el art. 2002 frente al 2007, o el 220 frente al 2001) son distractores y no están en los qrels.

## Respuestas de referencia (experimento 2)

`respuestas_referencia.jsonl`: 10 de las 20 consultas, fijadas antes de generar, con una respuesta de referencia redactada a partir del texto de los artículos y la evidencia esperada. Incluye 2 controles negativos (N01 y N02) cuya respuesta no está en el Código Civil, donde lo correcto es abstenerse.

## Herramienta (experimento 3)

`feriados_peru_2026.json`: los 16 feriados nacionales de 2026. Se tomaron del calendario oficial publicado por El Comercio y Caretas, consultados el 26 de octubre de 2026; las dos listas coinciden.

`casos_herramienta.jsonl`: 13 casos con el resultado esperado de cada uno (ejecución válida, entrada inválida, error de dominio, fallo transitorio, timeout, contrato de salida y protocolo). Las fechas esperadas se calcularon a mano con las reglas del art. 183 antes de ejecutar la herramienta.
