| Experimento | Variable modificada | Variable fija | Métrica | Resultado | Conclusión |
|---|---|---|---|---|---|
| Consulta | Q0 original / Q1 estructurada | corpus, target_words=180, overlap_passages=1, E5, IndexFlatIP, k | schema_valid_rate + Recall@k | schema_valid_rate=0.333; intent_acc=0.0; mean R@3 Q0=1.0 vs Q1=0.25 (respaldo 0.917); efecto R@3: {'degrada': 5, 'conserva': 1} | (completar) |
| Chunking | target_words 100 / 180 / 320 | corpus, Q0, qrels, overlap_passages=1, E5, L2, IP, IndexFlatIP, k | Recall@1/3/5 | mean R@3: small=1.0, baseline=1.0, large=1.0 | (completar) |
