# Baseline Cardiff, 2026-11-02

Medição do braço baseline na máquina do João (Fedora). A fonte desta nota é o resumo do harness das sementes 0, 1 e 2, datado de 2026-11-02. Os JSON com predição (`runs/baseline-cardiff-seed0.json`, `runs/baseline-cardiff-seed1.json`, `runs/baseline-cardiff-seed2.json`) ficam fora do git.

`official_result` está `false` nas três sementes. O harness grava esse flag em toda corrida. Esta nota copia as métricas medidas. Não é o registro do artigo.

Checkpoint: `cardiffnlp/twitter-roberta-base-sentiment-latest`. Métrica de comparação: macro F1. Flags desta corrida: `--hf --test-fraction 0.12 --per-class 5000 --strict-per-class`, sementes 0, 1 e 2. O `per_class` realizado foi 5000 em SAT, DSAT e sem sinal, nas três sementes.

O protocolo escrito e o padrão do código continuam em cerca de 10 mil por classe. 5000 é o tamanho desta medição. Trocar um pelo outro troca o hold-out. Estes macro F1 não são o braço com 10 mil.

## CPU

CUDA não iniciou. O torch da UFA não tem sm_61 para a GTX 1060. O runner caiu em CPU, batch 8 (`classifier.device = cpu`, `classifier.batch_size = 8`). O pico de VRAM não foi medido: a inferência não chegou na GPU.

`mtime_local` do arquivo: semente 0 em 2026-11-02T05:07:38, semente 1 em 2026-11-02T07:55:16, semente 2 em 2026-11-02T10:42:15. O pedido desta nota situa esse relógio em America/Sao_Paulo. O resumo não traz campo de fuso.

O resumo também não traz `split_unit` nem `role`. A nota não afirma esses dois parâmetros.

## Partição registrada

`eligible_class_counts`, iguais nas três sementes: SAT 6726, DSAT 24252, sem sinal 477416. Soma 508394. No código, esse campo é a contagem por classe que entra no split, depois do filtro de papel e do descarte dos dois booleanos verdadeiros, antes do hold-out.

Parcela desses 508394, divisão dos três inteiros, não métrica do harness: SAT 1,32%, DSAT 4,77%, sem sinal 93,91%. A estimativa da [definição](definicao.md) (~2,8% SAT, ~9,7% DSAT, ~87% sem sinal) não é esta contagem.

Teste estratificado, fração pedida 0,12. Tamanho realizado e suporte do F1 (o suporte de cada classe é o `test_class_counts` da mesma semente):

| Semente | Teste | SAT | DSAT | sem sinal |
| --- | ---: | ---: | ---: | ---: |
| 0 | 63000 | 1445 | 4259 | 57296 |
| 1 | 62865 | 1426 | 4149 | 57290 |
| 2 | 62920 | 1434 | 4194 | 57292 |

## Métricas

A tabela usa quatro casas decimais. O float do harness, sem arredondar, vem abaixo. Não há média das três sementes: o protocolo deixa o agregado em aberto.

| Semente | macro F1 | macro precisão | macro recall |
| --- | ---: | ---: | ---: |
| 0 | 0.4800 | 0.4577 | 0.5642 |
| 1 | 0.4754 | 0.4538 | 0.5597 |
| 2 | 0.4836 | 0.4607 | 0.5689 |

F1 por classe:

| Semente | SAT | DSAT | sem sinal |
| --- | ---: | ---: | ---: |
| 0 | 0.2535 | 0.2804 | 0.9061 |
| 1 | 0.2412 | 0.2791 | 0.9061 |
| 2 | 0.2506 | 0.2935 | 0.9068 |

Floats do harness:

| Semente | macro F1 | macro precisão | macro recall | F1 SAT | F1 DSAT | F1 sem sinal |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.4799951454468643 | 0.45773803630248117 | 0.5641593527538774 | 0.25347862531433357 | 0.2803908202681209 | 0.9061159907581383 |
| 1 | 0.47544926740932314 | 0.4538055619226203 | 0.5596668677718117 | 0.24115595416043845 | 0.27912873112827014 | 0.9060631169392608 |
| 2 | 0.48361834735827774 | 0.460652005861876 | 0.568927408245238 | 0.25062993448681337 | 0.2934547524188959 | 0.906770355169124 |

O macro F1 fica perto de 0,48 porque o F1 de SAT e o de DSAT ficam perto de 0,25–0,29 e o de sem sinal fica perto de 0,91. O suporte de sem sinal é cerca de 57 mil exemplos; o de SAT fica perto de 1,4 mil.
