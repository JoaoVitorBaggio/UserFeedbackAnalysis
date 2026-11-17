# Transformer mBERT, 5000 com strict, 2026-11-07

Medição do braço Transformer. A fonte desta nota são os JSON `transformer-seed-0.json`, `transformer-seed-1.json` e `transformer-seed-2.json`, na raiz do dataset [pln-ufrgs/ufa-transformer-protocol-5000-runs](https://huggingface.co/datasets/pln-ufrgs/ufa-transformer-protocol-5000-runs). A ablação está na pasta `ablation-perclass10000/` desse mesmo dataset. Os números também estão no repositório de modelo [pln-ufrgs/ufa-wildfeedback-satisfaction-mbert](https://huggingface.co/pln-ufrgs/ufa-wildfeedback-satisfaction-mbert). Nesse repositório, os pesos da corrida oficial estão em `protocol-5000/` e os da corrida de 10000 em `ablation-perclass10000/`. Os JSON ficam fora do git.

`official_result` sai `false`. O harness grava esse flag em toda corrida que não o sobrescreve, e o `true` é regra do braço LLM. Esta nota chama de oficial a corrida do protocolo de 2026-11-07: `--per-class 5000 --strict-per-class --val-size 15000`. A corrida `--per-class 10000` sem strict é ablação. Esta nota copia as métricas medidas, na precisão em que foram registradas. O float cru do harness não entrou nesta fonte. Não é o registro do artigo.

Retreino de Nikolas Chemale em 2026-11-07. Encoder: `bert-base-multilingual-cased`. Métrica de comparação: macro F1. Flags: `--per-class 5000 --strict-per-class --val-size 15000`, teste 12% estratificado, sementes 0, 1 e 2. A avaliação é o teste completo, cerca de 63 mil exemplos.

## Configuração

| Campo | Valor |
| --- | --- |
| encoder | `bert-base-multilingual-cased` |
| test_fraction | 0,12, estratificado |
| avaliação | teste completo, cerca de 63 mil exemplos |
| per_class / strict | 5000 / true |
| val_size pedido | 15000, adotado em 2026-11-07 |
| sementes | 0, 1 e 2 |
| official_result | false no JSON; a corrida de 5000 com strict é a do protocolo |

`--val-size 15000` é o tamanho de validação do Transformer. A validação sai do lado de treino, depois do teste, e não é estratificada. O teste oficial não entra em early stopping. Val 3000 deixaria cerca de 31 SAT na validação.

A nota não copia épocas, taxa de aprendizado, batch, `max_length`, precisão e recall macro, nem matriz de confusão.

## Partição registrada

O pedido de validação é 15000 nas três sementes. O realizado foi 15000, 15001 e 15002. O pool é o de antes da amostra de treino.

| Semente | Val realizado | pool SAT | pool DSAT | pool sem sinal | treino SAT | treino DSAT | treino sem sinal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 15000 | 5119 | 19248 | 406027 | 5000 | 5000 | 5000 |
| 1 | 15001 | 5126 | 19421 | 405981 | 5000 | 5000 | 5000 |
| 2 | 15002 | 5108 | 19282 | 406082 | 5000 | 5000 | 5000 |

A semente 0 registrou o treino 5000/5000/5000. Nas sementes 1 e 2 a flag é a mesma, `--strict-per-class`, e os três pools passam de 5000. O treino realizado fica 5000/5000/5000 nas três.

Soma do pool: semente 0, 5119 + 19248 + 406027 = 430394; semente 1, 5126 + 19421 + 405981 = 430528; semente 2, 5108 + 19282 + 406082 = 430472. Pool mais validação realizada: 445394, 445529 e 445474. A contagem elegível da [nota do baseline](cardiff-baseline-2026-11-02.md) é 508394, e os testes dessa nota, nas mesmas sementes, são 63000, 62865 e 62920. 508394 menos esses testes fecha as três somas. O hold-out de cada semente é o mesmo dos outros braços. O suporte do F1 desta nota é esse teste completo. A nota não copia `test_class_counts` do JSON do Transformer.

## Métricas

Macro F1 com quatro casas. F1 por classe com três, as que entraram nesta fonte.

| Semente | macro F1 | F1 SAT | F1 DSAT | F1 sem sinal |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.5844 | 0.361 | 0.480 | 0.913 |
| 1 | 0.5414 | 0.358 | 0.393 | 0.873 |
| 2 | 0.5826 | 0.440 | 0.414 | 0.894 |

O macro F1 fica em 0,5414 a 0,5844. O F1 de sem sinal fica em 0,873 a 0,913. O de DSAT fica em 0,393 a 0,480. O de SAT fica em 0,358 a 0,440.

## Agregado

O protocolo deixa o agregado das três sementes em aberto. A média e o desvio-padrão amostral abaixo, n=3, são só apoio ao relatório. O F1 por classe entrou com três casas. A média e o desvio arredondam essa conta para quatro.

| Corrida | macro F1 | F1 SAT | F1 DSAT | F1 sem sinal |
| --- | ---: | ---: | ---: | ---: |
| 5000 com strict (oficial) | 0.5695 ± 0.0243 | 0.3863 ± 0.0465 | 0.4290 ± 0.0454 | 0.8933 ± 0.0200 |
| 10000 sem strict (ablação) | 0.6121 ± 0.0201 | 0.4763 ± 0.0193 | 0.4510 ± 0.0406 | 0.9093 ± 0.0163 |

## Ablação

Mesma validação pedida, 15000. `--per-class 10000` sem `--strict-per-class`. SAT fica no pool inteiro, cerca de 5,1 mil. DSAT e sem sinal ficam em 10000. Esta corrida não é o resultado oficial. Os pesos dela estão em `ablation-perclass10000/` no repositório do modelo.

| Semente | macro F1 | F1 SAT | F1 DSAT | F1 sem sinal |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.6349 | 0.488 | 0.490 | 0.927 |
| 1 | 0.5971 | 0.487 | 0.409 | 0.895 |
| 2 | 0.6043 | 0.454 | 0.454 | 0.906 |

O macro F1 médio sai de 0,5695 para 0,6121. A diferença é 0,0426, cerca de 4 pontos. DSAT e sem sinal passam de 5000 exemplos de treino para 10000. SAT sai do teto de 5000 e fica no pool inteiro, cerca de 5,1 mil. O F1 médio de SAT é o que mais sobe, de 0,3863 para 0,4763. DSAT sobe de 0,4290 para 0,4510. Sem sinal sobe de 0,8933 para 0,9093.

SAT é a classe mais difícil no agregado. O F1 médio de SAT fica abaixo do de DSAT, e os dois ficam abaixo do de sem sinal, na corrida oficial e na ablação. Na semente 2 da corrida oficial o F1 de SAT (0,440) passa o de DSAT (0,414). Na ablação da semente 2 os dois ficam em 0,454.

## Comparação

Três modelos. O macro F1 do LLM é o da [nota do LLM](llm-gptoss20b-sub1k-2026-11-06.md), com o desvio em três casas. O Cardiff entra como 0,480. As sementes da [nota do baseline](cardiff-baseline-2026-11-02.md) são 0.4800, 0.4754 e 0.4836. A linha de 5000 com strict é o resultado oficial do Transformer. A linha de 10000 é ablação.

| Modelo | Avaliação | macro F1 médio |
| --- | --- | ---: |
| LLM gpt-oss-20b zero-shot | subamostra estratificada de 1000 | 0.7096 ± 0.026 |
| LLM gpt-oss-20b few-shot | subamostra estratificada de 1000 | 0.7090 ± 0.026 |
| Transformer mBERT, 5000 com strict | teste completo, cerca de 63 mil | 0.5695 ± 0.0243 |
| Transformer mBERT, 10000 sem strict (ablação) | teste completo, cerca de 63 mil | 0.6121 ± 0.0201 |
| Cardiff | teste completo, cerca de 63 mil | 0.480 |

O LLM pontua uma subamostra estratificada de 1000 turnos do teste oficial. O Transformer e o Cardiff pontuam o teste inteiro, cerca de 63 mil exemplos. A mesma semente é o mesmo split. O macro F1 do LLM e o do Transformer não estão no mesmo suporte.
