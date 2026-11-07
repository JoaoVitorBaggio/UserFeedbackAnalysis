# LLM gpt-oss-20b, subamostra 1000, 2026-11-06

Medição dos braços LLM zero-shot e few-shot. A fonte desta nota é a extração dos JSON `runs/llm-gptoss20b-sub1k-zero-seed0.json`, `runs/llm-gptoss20b-sub1k-zero-seed1.json`, `runs/llm-gptoss20b-sub1k-zero-seed2.json`, `runs/llm-gptoss20b-sub1k-few-seed0.json`, `runs/llm-gptoss20b-sub1k-few-seed1.json` e `runs/llm-gptoss20b-sub1k-few-seed2.json`, lidos em main `ddc9179`, e o log `~/.local/state/overnight/ufa-llm-gptoss20b-sub1k.log`. Os JSON ficam fora do git.

`official_result` está `true` nas seis corridas. `protocol_deviations` está vazio nas seis. O campo `refusal_fallback` não aparece nos seis JSON. Esta nota copia as métricas medidas. Não é o registro do artigo.

Modelo: `openai/gpt-oss-20b`. Base: `https://api.deepinfra.com/v1/openai`. Métrica de comparação: macro F1. A chamada é a decisão de 2026-11-03 emendada em 2026-11-04: fração 0,12, subamostra estratificada de 1000, `--per-class 5000 --strict-per-class`, K=3, few-shot com k=3, sementes 0, 1 e 2.

## Configuração

Os campos abaixo são iguais nas seis corridas.

| Campo | Valor |
| --- | --- |
| model / model_id | `openai/gpt-oss-20b` |
| base_url | `https://api.deepinfra.com/v1/openai` |
| dry_run / live | false / true |
| test_fraction | 0,12 (split por conversa, papel User) |
| per_class / strict | 5000 / true (`n_train` 15000, `n_val` 0) |
| test_subsample | 1000, estratificada pelos maiores restos; a semente da subamostra é a da corrida |
| inference_k / vote / tie_break | 3 / majority / `first_inference` |
| temperature | 0,7 |
| max_chars | 4000 (`n_demo_truncated` 0) |
| k | 3 no few-shot; 0 no zero-shot |
| n_tie_breaks | 0 nas seis |
| official_result | true nas seis |
| protocol_deviations | vazio nas seis |
| refusal_fallback | campo ausente nos seis JSON |
| eligible_class_counts | SAT 6726, DSAT 24252, sem sinal 477416 (soma 508394) |

Os JSON não trazem contagem de tokens, tempo ou latência por chamada, nem contadores de retry ou de parse. A nota não afirma esses números. O que a extração traz além das métricas é `n_test_truncated`, `n_tie_breaks`, `inference_votes` e `resume.n_reused`.

`n_test_truncated` conta textos da subamostra cortados em 4000 caracteres. `n_tie_breaks` 0 quer dizer que o desempate `first_inference` não foi usado. Votos unânimes são contados em 1000.

| Corrida | Semente | n_test_truncated | votos unânimes | resume.n_reused | demonstrações |
| --- | ---: | ---: | ---: | ---: | --- |
| zero-shot | 0 | 32 | 942 | 1000 | nenhuma |
| zero-shot | 1 | 25 | 945 | 1000 | nenhuma |
| zero-shot | 2 | 28 | 940 | 1000 | nenhuma |
| few-shot | 0 | 32 | 947 | 1000 | r511674, r81140, r210258 |
| few-shot | 1 | 25 | 956 | 1000 | r1012240, r819890, r87680 |
| few-shot | 2 | 28 | 945 | 786 | r802684, r375026, r148156 |

`resume.n_reused` 1000 nas cinco primeiras é arquivo já completo, regravado na retomada de 2026-11-06. No few-shot da semente 2 a retomada reaproveitou 786 e completou o resto.

## Partição registrada

`eligible_class_counts`, iguais nas seis corridas: SAT 6726, DSAT 24252, sem sinal 477416. Soma 508394. É a mesma contagem da [nota do baseline](cardiff-baseline-2026-11-02.md).

Teste oficial, fração pedida 0,12. A subamostra de 1000 sai desse teste, na mesma semente. O suporte do F1 desta nota é a subamostra, não o teste oficial. Zero-shot e few-shot da mesma semente repetem o mesmo teste e a mesma subamostra.

| Corrida | Semente | Teste oficial | SAT | DSAT | sem sinal |
| --- | ---: | ---: | ---: | ---: | ---: |
| zero-shot | 0 | 63000 | 23 | 68 | 909 |
| zero-shot | 1 | 62865 | 23 | 66 | 911 |
| zero-shot | 2 | 62920 | 23 | 67 | 910 |
| few-shot | 0 | 63000 | 23 | 68 | 909 |
| few-shot | 1 | 62865 | 23 | 66 | 911 |
| few-shot | 2 | 62920 | 23 | 67 | 910 |

## Métricas

A tabela usa quatro casas decimais, as da extração. O float cru do harness não veio nesta fonte. A nota não o reconstitui.

| Corrida | Semente | macro F1 | macro precisão | macro recall |
| --- | ---: | ---: | ---: | ---: |
| zero-shot | 0 | 0.7098 | 0.7569 | 0.6738 |
| zero-shot | 1 | 0.6834 | 0.7171 | 0.6564 |
| zero-shot | 2 | 0.7357 | 0.8009 | 0.6933 |
| few-shot | 0 | 0.6869 | 0.7062 | 0.6757 |
| few-shot | 1 | 0.7032 | 0.7617 | 0.6601 |
| few-shot | 2 | 0.7370 | 0.8395 | 0.6725 |

Precisão, recall e F1 por classe. O suporte é o da subamostra.

| Corrida | Semente | Classe | Precisão | Recall | F1 | Suporte |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| zero-shot | 0 | SAT | 0.7059 | 0.5217 | 0.6000 | 23 |
| zero-shot | 0 | DSAT | 0.6102 | 0.5294 | 0.5669 | 68 |
| zero-shot | 0 | sem sinal | 0.9545 | 0.9703 | 0.9624 | 909 |
| zero-shot | 1 | SAT | 0.6667 | 0.5217 | 0.5854 | 23 |
| zero-shot | 1 | DSAT | 0.5333 | 0.4848 | 0.5079 | 66 |
| zero-shot | 1 | sem sinal | 0.9512 | 0.9627 | 0.9569 | 911 |
| zero-shot | 2 | SAT | 0.9333 | 0.6087 | 0.7368 | 23 |
| zero-shot | 2 | DSAT | 0.5152 | 0.5075 | 0.5113 | 67 |
| zero-shot | 2 | sem sinal | 0.9543 | 0.9637 | 0.9590 | 910 |
| few-shot | 0 | SAT | 0.7222 | 0.5652 | 0.6341 | 23 |
| few-shot | 0 | DSAT | 0.4430 | 0.5147 | 0.4762 | 68 |
| few-shot | 0 | sem sinal | 0.9535 | 0.9472 | 0.9503 | 909 |
| few-shot | 1 | SAT | 0.7059 | 0.5217 | 0.6000 | 23 |
| few-shot | 1 | DSAT | 0.6275 | 0.4848 | 0.5470 | 66 |
| few-shot | 1 | sem sinal | 0.9517 | 0.9737 | 0.9626 | 911 |
| few-shot | 2 | SAT | 1.0000 | 0.6087 | 0.7568 | 23 |
| few-shot | 2 | DSAT | 0.5686 | 0.4328 | 0.4915 | 67 |
| few-shot | 2 | sem sinal | 0.9497 | 0.9758 | 0.9626 | 910 |

O macro F1 fica perto de 0,71. O F1 de sem sinal fica perto de 0,96, com suporte 909, 911 ou 910. O de SAT fica em 0,5854 a 0,7568, com suporte 23. O de DSAT fica em 0,4762 a 0,5669, com suporte 66, 67 ou 68.

## Agregado

O protocolo deixa o agregado das três sementes em aberto. A média e o desvio-padrão amostral abaixo, n=3, são só apoio ao relatório. A linha do Cardiff também é conta da extração. A [nota do baseline](cardiff-baseline-2026-11-02.md) não publica média.

| Braço | macro F1 | F1 SAT | F1 DSAT | F1 sem sinal | macro precisão | macro recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LLM zero-shot (subamostra 1000) | 0.7096 ± 0.0262 | 0.6407 ± 0.0836 | 0.5287 ± 0.0331 | 0.9594 ± 0.0028 | 0.7583 ± 0.0419 | 0.6745 ± 0.0184 |
| LLM few-shot (subamostra 1000) | 0.7090 ± 0.0255 | 0.6636 ± 0.0824 | 0.5049 ± 0.0373 | 0.9585 ± 0.0071 | 0.7691 ± 0.0669 | 0.6694 ± 0.0082 |
| Cardiff (teste completo, cerca de 63 mil) | 0.4797 ± 0.0041 | 0.2484 ± 0.0065 | 0.2843 ± 0.0079 | 0.9063 ± 0.0004 | 0.4574 ± 0.0034 | 0.5643 ± 0.0046 |

## Matrizes de confusão

Linhas são o ouro. Colunas são o predito. Ordem: SAT, DSAT, sem sinal.

### Zero-shot, semente 0

| ouro \\ predito | SAT | DSAT | sem sinal |
| --- | ---: | ---: | ---: |
| SAT | 12 | 1 | 10 |
| DSAT | 0 | 36 | 32 |
| sem sinal | 5 | 22 | 882 |

### Zero-shot, semente 1

| ouro \\ predito | SAT | DSAT | sem sinal |
| --- | ---: | ---: | ---: |
| SAT | 12 | 0 | 11 |
| DSAT | 0 | 32 | 34 |
| sem sinal | 6 | 28 | 877 |

### Zero-shot, semente 2

| ouro \\ predito | SAT | DSAT | sem sinal |
| --- | ---: | ---: | ---: |
| SAT | 14 | 0 | 9 |
| DSAT | 0 | 34 | 33 |
| sem sinal | 1 | 32 | 877 |

### Few-shot, semente 0

| ouro \\ predito | SAT | DSAT | sem sinal |
| --- | ---: | ---: | ---: |
| SAT | 13 | 1 | 9 |
| DSAT | 0 | 35 | 33 |
| sem sinal | 5 | 43 | 861 |

### Few-shot, semente 1

| ouro \\ predito | SAT | DSAT | sem sinal |
| --- | ---: | ---: | ---: |
| SAT | 12 | 0 | 11 |
| DSAT | 0 | 32 | 34 |
| sem sinal | 5 | 19 | 887 |

### Few-shot, semente 2

| ouro \\ predito | SAT | DSAT | sem sinal |
| --- | ---: | ---: | ---: |
| SAT | 14 | 0 | 9 |
| DSAT | 0 | 29 | 38 |
| sem sinal | 0 | 22 | 888 |

## Comparação com o Cardiff

O macro F1 do baseline, no teste oficial inteiro, é 0.4800, 0.4754 e 0.4836. A tabela e o float do harness estão na [nota do baseline](cardiff-baseline-2026-11-02.md). A diferença abaixo é LLM menos Cardiff, no macro F1, como na extração. Não é a subtração refeita das quatro casas impressas ao lado.

| Semente | Cardiff | zero-shot | zero-shot menos Cardiff | few-shot | few-shot menos Cardiff |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.4800 | 0.7098 | +0.2298 | 0.6869 | +0.2069 |
| 1 | 0.4754 | 0.6834 | +0.2080 | 0.7032 | +0.2277 |
| 2 | 0.4836 | 0.7357 | +0.2521 | 0.7370 | +0.2533 |

O Cardiff foi medido no teste oficial inteiro (63000, 62865 e 62920 exemplos). O LLM foi medido numa subamostra estratificada de 1000 desse mesmo teste. A mesma semente é o mesmo split. As proporções de classe se preservam. SAT tem 23 exemplos nas seis corridas. DSAT tem 68, 66 e 67, conforme a semente. O F1 dessas classes no LLM é ruidoso: na conta da extração, um acerto de SAT mexe o F1 dessa classe em cerca de 0,03 a 0,04. A diferença de macro F1, cerca de +0,23, é muito maior que a variação entre sementes.

## Linha do tempo

Em 2026-11-04 (America/Sao_Paulo) o backend destas corridas passou da Groq para a DeepInfra. A cota gratuita diária da Groq tinha esgotado, e a rota que mandaria o mesmo modelo pela OpenRouter até a Groq foi abandonada. A base medida nos seis JSON é `https://api.deepinfra.com/v1/openai`. A base `https://api.groq.com/openai/v1` não aparece nestes arquivos. O modelo continua `openai/gpt-oss-20b`. O log começa depois dessa decisão e não carimba data na abertura da primeira execução. A corrida foi lançada em 2026-11-04.

| Quando (America/Sao_Paulo) | Evento |
| --- | --- |
| 2026-11-04 | Início. Zero-shot semente 0 fechou com macro F1 0.7098. Zero-shot semente 1 fechou com 0.6834. |
| 2026-11-04, cerca de 23:15 | Pausa pedida por João. Zero-shot semente 2 parou com checkpoint em 751/1000. |
| 2026-11-06 11:36:36 | Retomada do zero-shot da semente 2, checkpoint em 751. |
| 2026-11-06, durante o dia | Zero-shot semente 2 fechou com macro F1 0.7357. Few-shot semente 0 fechou com 0.6869. Few-shot semente 1 fechou com 0.7032. |
| 2026-11-06, cerca de 17:58 | Few-shot semente 2, exemplo 786/1000, id `r993972`. O log registra `erro: Resposta do LLM sem um rótulo único: 'I’m sorry, but I can’t help with that.'. (após 8 tentativas)`. A corrida parou. O checkpoint ficou em 786. |
| 2026-11-06 | PR #12 (`ddc9179`) squash-merged. Recusa persistente passa a contar como sem sinal, entra em `protocol_deviations` e deixa `official_result` falso. |
| 2026-11-06 19:37:21 | Retomada do few-shot da semente 2 no `ddc9179`, checkpoint em 786. As outras cinco corridas já estavam em 1000 de 1000 e foram puladas (`retomada 1000 de 1000`). |
| 2026-11-06 20:11 | Few-shot semente 2 fechou. O mtime do JSON é esse horário. Macro F1 0.7370. |

Mtimes dos JSON finais, reescritos na retomada, em 2026-11-06: zero-shot semente 0 às 19:38, semente 1 às 19:39, semente 2 às 19:40, few-shot semente 0 às 19:42, semente 1 às 19:43, semente 2 às 20:11. A extração não traz campo de fuso nesses mtimes. Os marcadores da tabela estão em America/Sao_Paulo. Esses mtimes não são o relógio da primeira conclusão das cinco corridas que já estavam prontas.

### Episódio r993972

O ouro de `r993972` é sem sinal. Na retomada depois do PR #12 o modelo devolveu rótulo válido nas três inferências. Os votos foram DSAT, DSAT e DSAT. O predito foi DSAT. É erro de classe, sem recusa. O caminho que contaria a recusa persistente como sem sinal não disparou. `protocol_deviations` ficou vazio e `official_result` ficou `true`.

No zero-shot da semente 2 o mesmo id está no índice 785, predito DSAT por unanimidade. Ali também não houve recusa. O log dessa corrida mostra `exemplo 786/1000 r993972` na linha 2864. A extração trata a linha como a mesma subamostra da semente 2.

Nas seis corridas finais, `official_result` está `true`, `protocol_deviations` está vazio e `refusal_fallback` não foi gravado. O fallback do PR #12 não chegou a ser usado.
