# Protocolo experimental

Nada disto foi executado. Onde faltar número, o texto diz **a definir**. Estimativas de mistura e de tamanho de treino estão na [definição](definicao.md); não são contagem deste repositório.

## Experimento mínimo

1. Três braços no espaço `{SAT, DSAT, sem sinal}`.
2. Um harness. Três backends. A comparação usa o mesmo hold-out e as mesmas métricas.
3. Treino balanceado, cerca de 10 mil exemplos por classe, montado só depois de o teste estar separado. Os cerca de 10 mil de **sem sinal** são amostra aleatória do pool sem sinal (~87%), não o pool inteiro e não um recorte curado, salvo especificação posterior.
4. Teste na distribuição real.
5. Loss do braço núcleo: cross-entropy categórica de 3 classes. Ver [funcao-de-loss.md](funcao-de-loss.md).

O primeiro experimento a rodar é o classificador de sentimento genérico pré-treinado (baseline): é o braço mais barato e a referência para os outros dois (Transformer fine-tunado e LLM zero/few-shot).

## Os três braços

| # | Nome | O que é | Dono |
| --- | --- | --- | --- |
| 1 | Baseline | Classificador de sentimento genérico, já pré-treinado, sem treino neste corpus | João Vítor Baggio |
| 2 | Núcleo | Transformer da família BERT/RoBERTa, fine-tunado neste rótulo | Nikolas Chemale (treino / fine-tuning, só este braço) |
| 3 | Comparação | LLM em zero-shot ou few-shot, sem gradiente neste corpus | João Vítor Baggio |

O checkpoint de cada braço está **a definir**. O baseline projeta a saída nativa para `{SAT, DSAT, sem sinal}` dentro do harness. Hipótese de trabalho do mapa: positivo → SAT, negativo → DSAT, neutro → sem sinal. Se o checkpoint escolhido não tiver neutro, o mapa fica **a definir** antes da corrida. Não há colapso silencioso para duas classes.

## Harness

Um código de avaliação. Três backends. Cada backend recebe o hold-out e devolve um rótulo em `{SAT, DSAT, sem sinal}`.

O harness calcula precisão, recall e **macro F1** (por classe quando o relatório pedir). F1 contado fora desse código não entra na tabela.

**Dono do harness inicial:** João Vítor Baggio. Nikolas Chemale não escreve um avaliador próprio: o checkpoint do Transformer entra como backend desse harness.

## O que fica igual entre os braços

- O mesmo teste, na distribuição real, sem rebalancear.
- O mesmo espaço de rótulo.
- As mesmas métricas, no mesmo harness.
- Nenhum braço vê o teste no treino, no few-shot ou no early stopping.

Só o braço 2 otimiza loss neste corpus. O baseline chega pré-treinado. O LLM não recebe gradiente aqui.

## Partições

Ordem fechada:

1. Separar o teste na distribuição real. Tamanho: **a definir**. Disjunto do resto.
2. Do que sobrou, montar o treino balanceado: cerca de 10 mil SAT, cerca de 10 mil DSAT, cerca de 10 mil sem sinal.
3. Os cerca de 10 mil **sem sinal** saem por amostra aleatória do pool sem sinal (~87% dos dados). Não é o pool inteiro. Não é subconjunto escolhido à mão, salvo especificação posterior.
4. Validação: **a definir**, tirada do lado de treino. O teste oficial não entra em early stopping.

A semente da amostra aleatória fica registrada. O valor está **a definir**. A unidade do split (conversa ou turno) também está **a definir** na inspeção do arquivo; a preferência de trabalho é não colocar turnos da mesma conversa em treino e teste, se houver chave de conversa.

## Métricas (hold-out)

| Métrica | Nota |
| --- | --- |
| Macro F1 | Principal de comparação |
| Precisão | No agregado do harness; por classe se útil |
| Recall | No agregado do harness; por classe se útil |

Acurácia não decide a comparação: o teste é desbalanceado. Sementes de treino: **a definir**. O marco de 4/11/2026 é a corrida do baseline.

## Divisão de tarefas

| Frente | Dono |
| --- | --- |
| Baseline de sentimento genérico | João Vítor Baggio |
| LLM zero-shot / few-shot | João Vítor Baggio |
| Harness inicial compartilhado (precisão, recall, macro F1) | João Vítor Baggio |
| Fine-tuning / treino do Transformer (braço núcleo, só isso) | Nikolas Chemale |
| Slides | Grupo 16 |
| Artigo SBC (8+ páginas) | Grupo 16, depois dos slides |

## Cronograma

| Data | O que trava |
| --- | --- |
| 4/11/2026 (quarta-feira) | Meta interna de trabalho do grupo: primeiro resultado de F1, o do baseline (macro F1 no harness compartilhado). |
| 24/11/2026 | Slides da disciplina. |
| Depois dos slides | Artigo SBC, 8+ páginas. Entrega posterior. Não bloqueia o blueprint nem os slides. |

4/11/2026 é alvo de trabalho do grupo, não número medido e não prazo lido de um edital neste arquivo. Esse primeiro F1 é o do baseline no harness compartilhado.
