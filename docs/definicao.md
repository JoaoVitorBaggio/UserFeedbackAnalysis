# Problema e escopo

**Disciplina:** Processamento de Linguagem Natural (INF01100), UFRGS, 2026/2.  
**Grupo 16:** João Vítor Baggio e Nikolas Cunha Chemale.  
**Estado:** blueprint (out/2026). Nenhum experimento foi executado. Não há resultado numérico deste repositório.

## Problema

Análise de sentimento costuma dizer se um texto é positivo ou negativo. Aqui o alvo é outro: a mensagem do usuário indica satisfação, insatisfação, ou ausência de sinal sobre a resposta de um assistente de IA.

O corpus é o WildFeedback (Microsoft; Shi et al., 2024), subconjunto `sat_dsat_annotation`, com rótulos SAT e DSAT. A terceira classe de trabalho é **sem sinal**.

## Pergunta de pesquisa

Análise de sentimento detecta satisfação e insatisfação do usuário com respostas de assistente no `sat_dsat_annotation`?

A resposta deste trabalho é comparativa: três braços, o mesmo hold-out, o mesmo harness, as mesmas métricas. A loss de treino do Transformer está em [funcao-de-loss.md](funcao-de-loss.md): cross-entropy categórica com as três classes, e esse é o espaço de rótulo da avaliação inteira.

## Experimento mínimo

- Três braços no espaço `{SAT, DSAT, sem sinal}`.
- Um harness compartilhado, com precisão, recall e macro F1.
- Treino balanceado, cerca de 10 mil exemplos por classe. Teste na distribuição real.
- Métrica que decide a comparação: **macro F1**. Precisão e recall acompanham; por classe, quando ajudar a ler o erro.

O protocolo, os donos e o cronograma estão em [protocolo-experimental.md](protocolo-experimental.md).

## Dados

Fonte pública: [microsoft/WildFeedback](https://huggingface.co/datasets/microsoft/WildFeedback), config `sat_dsat_annotation`. O card descreve as anotações SAT/DSAT ao lado dos pares de preferência. O arquivo fica fora do git.

Colunas que o card expõe e que este trabalho usa (grafia do arquivo): `Role` (`User` / `Agent`), `Satisfaction` (bool), `Disatisfaction` (bool). Mapa de classe:

| Classe | Regra |
| --- | --- |
| SAT | `Satisfaction` verdadeiro e `Disatisfaction` falso |
| DSAT | `Disatisfaction` verdadeiro e `Satisfaction` falso |
| sem sinal | os dois falsos |

Os dois verdadeiros ao mesmo tempo ficam de fora do treino até haver regra. Ver pontos em aberto.

Mistura de classes — **estimativa combinada do grupo, não contagem deste repositório:** ~9,7% DSAT, ~2,8% SAT, ~87% sem sinal.

**Treino balanceado.** O treino balanceado mira cerca de 10 mil exemplos por classe (SAT, DSAT e sem sinal). Os cerca de 10 mil exemplos de **sem sinal** são uma amostra aleatória do pool grande sem sinal (~87% dos dados), não o pool inteiro e não um subconjunto curado ou escolhido à mão, salvo especificação posterior.

**Teste.** Distribuição real (desbalanceada), disjunta do treino. O tamanho está **a definir**. A separação do teste vem antes da amostra de treino, para essa amostra não entrar no hold-out.

O card declara o corpus em inglês. O caminho mínimo não traduz.

## Decisões fechadas

| Item | Escolha |
| --- | --- |
| Corpus | WildFeedback, `sat_dsat_annotation` |
| Classes | SAT, DSAT, sem sinal |
| Loss do Transformer | Cross-entropy categórica, 3 classes |
| Espaço de rótulo na avaliação | O mesmo para os três braços |
| Braços | Baseline genérico; Transformer fine-tunado (núcleo); LLM zero-shot / few-shot |
| Avaliação | Um harness, três backends |
| Métrica de comparação | Macro F1; também precisão e recall |
| Treino | ~10 mil por classe; sem sinal = amostra aleatória do pool (~87%) |
| Teste | Distribuição real, separada antes do treino |
| Harness inicial | João Vítor Baggio |
| Fine-tuning | Nikolas Chemale, só o braço núcleo |
| Primeiro F1 | 4/11/2026, meta interna do grupo |
| Slides | 24/11/2026 |

## Fora de escopo

Chatbot de produção. Alinhar um modelo com pares de preferência — o artigo do WildFeedback faz isso; este trabalho compara classificadores de sentimento. Reanotar o corpus inteiro. Traduzir o dataset.

## Pontos em aberto

1. Só mensagens com `Role == "User"` carregam rótulo? O card mostra `Role` nos dois papéis. O filtro fica aberto até a inspeção do arquivo.
2. O artigo descreve a classificação SAT/DSAT feita por GPT-4. Se for o que está neste arquivo, rotular à mão cerca de 200 exemplos e medir a concordância com o rótulo automático. Essa checagem não foi feita.

Os dois booleanos verdadeiros no mesmo turno também estão **a definir** (precedência ou descarte). Não há regra fechada.

Nomes ainda sem checkpoint, e que não são resultado: modelo do baseline, variante BERT/RoBERTa, qual LLM e o *k* do few-shot, tamanho do teste e da validação, valor da semente da amostra aleatória. A semente entra no registro quando a amostra for tirada.
