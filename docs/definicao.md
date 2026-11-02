# Problema e escopo

**Disciplina:** Processamento de Linguagem Natural (INF01100), UFRGS, 2026/2.  
**Grupo 16:** João Vítor Baggio e Nikolas Cunha Chemale.  
**Estado:** blueprint (out/2026). O baseline Cardiff tem macro F1 medido em 2026-11-02; ver [cardiff-baseline-2026-11-02.md](cardiff-baseline-2026-11-02.md). Transformer e LLM ainda não têm número.

## Problema

Análise de sentimento costuma dizer se um texto é positivo ou negativo. Aqui o alvo é outro: a mensagem do usuário indica satisfação, insatisfação, ou ausência de sinal sobre a resposta de um assistente de IA.

O corpus é o WildFeedback (Microsoft; Shi et al., 2024), subconjunto `sat_dsat_annotation`, com rótulos SAT e DSAT. A terceira classe de trabalho é **sem sinal**.

## Pergunta de pesquisa

Análise de sentimento detecta satisfação e insatisfação do usuário com respostas de assistente no `sat_dsat_annotation`?

A resposta deste trabalho é comparativa: três braços, o mesmo hold-out, o mesmo harness, as mesmas métricas. A loss de treino do Transformer está em [funcao-de-loss.md](funcao-de-loss.md): cross-entropy categórica com as três classes, e esse é o espaço de rótulo da avaliação inteira. BCE binária fica só como ablação ou apêndice.

Não há modelo aberto treinado neste `sat_dsat_annotation` com SAT, DSAT e sem sinal. O mais próximo é o feedback-detector, com outra taxonomia; a nota está em [referencias.md](referencias.md).

## Experimento mínimo

- Três braços no espaço `{SAT, DSAT, sem sinal}`.
- Um harness compartilhado, com precisão, recall e macro F1.
- Treino balanceado, cerca de 10 mil exemplos por classe. Teste: 12% estratificado, na distribuição real.
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

Mistura de classes — **estimativa combinada do grupo, não a contagem da corrida:** ~9,7% DSAT, ~2,8% SAT, ~87% sem sinal. A corrida de 2026-11-02 registrou os exemplos elegíveis do split: SAT 6726, DSAT 24252, sem sinal 477416 (soma 508394; cerca de 1,32%, 4,77% e 93,91% desses elegíveis). Essa conta não confirma a estimativa. Os inteiros estão na [nota do baseline](cardiff-baseline-2026-11-02.md).

**Treino balanceado.** O treino balanceado mira cerca de 10 mil exemplos por classe (SAT, DSAT e sem sinal). Os cerca de 10 mil exemplos de **sem sinal** são uma amostra aleatória do pool grande sem sinal, não o pool inteiro e não um subconjunto curado ou escolhido à mão, salvo especificação posterior. A parcela desse pool na corrida de 2026-11-02 está na nota do baseline, e não é a estimativa de ~87%.

**Teste.** 12% dos exemplos elegíveis, estratificado pelas três classes, disjunto do treino. A separação do teste vem antes da amostra de treino balanceado, para essa amostra não entrar no hold-out. Depois da estratificação o teste permanece na distribuição real (desbalanceada): não é rebalanceado. A unidade do split (conversa ou turno) continua em aberto. O código cobre as duas: `--test-fraction 0.12` estratifica; em conversa, a chave é a maioria dos turnos elegíveis e a conversa não é cortada.

O card declara o corpus em inglês. O caminho mínimo não traduz.

## Decisões fechadas

| Item | Escolha |
| --- | --- |
| Corpus | WildFeedback, `sat_dsat_annotation` |
| Classes | SAT, DSAT, sem sinal |
| Loss do Transformer | Cross-entropy categórica, 3 classes. BCE binária só como ablação ou apêndice; ver [funcao-de-loss.md](funcao-de-loss.md) |
| Espaço de rótulo na avaliação | O mesmo para os três braços |
| Braços | Baseline genérico; Transformer fine-tunado (núcleo); LLM zero-shot / few-shot |
| Ordem das quatro corridas | (1) baseline de sentimento genérico; (2) Transformer fine-tunado; (3) LLM zero-shot; (4) LLM few-shot |
| Avaliação | Um harness, três backends |
| Métrica de comparação | Macro F1; também precisão e recall |
| Treino | ~10 mil por classe; sem sinal = amostra aleatória do pool. A estimativa ~87% não foi confirmada pela contagem de 2026-11-02; ver a nota do baseline |
| Teste | 12% estratificado pelas três classes, separado antes da amostra de treino; o lado de teste fica na distribuição real |
| Sementes | 0, 1 e 2, nesta ordem, as mesmas nas quatro corridas |
| Baseline principal | `cardiffnlp/twitter-roberta-base-sentiment-latest`. Mapa: positive → SAT, negative → DSAT, neutral → sem sinal. O descompasso Twitter versus feedback de assistente é intencional e faz parte da pergunta |
| Baseline alternativo | `j-hartmann/sentiment-roberta-large-english-3-classes` (três classes no card: negative, neutral, positive). Mesmo mapa. Não é a corrida principal |
| LLM | Llama 3.1 8B Instruct, via Groq ou OpenRouter |
| Agregação do LLM | K=3 inferências por exemplo; voto majoritário (moda). Temperatura 0 não é exigida. Sem moda única, o código usa a primeira inferência empatada (`first_inference`) |
| Encoder do Transformer | BERT-base multilíngue ou E5-base. Os dois são aceitáveis |
| GPU do Transformer | RTX 4060 8GB. O loop de treino continua com Nikolas Chemale |
| Harness inicial | João Vítor Baggio |
| Fine-tuning | Nikolas Chemale, só o braço núcleo |
| Primeiro F1 | 4/11/2026, meta interna do grupo |
| Slides | 24/11/2026 |

Call e chat de 31/10/2026 (João Vítor Baggio) fecharam teste, sementes, baseline, LLM, agregação, encoder e GPU. O macro F1 do baseline Cardiff (2026-11-02) está na [nota](cardiff-baseline-2026-11-02.md). Transformer e LLM ainda não têm número.

## Fora de escopo

Chatbot de produção. Alinhar um modelo com pares de preferência — o artigo do WildFeedback faz isso; este trabalho compara classificadores de sentimento. Reanotar o corpus inteiro. Traduzir o dataset.

## Pontos em aberto

1. Só mensagens com `Role == "User"` carregam rótulo? O card mostra `Role` nos dois papéis. O filtro fica aberto até a inspeção do arquivo.
2. O artigo descreve a classificação SAT/DSAT feita por GPT-4. Se for o que está neste arquivo, rotular à mão cerca de 200 exemplos e medir a concordância com o rótulo automático. Essa checagem não foi feita.
3. Os dois booleanos verdadeiros no mesmo turno (precedência ou descarte). Não há regra fechada.
4. Unidade do split: conversa ou turno. A preferência de trabalho é não colocar turnos da mesma conversa em treino e teste, se houver chave de conversa. Não foi decidido. O código já cobre as duas unidades; em conversa, estratifica pela maioria e não corta o grupo.
5. Tamanho da validação, tirada do lado de treino, depois do teste. O teste oficial não entra em early stopping.
6. Número de demonstrações do few-shot (quantos exemplos entram no prompt). Isso não é o K=3 já fechado: K=3 é o número de inferências por exemplo, agregadas por voto majoritário.
7. Como resumir os três macro F1, um por semente. Cada semente entra no registro. Média ou outro agregado não foi fechado.

As três sementes do protocolo são 0, 1 e 2, nesta ordem, as mesmas nas quatro corridas. Cada semente entra no manifesto quando a amostra for tirada.
