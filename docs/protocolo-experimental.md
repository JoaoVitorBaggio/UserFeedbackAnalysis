# Protocolo experimental

O baseline Cardiff foi medido em 2026-11-02. Os outros braços não. Onde faltar número, o texto diz **a definir**. Estimativas de mistura e de tamanho de treino estão na [definição](definicao.md). A contagem elegível e o macro F1 dessa corrida estão na [nota do baseline](cardiff-baseline-2026-11-02.md).

## Experimento mínimo

1. Três braços no espaço `{SAT, DSAT, sem sinal}`.
2. Um harness. As quatro corridas abaixo entram nele. Em cada semente, a comparação usa o mesmo hold-out e as mesmas métricas.
3. Treino balanceado, cerca de 10 mil exemplos por classe, montado só depois de o teste estar separado. Os cerca de 10 mil de **sem sinal** são amostra aleatória do pool sem sinal, não o pool inteiro e não um recorte curado, salvo especificação posterior. A estimativa antiga desse pool era ~87%. A contagem elegível de 2026-11-02 não confirma esse número.
4. Teste na distribuição real: 12% estratificado pelas três classes, separado antes da amostra de treino. O teste não é rebalanceado.
5. Loss do braço núcleo: cross-entropy categórica de 3 classes. BCE binária fica só como ablação ou apêndice. Ver [funcao-de-loss.md](funcao-de-loss.md).

O primeiro experimento a rodar é o classificador de sentimento genérico pré-treinado (baseline): é o braço mais barato e a referência para os outros (Transformer fine-tunado, LLM zero-shot e LLM few-shot).

## Os quatro experimentos

Quatro corridas, nesta ordem, fechada na call de 31/10/2026. O enquadramento de três braços (baseline, Transformer, LLM) continua no restante do protocolo; zero-shot e few-shot não se fundem numa linha só.

1. Baseline de sentimento genérico. Roda primeiro. Dono: João Vítor Baggio.
2. Transformer fine-tunado. Dono: Nikolas Chemale.
3. LLM zero-shot (corrida separada). Dono: João Vítor Baggio.
4. LLM few-shot (corrida separada). Dono: João Vítor Baggio.

João Vítor Baggio também é dono do harness inicial, que avalia as quatro corridas.

Checkpoints fechados em 31/10/2026. O baseline tem métrica medida em [cardiff-baseline-2026-11-02.md](cardiff-baseline-2026-11-02.md). Os outros ids abaixo não são resultado.

**Baseline.** Checkpoint principal: `cardiffnlp/twitter-roberta-base-sentiment-latest`. O harness projeta a saída para `{SAT, DSAT, sem sinal}` com o mapa positivo → SAT, negativo → DSAT, neutro → sem sinal. O card é de sentimento no Twitter; o corpus é feedback sobre resposta de assistente. Esse descompasso de domínio é intencional e faz parte da pergunta de pesquisa. Alternativa de três classes, se a corrida principal pedir contraste: `j-hartmann/sentiment-roberta-large-english-3-classes` (negative, neutral, positive no card do Hub, consulta em 31/10/2026), com o mesmo mapa. Checkpoint sem neutro não entra: o mapa ficaria **a definir** e não há colapso para duas classes.

Corrida de 2026-11-02, na máquina do João (Fedora). Checkpoint principal, sementes 0, 1 e 2, `--hf`, teste 12% estratificado, `--per-class 5000 --strict-per-class`. O treino deste protocolo continua em cerca de 10 mil por classe. 5000 é o tamanho dessa medição e troca o hold-out. CUDA não subiu (torch sem sm_61 para a GTX 1060); a inferência foi em CPU, batch 8. Macro F1: 0.4800 (semente 0), 0.4754 (semente 1), 0.4836 (semente 2). Precisão e recall macro e o F1 por classe estão na [nota](cardiff-baseline-2026-11-02.md). Não há média: o agregado das três sementes continua **a definir**.

**Transformer.** Encoder: BERT-base multilíngue ou E5-base. Os dois são aceitáveis; o id usado na corrida entra no manifesto. O loop de treino é de Nikolas Chemale, na RTX 4060 8GB. A loss é a cross-entropy categórica de 3 classes.

**LLM (zero-shot e few-shot).** Modelo: Llama 3.1 8B Instruct, via Groq ou OpenRouter. O id da API entra no JSON da corrida. Cada exemplo recebe K=3 inferências; o rótulo entregue ao harness é o voto majoritário (a moda). Um placar 2–1 tem moda. Três rótulos diferentes não têm moda única: o código usa a primeira inferência entre as empatadas e grava `tie_break: first_inference`. A call de 31/10/2026 deixou esse desempate em aberto; esta é a regra implementada. Temperatura 0 não é exigida; o valor usado entra no JSON. Se `--temperature` não vier, o cliente envia 0.7 (padrão local, para as três amostras poderem divergir; não é hiperparâmetro medido). O número de demonstrações do few-shot continua **a definir**. O teste não entra no prompt.

## Harness

Um código de avaliação. As quatro corridas entram nele. Cada uma recebe o hold-out e devolve um rótulo em `{SAT, DSAT, sem sinal}`.

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

1. Separar o teste: 12% dos exemplos elegíveis, estratificado pelas três classes. Disjunto do resto. A amostra de treino só começa depois dessa separação. O teste não é rebalanceado: a estratificação preserva a mistura de classes (distribuição real).
2. Do que sobrou, montar o treino balanceado: cerca de 10 mil SAT, cerca de 10 mil DSAT, cerca de 10 mil sem sinal.
3. Os cerca de 10 mil **sem sinal** saem por amostra aleatória do pool sem sinal. Não é o pool inteiro. Não é subconjunto escolhido à mão, salvo especificação posterior. A estimativa antiga desse pool era ~87%; a contagem elegível de 2026-11-02 está na nota do baseline.
4. Validação: **a definir**, tirada do lado de treino. O teste oficial não entra em early stopping.

As três sementes do protocolo são 0, 1 e 2, nesta ordem, as mesmas nas quatro corridas. Cada semente redesenha a partição e, no Transformer, o treino. O hold-out de uma semente é o mesmo nos quatro braços. Cada semente entra no manifesto da corrida. O harness registra cada semente; agregar os três macro F1 não foi fechado.

A unidade do split (conversa ou turno) está **a definir** na inspeção do arquivo. A preferência de trabalho é não colocar turnos da mesma conversa em treino e teste, se houver chave de conversa. Se a unidade for a conversa, não cortar conversa prevalece sobre a fração exata: o tamanho realizado pode passar de 12%. A estratificação continua o alvo.

O código implementa essa fração em `--test-fraction` (padrão 0.12 quando `--test-size` não vem). Em `turn`, cada classe contribui o arredondamento half-up de `n_classe × 0.12`. Em `conversation`, a conversa inteira fica de um lado só; a chave é o rótulo majoritário dos turnos elegíveis, e empate de contagem segue SAT, DSAT, sem sinal. O teste não é rebalanceado. `--test-size` inteiro permanece para smoke, sem estratificar. Semente, fração ou contagem, regra e contagens realizadas por classe entram no JSON.

## Métricas (hold-out)

| Métrica | Nota |
| --- | --- |
| Macro F1 | Principal de comparação |
| Precisão | No agregado do harness; por classe se útil |
| Recall | No agregado do harness; por classe se útil |

Acurácia não decide a comparação: o teste é desbalanceado. Sementes do protocolo: 0, 1 e 2, nesta ordem, compartilhadas pelos quatro braços. O marco de 4/11/2026 é a corrida do baseline. O macro F1 medido em 2026-11-02 está na [nota](cardiff-baseline-2026-11-02.md).

## Divisão de tarefas

| Frente | Dono |
| --- | --- |
| Baseline de sentimento genérico | João Vítor Baggio |
| LLM zero-shot (corrida separada) | João Vítor Baggio |
| LLM few-shot (corrida separada) | João Vítor Baggio |
| Harness inicial compartilhado (precisão, recall, macro F1) | João Vítor Baggio |
| Fine-tuning / treino do Transformer (braço núcleo, só isso), RTX 4060 8GB, encoder BERT-base multilíngue ou E5-base | Nikolas Chemale |
| Slides | Grupo 16 |
| Artigo SBC (8+ páginas) | Grupo 16, depois dos slides |

## Cronograma

| Data | O que trava |
| --- | --- |
| 4/11/2026 (quarta-feira) | Meta interna de trabalho do grupo: primeiro resultado de F1, o do baseline (macro F1 no harness compartilhado). A medição de 2026-11-02 está na [nota do baseline](cardiff-baseline-2026-11-02.md). |
| 24/11/2026 | Slides da disciplina. |
| Depois dos slides | Artigo SBC, 8+ páginas. Entrega posterior. Não bloqueia o blueprint nem os slides. |

4/11/2026 é alvo de trabalho do grupo, não prazo lido de um edital neste arquivo. O macro F1 do baseline medido em 2026-11-02 está na [nota](cardiff-baseline-2026-11-02.md). Esse primeiro F1 é o do baseline no harness compartilhado.
