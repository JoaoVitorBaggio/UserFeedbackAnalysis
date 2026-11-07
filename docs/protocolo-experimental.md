# Protocolo experimental

O baseline Cardiff foi medido em 2026-11-02. O LLM zero-shot e o few-shot foram medidos em 2026-11-06, na subamostra de 1000. O Transformer não. Onde faltar número, o texto diz **a definir**. Estimativas de mistura e de tamanho de treino estão na [definição](definicao.md). A contagem elegível e o macro F1 do baseline estão na [nota do baseline](cardiff-baseline-2026-11-02.md). O macro F1 do LLM está na [nota do LLM](llm-gptoss20b-sub1k-2026-11-06.md).

## Experimento mínimo

1. Três braços no espaço `{SAT, DSAT, sem sinal}`.
2. Um harness. As quatro corridas abaixo entram nele. Em cada semente, a comparação usa o mesmo hold-out e as mesmas métricas.
3. Treino balanceado, 5000 exemplos por classe, com `--strict-per-class`, montado só depois de o teste estar separado. João fechou esse tamanho em 2026-11-01: o pool de SAT que sobra depois do teste fica em cerca de 5,3 mil, e 10000 com strict não fecha. Os 5000 de **sem sinal** são amostra aleatória do pool sem sinal, não o pool inteiro e não um recorte curado, salvo especificação posterior. A estimativa antiga desse pool era ~87%. A contagem elegível de 2026-11-02 não confirma esse número.
4. Teste na distribuição real: 12% estratificado pelas três classes, separado antes da amostra de treino. O teste não é rebalanceado. Cardiff e Transformer medem esse teste inteiro. Os braços LLM medem uma subamostra estratificada de 1000 turnos desse mesmo teste (decisão de 2026-11-03). A subamostra não muda o hold-out nem o treino.
5. Loss do braço núcleo: cross-entropy categórica de 3 classes. BCE binária fica só como ablação ou apêndice. Ver [funcao-de-loss.md](funcao-de-loss.md).

O primeiro experimento a rodar é o classificador de sentimento genérico pré-treinado (baseline): é o braço mais barato e a referência para os outros (Transformer fine-tunado, LLM zero-shot e LLM few-shot).

## Os quatro experimentos

Quatro corridas, nesta ordem, fechada na call de 31/10/2026. O enquadramento de três braços (baseline, Transformer, LLM) continua no restante do protocolo; zero-shot e few-shot não se fundem numa linha só.

1. Baseline de sentimento genérico. Roda primeiro. Dono: João Vítor Baggio.
2. Transformer fine-tunado. Dono: Nikolas Chemale.
3. LLM zero-shot (corrida separada). Dono: João Vítor Baggio.
4. LLM few-shot (corrida separada). Dono: João Vítor Baggio.

João Vítor Baggio também é dono do harness inicial, que avalia as quatro corridas.

Checkpoints fechados em 31/10/2026. O baseline tem métrica medida em [cardiff-baseline-2026-11-02.md](cardiff-baseline-2026-11-02.md). O LLM `openai/gpt-oss-20b` tem métrica medida em [llm-gptoss20b-sub1k-2026-11-06.md](llm-gptoss20b-sub1k-2026-11-06.md). O encoder do Transformer abaixo não é resultado.

**Baseline.** Checkpoint principal: `cardiffnlp/twitter-roberta-base-sentiment-latest`. O harness projeta a saída para `{SAT, DSAT, sem sinal}` com o mapa positivo → SAT, negativo → DSAT, neutro → sem sinal. O card é de sentimento no Twitter; o corpus é feedback sobre resposta de assistente. Esse descompasso de domínio é intencional e faz parte da pergunta de pesquisa. Alternativa de três classes, se a corrida principal pedir contraste: `j-hartmann/sentiment-roberta-large-english-3-classes` (negative, neutral, positive no card do Hub, consulta em 31/10/2026), com o mesmo mapa. Checkpoint sem neutro não entra: o mapa ficaria **a definir** e não há colapso para duas classes.

Corrida de 2026-11-02, na máquina do João (Fedora). Checkpoint principal, sementes 0, 1 e 2, `--hf`, teste 12% estratificado, `--per-class 5000 --strict-per-class`. Essas flags são a regra fechada em 2026-11-01, não um recorte desta noite. CUDA não subiu (torch sem sm_61 para a GTX 1060); a inferência foi em CPU, batch 8. Macro F1: 0.4800 (semente 0), 0.4754 (semente 1), 0.4836 (semente 2). Precisão e recall macro e o F1 por classe estão na [nota](cardiff-baseline-2026-11-02.md). Não há média: o agregado das três sementes continua **a definir**.

**Transformer.** Encoder: BERT-base multilíngue ou E5-base. Os dois são aceitáveis; o id usado na corrida entra no manifesto. O loop de treino é de Nikolas Chemale, na RTX 4060 8GB. A loss é a cross-entropy categórica de 3 classes.

**LLM (zero-shot e few-shot).** O modelo continua `openai/gpt-oss-20b`, fechado em 2026-11-03 (America/Sao_Paulo) por João Vítor Baggio. Isso revoga o requisito de Llama 3.1 8B Instruct fechado na call de 31/10/2026. A conta Groq do João não lista mais Llama 3.1 8B; esse modelo não entra nestas corridas. O id vai em `LLM_MODEL` e no JSON.

Em 2026-11-04 (America/Sao_Paulo) João trocou o backend. A cota gratuita diária da Groq (TPD) tinha esgotado. A rota que mandaria o mesmo modelo pela OpenRouter até a Groq foi abandonada: o valor mínimo de recarga pareceu caro. Os créditos pré-pagos da DeepInfra estão cobertos. A base aceita passou a ser a API compatível com a OpenAI da DeepInfra, `OPENAI_BASE_URL=https://api.deepinfra.com/v1/openai` (POST `{base}/chat/completions`; consulta à documentação da DeepInfra em 2026-11-04). A base `https://api.groq.com/openai/v1` fica revogada para estas corridas, do mesmo modo que Llama tinha sido revogado. O id do modelo não muda. `OPENAI_API_KEY` é a chave da DeepInfra e fica fora do repositório.

A avaliação dos braços LLM não percorre o teste oficial inteiro. Na medição de 2026-11-02 esse teste ficou em cerca de 63 mil turnos, e K=3 ao vivo nesse volume não fecha. João fechou uma subamostra estratificada de 1000 turnos, tirada depois do split de 12%. Os alvos por classe saem dos maiores restos sobre a mistura desse teste: a subamostra não rebalanceia e não redesenha o treino. Cardiff e Transformer continuam no teste oficial completo, salvo outra nota. As sementes continuam 0, 1 e 2.

Cada exemplo avaliado recebe K=3 inferências; o rótulo entregue ao harness é o voto majoritário (a moda). Um placar 2–1 tem moda. Três rótulos diferentes não têm moda única: o código usa a primeira inferência entre as empatadas e grava `tie_break: first_inference`. A call de 31/10/2026 deixou esse desempate em aberto; esta é a regra implementada. Temperatura 0 não é exigida; o valor usado entra no JSON. Se `--temperature` não vier, o cliente envia 0.7 (padrão local, para as três amostras poderem divergir; não é hiperparâmetro medido).

Uma resposta HTTP 200 sem rótulo único repete só aquela inferência, no mesmo prompt e na mesma temperatura, dentro do teto de tentativas que o cliente já usa para HTTP 429 e falha de rede. Entram aí a recusa do modelo e o `content` vazio (o gpt-oss na DeepInfra às vezes deixa o rótulo de fora e manda a cadeia em outro campo; esse outro campo não é lido). O parser não aceita a recusa como SAT nem DSAT. A repetição não muda modelo, base, subamostra, K, k nem semente. Em 2026-11-06 (America/Sao_Paulo) João decidiu que, esgotado o teto, essa inferência conta como `sem sinal` e a semente segue. O voto majoritário usa esse rótulo como os outros. Cada caso fica na predição (`refusal_fallback` e o último texto) e em `protocol_deviations`, como recusa persistente contada como sem sinal, com o id do exemplo e o índice da inferência. Isso é desvio de protocolo: a corrida entra no relatório e `official_result` fica `false`. HTTP 429, HTTP 5xx e falha de rede, esgotadas as tentativas, continuam parando a semente.

Com `--output`, a corrida grava nesse caminho, depois de cada exemplo concluído, um checkpoint com o índice do próximo exemplo e os votos já fechados. A escrita é replace atômico. Rodar de novo a mesma semente no mesmo caminho continua dali e não manda de volta à DeepInfra o que já foi votado. Isso foi pedido em 2026-11-04 (America/Sao_Paulo), depois de uma queda no meio da semente zero jogar fora a chamada já paga. O checkpoint não é o JSON final e não liga `official_result`. A linha concluída guarda a recusa persistente, para a retomada não perder o desvio. A regra do resultado oficial olha os botões da chamada e, no JSON final, essa recusa. O envelope do checkpoint pode estar na versão 2; a versão 1 continua legível. O `version` do fingerprint não muda, senão o arquivo já pago deixaria de ser esta corrida. Sem `--output` não há retomada.

Few-shot: k=3 demonstrações, só do treino, fechado em 2026-11-03. Antes o número estava em aberto. Esse k não é o K=3 das inferências. O teste, inclusive a parte que ficou de fora da subamostra, não entra no prompt.

`official_result` no JSON do LLM sai `true` só na chamada real que casa com essa decisão, já com o backend de 2026-11-04: modelo `openai/gpt-oss-20b`, base da DeepInfra, fração de teste 0.12, `--test-subsample 1000`, `--per-class 5000 --strict-per-class`, `--inferences 3`, few-shot com k=3, semente 0, 1 ou 2. Nesse caso, `true` quer dizer que a chamada é o protocolo de LLM fechado em 2026-11-03 e emendado em 2026-11-04. A base da Groq não é mais aceita. O arquivo continua em `runs/`, fora do git, e não é sozinho o registro do artigo. Dry-run e qualquer outro arranjo ficam `false`, com a lista `protocol_deviations`. Recusa persistente contada como sem sinal é desvio mesmo quando os botões casam com o protocolo: a corrida é relatada e `official_result` fica `false`. Cardiff e Transformer seguem com `official_result` falso. A nota do baseline não usa a regra nova.

Corrida de 2026-11-06, modelo `openai/gpt-oss-20b` na DeepInfra, sementes 0, 1 e 2, zero-shot e few-shot k=3, K=3, subamostra estratificada de 1000 do teste de 12%. `official_result` está `true` nas seis, com `protocol_deviations` vazio. Macro F1 zero-shot: 0.7098 (semente 0), 0.6834 (semente 1), 0.7357 (semente 2). Few-shot: 0.6869, 0.7032 e 0.7370. Precisão, recall, F1 por classe, matrizes e a linha do tempo estão na [nota](llm-gptoss20b-sub1k-2026-11-06.md). O protocolo deixa o agregado das três sementes em aberto; a média na nota é só apoio ao relatório. O Cardiff foi medido no teste inteiro. O LLM foi medido na subamostra (23 SAT; DSAT 66, 67 ou 68). O F1 por classe do LLM é ruidoso nesse suporte.

## Harness

Um código de avaliação. As quatro corridas entram nele. Cada uma recebe o hold-out e devolve um rótulo em `{SAT, DSAT, sem sinal}`.

O harness calcula precisão, recall e **macro F1** (por classe quando o relatório pedir). F1 contado fora desse código não entra na tabela.

**Dono do harness inicial:** João Vítor Baggio. Nikolas Chemale não escreve um avaliador próprio: o checkpoint do Transformer entra como backend desse harness.

## O que fica igual entre os braços

- O mesmo teste oficial, 12% estratificado, na distribuição real, sem rebalancear. Cardiff e Transformer avaliam esse teste inteiro. O LLM avalia 1000 turnos estratificados desse mesmo teste (decisão de 2026-11-03). A subamostra não muda o hold-out nem o treino.
- O mesmo espaço de rótulo.
- As mesmas métricas, no mesmo harness.
- Nenhum braço vê o teste no treino, no few-shot ou no early stopping.

Só o braço 2 otimiza loss neste corpus. O baseline chega pré-treinado. O LLM não recebe gradiente aqui.

## Partições

Ordem fechada:

1. Separar o teste: 12% dos exemplos elegíveis, estratificado pelas três classes. Disjunto do resto. A amostra de treino só começa depois dessa separação. O teste não é rebalanceado: a estratificação preserva a mistura de classes (distribuição real).
2. Do que sobrou, montar o treino balanceado: 5000 SAT, 5000 DSAT, 5000 sem sinal, com `--strict-per-class`. João fechou esse tamanho em 2026-11-01. O pool de SAT depois do teste fica em cerca de 5,3 mil; 10000 com strict não fecha.
3. Os 5000 **sem sinal** saem por amostra aleatória do pool sem sinal. Não é o pool inteiro. Não é subconjunto escolhido à mão, salvo especificação posterior. A estimativa antiga desse pool era ~87%; a contagem elegível de 2026-11-02 está na nota do baseline.
4. Validação: **a definir**, tirada do lado de treino. O teste oficial não entra em early stopping.
5. Só nos braços LLM, e só quando `--test-subsample` vem: da lista de teste já fechada, tirar essa quantidade de turnos, estratificada, sem rebalancear. A decisão de 2026-11-03 usa 1000. Sem a flag, o código avalia o teste inteiro. Baseline e Transformer não têm essa flag.

As três sementes do protocolo são 0, 1 e 2, nesta ordem, as mesmas nas quatro corridas. Cada semente redesenha a partição e, no Transformer, o treino. O hold-out de uma semente é o mesmo nos quatro braços. O LLM não troca esse hold-out: pontua uma subamostra dele. Cada semente entra no manifesto da corrida. O harness registra cada semente; agregar os três macro F1 não foi fechado.

A unidade do split (conversa ou turno) está **a definir** na inspeção do arquivo. A preferência de trabalho é não colocar turnos da mesma conversa em treino e teste, se houver chave de conversa. Se a unidade for a conversa, não cortar conversa prevalece sobre a fração exata: o tamanho realizado pode passar de 12%. A estratificação continua o alvo.

O código implementa essa fração em `--test-fraction` (padrão 0.12 quando `--test-size` não vem). Em `turn`, cada classe contribui o arredondamento half-up de `n_classe × 0.12`. Em `conversation`, a conversa inteira fica de um lado só; a chave é o rótulo majoritário dos turnos elegíveis, e empate de contagem segue SAT, DSAT, sem sinal. O teste não é rebalanceado. `--test-size` inteiro permanece para smoke, sem estratificar. Semente, fração ou contagem, regra e contagens realizadas por classe entram no JSON.

`--test-subsample` não repete essa fração. Só existe no runner de LLM e só roda depois do split. Os alvos saem dos maiores restos sobre as contagens do teste oficial; um `Random(seed)` próprio embaralha cada classe e fica com o prefixo do alvo. O tamanho pedido, o realizado e as contagens entram em `test_subsample` no JSON da corrida. O manifesto do split continua sendo o teste oficial inteiro.

O treino do relatório é `--per-class 5000` com `--strict-per-class`. O default numérico de `--per-class` no CLI continua 10000, e strict continua desligado se a flag não vier. A flag da corrida é o que vale. Sem ela, o código não aplica a regra fechada em 2026-11-01.

## Métricas (hold-out)

| Métrica | Nota |
| --- | --- |
| Macro F1 | Principal de comparação |
| Precisão | No agregado do harness; por classe se útil |
| Recall | No agregado do harness; por classe se útil |

Acurácia não decide a comparação: o teste é desbalanceado. Sementes do protocolo: 0, 1 e 2, nesta ordem, compartilhadas pelos quatro braços. O marco de 4/11/2026 é a corrida do baseline. O macro F1 medido em 2026-11-02 está na [nota](cardiff-baseline-2026-11-02.md). O macro F1 do LLM medido em 2026-11-06 está na [nota do LLM](llm-gptoss20b-sub1k-2026-11-06.md).

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

4/11/2026 é alvo de trabalho do grupo, não prazo lido de um edital neste arquivo. O macro F1 do baseline medido em 2026-11-02 está na [nota](cardiff-baseline-2026-11-02.md). Esse primeiro F1 é o do baseline no harness compartilhado. O macro F1 do LLM medido em 2026-11-06 está na [nota do LLM](llm-gptoss20b-sub1k-2026-11-06.md).
