# UserFeedbackAnalysis

Trabalho prático de PLN (INF01100, UFRGS 2026/2, Grupo 16: João Vítor Baggio e Nikolas Cunha Chemale).

O baseline Cardiff tem macro F1 medido em 2026-11-02, sementes 0, 1 e 2: 0.4800, 0.4754 e 0.4836. A nota está em [docs/cardiff-baseline-2026-11-02.md](docs/cardiff-baseline-2026-11-02.md). O LLM `openai/gpt-oss-20b`, zero-shot e few-shot, tem macro F1 medido em 2026-11-06 na subamostra de 1000, sementes 0, 1 e 2. Zero-shot: 0.7098, 0.6834 e 0.7357. Few-shot: 0.6869, 0.7032 e 0.7370. A nota está em [docs/llm-gptoss20b-sub1k-2026-11-06.md](docs/llm-gptoss20b-sub1k-2026-11-06.md). O Transformer mBERT (`bert-base-multilingual-cased`) tem macro F1 medido em 2026-11-07 no teste completo, sementes 0, 1 e 2: 0.5844, 0.5414 e 0.5826. A nota está em [docs/transformer-mbert-5000-strict-2026-11-07.md](docs/transformer-mbert-5000-strict-2026-11-07.md). A corrida `--per-class 10000` sem strict é ablação.

- [Problema, escopo e decisões](docs/definicao.md)
- [Protocolo experimental](docs/protocolo-experimental.md)
- [Baseline Cardiff, 2026-11-02](docs/cardiff-baseline-2026-11-02.md)
- [LLM gpt-oss-20b, subamostra 1000, 2026-11-06](docs/llm-gptoss20b-sub1k-2026-11-06.md)
- [Transformer mBERT, 5000 com strict, 2026-11-07](docs/transformer-mbert-5000-strict-2026-11-07.md)
- [Função de loss](docs/funcao-de-loss.md)
- [Referências](docs/referencias.md)

João Vítor Baggio: harness, baseline de sentimento, LLM zero-shot e LLM few-shot. Nikolas Cunha Chemale: fine-tuning do Transformer (braço núcleo).

## Layout

```
src/user_feedback_analysis/
  harness.py            avaliação compartilhada
  metrics.py            precisão, recall, macro F1
  data/                 carga e partição do sat_dsat_annotation
  baseline/             sentimento pré-treinado → três classes
  llm/                  zero-shot e few-shot
  transformer/          fine-tuning do braço núcleo
tests/fixtures/         JSONL sintético, não é o corpus
```

As quatro corridas chamam `user_feedback_analysis.harness.evaluate`. F1 contado fora desse código não entra na tabela.

## Ambiente

Python 3.10+. Os testes abaixo não baixam modelo nem dataset:

```bash
python -m pip install pytest
PYTHONPATH=src python -m pytest
```

A corrida real do baseline e a carga pelo Hub pedem o resto:

```bash
python -m pip install -r requirements.txt
```

CPU basta para a inferência do baseline. GPU é opcional nessa corrida. O treino do Transformer, de Nikolas, fica na RTX 4060 8GB; ver a seção abaixo.

## Harness

Entrada: listas alinhadas de ouro e predição em `{SAT, DSAT, sem sinal}`. Aliases `no-signal`, `sem_sinal` e variações de caixa viram `sem sinal`.

Saída: precisão, recall e F1 por classe, mais as médias macro. A média macro inclui as três classes mesmo com suporte 0; denominador vazio vira 0. A métrica de comparação é o macro F1.

```bash
python -m user_feedback_analysis.harness --gold gold.txt --pred pred.txt --output runs/harness.json
```

Um rótulo por linha, ou uma lista JSON. `runs/` fica de fora do git.

## Dados

Fonte: [microsoft/WildFeedback](https://huggingface.co/datasets/microsoft/WildFeedback), config `sat_dsat_annotation`. Licença e citação em [docs/referencias.md](docs/referencias.md). O arquivo não entra no git.

Colunas, com a grafia do card: `Role`, `Content`, `Satisfaction`, `Disatisfaction`, `UtterranceId`, `TurnId`. JSONL local também aceita `Dissatisfaction`.

Mapa de classe, o da definição:

| Classe | Regra |
| --- | --- |
| SAT | `Satisfaction` e não `Disatisfaction` |
| DSAT | `Disatisfaction` e não `Satisfaction` |
| sem sinal | os dois falsos |

Os dois verdadeiros saem da partição. Não há regra de precedência.

O card não traz chave de conversa. O loader infere `conversation_id` quando `UtterranceId` volta a 0 ou diminui, na ordem do arquivo. A unidade do split segue em aberto no protocolo. O padrão do código é `conversation`, para a conversa inferida ficar inteira de um lado do hold-out. A chave de estratificação, nesse modo, é o rótulo majoritário dos turnos elegíveis; empate de contagem segue SAT, depois DSAT, depois sem sinal. `--split-unit turn` estratifica no turno e pode vazar conversa; o manifesto avisa.

Filtro padrão: `Role == User`. É hipótese de trabalho, porque a pergunta é sobre a mensagem do usuário. O blueprint deixa o filtro em aberto. `--role any` mantém os dois papéis. `--role Agent` fica disponível.

Ordem da partição, a do protocolo:

1. Teste na distribuição dos exemplos elegíveis, separado antes do treino balanceado. `--test-fraction 0.12` é o protocolo: estratificado pelas três classes, sem rebalancear o teste. Se `--test-size` e `--test-fraction` faltarem, a fração 0.12 entra sozinha. `--test-size` continua contagem inteira sem estratificar, para smoke e fixtures. Os dois flags são exclusivos. O JSON grava fração ou contagem, semente, `stratified`, a regra e as contagens realizadas por classe (`test_class_counts`, `test_class_targets`).
2. Validação, se `--val-size` vier. No Transformer, o tamanho adotado em 2026-11-07 é 15000. A nota está em [docs/transformer-mbert-5000-strict-2026-11-07.md](docs/transformer-mbert-5000-strict-2026-11-07.md). É uma contagem inteira do que restou depois do teste, nunca do teste.
3. Treino balanceado. A regra fechada em 2026-11-01 é `--per-class 5000` com `--strict-per-class`. Sem sinal é amostra aleatória do pool que sobrou, não o pool inteiro. O pool de SAT depois do teste fica em cerca de 5,3 mil; 10000 com strict não fecha. O default numérico do CLI continua 10000, e strict só entra se a linha passar a flag. Na corrida de relatório, a flag é o que vale.
4. O protocolo usa as sementes 0, 1 e 2, nesta ordem, as mesmas nas quatro corridas. `--seed` é uma semente por invocação e entra no manifesto. Quem chama repete a corrida, uma invocação para cada inteiro.

Em `turn`, cada classe contribui `round half-up(n_classe × fração)` exemplos. Em `conversation`, a conversa inteira muda de lado: o realizado pode passar de 12%, e o manifesto registra alvo contra realizado. `--strict-per-class` faz parte da regra de 2026-11-01: a corrida falha se alguma classe ficar abaixo de 5000. Sem essa flag, o manifesto só registra o déficit, e isso não é a corrida de relatório.

`--limit` lê só um prefixo. Prefixo não é a distribuição real e não é o hold-out do experimento.

Cache local, opcional:

```bash
python -m user_feedback_analysis.data.fetch --limit 1000 --output data/smoke.jsonl
python -m user_feedback_analysis.data.fetch --output data/sat_dsat_annotation.jsonl
```

O card lista 1.062.105 linhas na config (consulta em out/2026). Isso é número do card, não contagem deste repositório. O pico de RAM da carga inteira não foi medido aqui.

O mesmo `SplitConfig` (semente, fração ou contagem de teste, `per_class`, unidade, papel) é o que alinha as quatro corridas. Trocar um desses parâmetros troca o hold-out. `--test-subsample` não entra nesse config: é flag só do LLM, aplicada depois do split, e não troca o hold-out. Baseline e Transformer não aceitam a flag.

## Baseline

Checkpoint principal: `cardiffnlp/twitter-roberta-base-sentiment-latest` (RoBERTa-base, três saídas no card: Negative, Neutral, Positive). O card é de Twitter; o corpus é feedback de assistente. Esse descompasso é intencional e faz parte da pergunta. Alternativa de três classes, não é a corrida principal: `j-hartmann/sentiment-roberta-large-english-3-classes` (negative, neutral, positive), com o mesmo mapa. Mapa do protocolo:

| Saída do modelo | Classe |
| --- | --- |
| positive | SAT |
| negative | DSAT |
| neutral | sem sinal |

Checkpoint sem neutro não roda. O mapa ficaria a definir, e o código não colapsa para duas classes. O modelo antigo `cardiffnlp/twitter-roberta-base-sentiment` (`LABEL_0/1/2`) tem exceção explícita no mesmo mapa.

O pré-processamento do card (`@menção` → `@user`, URL → `http`) vale para os ids `cardiffnlp/twitter-roberta-base-sentiment*`. O checkpoint da Hartmann não usa esse pré-processamento. A sequência é truncada em 512, o limite do modelo. `--checkpoint` (alias `--model-id`) troca o id; o padrão é o Cardiff.

Smoke, sem download e sem GPU:

```bash
python -m user_feedback_analysis.baseline.run \
  --data tests/fixtures/tiny_sat_dsat.jsonl \
  --test-size 3 --per-class 1 --seed 0 --mock \
  --output runs/baseline-smoke.json
```

`--mock` é um léxico local. O JSON marca `classifier.kind = mock-lexicon` e `official_result: false`. Não é o baseline. O JSONL de `tests/fixtures/` é sintético; uma tabela impressa nesse arquivo não é medição do WildFeedback.

Corrida com o checkpoint, CPU:

```bash
for seed in 0 1 2; do
  python -m user_feedback_analysis.baseline.run \
    --hf \
    --test-fraction 0.12 --per-class 5000 --seed "$seed" \
    --strict-per-class \
    --device cpu --batch-size 8 \
    --output "runs/baseline-seed-${seed}.json"
done
```

As sementes do protocolo são 0, 1 e 2, nesta ordem, nas quatro corridas. O laço acima é a corrida de relatório: `--per-class 5000` e `--strict-per-class`, fechados em 2026-11-01. Omitir a flag não cai nesse tamanho: o default do CLI continua 10000, e strict continua desligado. O inteiro da semente entra no JSON, junto com a fração, a regra de estratificação e as contagens por classe. Cada semente grava o próprio arquivo em `runs/`; `runs/` fica de fora do git. JSONL local troca `--hf` por `--data data/sat_dsat_annotation.jsonl`. A alternativa, na mesma partição, acrescenta `--checkpoint j-hartmann/sentiment-roberta-large-english-3-classes`.

A medição de 2026-11-02 seguiu essa regra, no mesmo checkpoint e nas mesmas sementes, com `--hf` e `--test-fraction 0.12`. Macro F1: 0.4800, 0.4754 e 0.4836. Tabela, floats e o F1 por classe: [docs/cardiff-baseline-2026-11-02.md](docs/cardiff-baseline-2026-11-02.md).

GTX 1060 6GB: nessa corrida o CUDA não iniciou. O torch da UFA não tem sm_61 para essa placa. O runner caiu em CPU, batch 8. O pico de VRAM não foi medido, porque a inferência não chegou na GPU.

## LLM

Decisão de 2026-11-03 (America/Sao_Paulo), emendada em 2026-11-04. O modelo destas corridas continua `openai/gpt-oss-20b`. Isso revoga Llama 3.1 8B Instruct, fechado na call de 31/10/2026: a conta Groq do João não lista mais esse modelo. Em 2026-11-04 João trocou o backend para a DeepInfra. A cota gratuita diária da Groq (TPD) tinha esgotado. A rota que mandaria o mesmo modelo pela OpenRouter até a Groq foi abandonada: o valor mínimo de recarga pareceu caro. Os créditos pré-pagos da DeepInfra estão cobertos. A base da Groq (`https://api.groq.com/openai/v1`) fica revogada para estas corridas, do mesmo modo que Llama tinha sido revogado. A base aceita é `https://api.deepinfra.com/v1/openai`, a API de chat compatível com a OpenAI (consulta à documentação da DeepInfra em 2026-11-04: o POST é `{base}/chat/completions`, modelo `openai/gpt-oss-20b`). Cardiff e Transformer não mudam de teste. O LLM avalia uma subamostra estratificada de 1000 turnos do teste oficial de 12%. O teste completo, cerca de 63 mil turnos na medição de 2026-11-02, com K=3 ao vivo, não fecha.

O texto classificado é o `Content` do turno do usuário, o mesmo texto do baseline. O prompt, em inglês, pede um rótulo em `{SAT, DSAT, sem sinal}`. Few-shot amostra `k` exemplos só do treino, em rodízio pelas três classes. Esse `k` foi fechado em 3 em 2026-11-03; antes estava em aberto. Não é o K=3 das inferências: K=3 é o número de inferências por exemplo. Se `--k` não vier no few-shot, a CLI usa 3. O teste, inclusive o que ficou de fora da subamostra, não entra no prompt.

`--test-subsample N` roda depois do split. O treino e o `split` do JSON continuam os do teste oficial, os mesmos dos outros braços naquela semente. A subamostra não rebalanceia. Os alvos por classe saem dos maiores restos: quota = N vezes a contagem da classe no teste oficial, dividida pelo tamanho desse teste; cada classe fica com o piso, e as vagas que sobram vão para as maiores partes fracionárias. Empate de fração segue SAT, DSAT, sem sinal. Dentro de cada classe, um `Random(seed)` próprio, independente do RNG do split, embaralha o pool e fica com o prefixo do alvo. A lista avaliada preserva a ordem do teste oficial. A unidade é o turno: uma conversa que estava inteira no teste pode entrar só em parte, e não volta ao treino. Sem a flag, o teste oficial inteiro é avaliado. Tamanho pedido, tamanho realizado, alvos, contagens e o tamanho do teste oficial entram em `test_subsample` no JSON.

Smoke, sem rede:

```bash
python -m user_feedback_analysis.llm.run \
  --data tests/fixtures/tiny_sat_dsat.jsonl \
  --shot zero --test-size 3 --per-class 1 --seed 0 --dry-run

python -m user_feedback_analysis.llm.run \
  --data tests/fixtures/tiny_sat_dsat.jsonl \
  --shot few --k 3 --test-size 3 --per-class 1 --seed 0 --dry-run
```

Chamada real. A chave não entra no repositório nem na linha de comando:

| Variável | Papel |
| --- | --- |
| `OPENAI_API_KEY` | obrigatória na chamada real. É a chave da DeepInfra. Não entra no repositório |
| `LLM_MODEL` | obrigatória. A decisão de 2026-11-03 usa `openai/gpt-oss-20b` |
| `OPENAI_BASE_URL` | na DeepInfra, `https://api.deepinfra.com/v1/openai`. Sem a variável, o cliente usa `https://api.openai.com/v1`, e isso não é a corrida fechada. A base da Groq foi revogada em 2026-11-04 |

```bash
export OPENAI_BASE_URL=https://api.deepinfra.com/v1/openai
export LLM_MODEL=openai/gpt-oss-20b
# OPENAI_API_KEY é a chave da DeepInfra. Não cole a chave aqui.

for seed in 0 1 2; do
  python -m user_feedback_analysis.llm.run \
    --hf --shot zero \
    --test-fraction 0.12 --test-subsample 1000 \
    --per-class 5000 --strict-per-class \
    --inferences 3 --seed "$seed" --live \
    --output "runs/llm-gptoss20b-sub1k-zero-seed${seed}.json"
done

for seed in 0 1 2; do
  python -m user_feedback_analysis.llm.run \
    --hf --shot few --k 3 \
    --test-fraction 0.12 --test-subsample 1000 \
    --per-class 5000 --strict-per-class \
    --inferences 3 --seed "$seed" --live \
    --output "runs/llm-gptoss20b-sub1k-few-seed${seed}.json"
done
```

A medição de 2026-11-06 seguiu essa regra, zero-shot e few-shot, sementes 0, 1 e 2. Macro F1 zero-shot: 0.7098, 0.6834 e 0.7357. Few-shot: 0.6869, 0.7032 e 0.7370. As seis corridas gravaram `official_result: true` e `protocol_deviations` vazio. Tabela, matrizes e a linha do tempo: [docs/llm-gptoss20b-sub1k-2026-11-06.md](docs/llm-gptoss20b-sub1k-2026-11-06.md).

O mesmo laço está em `scripts/run-llm-gptoss20b-sub1k.sh`. Sem argumento, usa `--hf`. Com um JSONL, troca para `--data`. O script recusa outro modelo e outra base, inclusive a base antiga da Groq, e não imprime a chave. `runs/` fica fora do git.

O cliente manda `User-Agent: Mozilla/5.0 (compatible; UserFeedbackAnalysis/llm-client)`. Esse cabeçalho foi posto porque, na Groq, a Cloudflare respondia HTTP 403 com `error code: 1010` ao urllib, e a corrida no Fedora não chegava na API. A chamada à DeepInfra leva o mesmo cabeçalho. HTTP 429, HTTP 5xx, falha transitória de rede e resposta sem rótulo único repetem no máximo 8 vezes, na mesma fila. A espera de 429 usa `try again in Xs` (ou `ms`) e o header `Retry-After` quando vêm no erro; senão, e também quando o `content` vem vazio ou sem um SAT/DSAT/sem sinal único, dobra a partir de 1s. O piso é 0,05s e o teto é 120s, então uma pista de horas (cota diária) não trava a noite: esgotadas as tentativas de rede, segue o `RuntimeError`. Esgotada a resposta sem rótulo (recusa ou `content` vazio), a inferência conta como `sem sinal` e a semente segue; o último texto fica na predição (`refusal_fallback`) e o caso entra em `protocol_deviations`. Isso é desvio: `official_result` fica `false` e a corrida continua no relatório. Decisão de João em 2026-11-06. 401, 403, URL inválida e JSON sem `choices` falham na hora. O gpt-oss na DeepInfra às vezes põe a cadeia de pensamento em `reasoning_content` e o rótulo em `content`. O cliente lê só `content`. Se `content` vem vazio, a chamada repete; o outro campo não vira rótulo. O prompt e a temperatura não mudam entre as tentativas.

Com `--output`, cada exemplo concluído substitui esse arquivo por um checkpoint (`kind: llm_checkpoint`, `next_index` no próximo exemplo, votos e predições do prefixo). A gravação é replace atômico: uma queda no meio não deixa JSON pela metade no caminho final. Rodar de novo o mesmo comando, no mesmo `--output`, continua dali e não refaz exemplo já votado. Sem `--output` não há o que retomar. O checkpoint marca `official_result: false`; não é o relatório. A linha de um exemplo com recusa persistente guarda `refusal_fallback` e o texto cru, para a retomada não perder o desvio. O envelope novo é a versão 2; a versão 1, sem esses campos, continua legível. O `version` do fingerprint não muda. Quando a semente fecha, o mesmo caminho vira o JSON final de sempre. Arquivo de outra semente, outro shot ou outro corpus no mesmo caminho é erro e não é apagado. A identidade inclui os botões da corrida e um hash de id, rótulo e texto, porque o `example_id` é só `r{índice}`. Retomar não entra em `official_result`, salvo pelo desvio de recusa que já estava gravado.

Sem `OPENAI_API_KEY`, a corrida cai em dry-run e não chama a rede. Chave presente sem `LLM_MODEL` é erro, para não gravar mock no lugar de uma chamada que parecia real. `--live` exige as duas. `--dry-run` não chama a rede mesmo com chave.

Cada exemplo avaliado recebe `--inferences` chamadas (padrão K=3). O rótulo entregue ao harness é a moda. Sem moda única, o desempate é a primeira inferência entre as empatadas (`tie_break: first_inference` no JSON). Para K=3, isso é o caso 1–1–1; um placar 2–1 fica na moda. `--temperature` é livre. O padrão do cliente é 0.7, para as K amostras poderem divergir num endpoint determinístico; não é hiperparâmetro medido e o protocolo não exige 0. K, a regra do voto, a temperatura e os votos entram no JSON.

A call de 31/10/2026 apontava Llama 3.1 8B Instruct, com uma rota de OpenRouter (`OPENAI_BASE_URL=https://openrouter.ai/api/v1`, `LLM_MODEL=meta-llama/llama-3.1-8b-instruct`) e o id Groq que o provedor servisse. A página de depreciação da Groq (consulta em 31/10/2026) lista `llama-3.1-8b-instant` como desligado em 16/08/2026 fora de contrato enterprise. Essa rota ficou para trás em 2026-11-03. Em 2026-11-04 a própria base da Groq saiu do protocolo: a cota gratuita diária tinha esgotado, e pagar o mesmo `openai/gpt-oss-20b` via OpenRouter até a Groq foi abandonado. Esta nota não é uma chamada de API.

Cada texto no prompt é cortado em `--max-chars` (padrão 4000; `0` desliga). O corte fica no JSON (`n_test_truncated`, `n_demo_truncated`) e conta os textos avaliados, já na subamostra quando a flag vem. Não é decisão do protocolo. Turnos do corpus podem ser longos; o card chega a centenas de milhares de caracteres.

O JSON grava `model_id` (o `LLM_MODEL` do ambiente, ou null), `model` (`mock` no dry-run; o id na chamada real), `dry_run`, `live` e `base_url` sem usuário, senha ou query. O dry-run marca `dry_run: true`, `live: false`, `model: mock` e `official_result: false`.

`official_result` sai `true` só na chamada real que casa com a decisão de 2026-11-03 emendada em 2026-11-04: `openai/gpt-oss-20b`, base da DeepInfra (`https://api.deepinfra.com/v1/openai`), fração 0.12, `--test-subsample 1000`, `--per-class 5000 --strict-per-class`, K=3, few-shot k=3, semente 0, 1 ou 2. A base da Groq não entra. `true` quer dizer que a chamada é esse protocolo de LLM e que nenhuma inferência caiu em recusa persistente. O arquivo continua em `runs/`, fora do git, e não é sozinho o registro do artigo. Qualquer desvio, inclusive dry-run e recusa persistente contada como sem sinal, fica `false`, e `protocol_deviations` diz o porquê. A corrida com esse desvio ainda é relatada. A regra está em `official_result_policy`. Cardiff e Transformer não usam essa regra.

## Transformer (Nikolas)

Encoder padrão: `bert-base-multilingual-cased`. A cabeça é linear sobre o `[CLS]`, 3 logits na ordem de `LABEL_TO_ID` (0 SAT, 1 DSAT, 2 sem sinal). A loss é `build_loss()`: `CrossEntropyLoss` sem peso. A decisão está em [docs/funcao-de-loss.md](docs/funcao-de-loss.md).

O loop usa `make_splits` (mesmo hold-out dos outros braços), `SatisfactionDataset` e early stopping por macro F1 na validação. `--val-size` é obrigatório; o teste oficial não entra no treino nem no early stopping. A métrica da corrida sai do mesmo `evaluate`.

Smoke, fixture sintética, CPU, sem baixar o encoder do Hub (os testes unitários usam um BERT minúsculo). A CLI abaixo baixa o mBERT se o id não estiver em cache:

```bash
python -m user_feedback_analysis.transformer.run \
  --data tests/fixtures/tiny_sat_dsat.jsonl \
  --test-size 3 --per-class 1 --val-size 3 --seed 0 \
  --device cpu --epochs 1 --batch-size 2 --max-length 32 \
  --output runs/transformer-smoke.json
```

Corrida oficial do protocolo, uma invocação por semente. Nikolas retreinou em 2026-11-07 com `--per-class 5000 --strict-per-class --val-size 15000`. O teste é 12% estratificado e a avaliação é o teste completo. `--val-size 15000` foi adotado em 2026-11-07: a validação não é estratificada, e val 3000 deixaria cerca de 31 SAT para o early stopping. Conferir `per_class_pool_before_sample` no JSON. Macro F1 das sementes 0, 1 e 2: 0.5844, 0.5414 e 0.5826. A nota está em [docs/transformer-mbert-5000-strict-2026-11-07.md](docs/transformer-mbert-5000-strict-2026-11-07.md).

```bash
for seed in 0 1 2; do
  python -m user_feedback_analysis.transformer.run \
    --hf \
    --test-fraction 0.12 --per-class 5000 --strict-per-class --val-size 15000 --seed "$seed" \
    --device cuda \
    --checkpoint-dir "runs/checkpoints/seed-${seed}" \
    --output "runs/transformer-seed-${seed}.json"
done
```

A ablação usa a mesma `--val-size 15000` e troca o treino para `--per-class 10000`, sem `--strict-per-class`. SAT fica no pool inteiro (cerca de 5,1 mil). DSAT e sem sinal ficam em 10000. Não é o resultado oficial. Checkpoints e model card: [pln-ufrgs/ufa-wildfeedback-satisfaction-mbert](https://huggingface.co/pln-ufrgs/ufa-wildfeedback-satisfaction-mbert) (pastas `seed-0`, `seed-1`, `seed-2`). Os pesos publicados são dessa ablação. Export: `user_feedback_analysis.transformer.export`. O GitHub não guarda os pesos nem o JSON em `runs/`. Os JSON da medição estão no dataset privado [pln-ufrgs/ufa-transformer-protocol-5000-runs](https://huggingface.co/datasets/pln-ufrgs/ufa-transformer-protocol-5000-runs).

`--encoder` troca o id; o usado entra no JSON. `--checkpoint-dir` grava o melhor `state_dict`. `--no-fp16` desliga AMP. Batch (padrão 16), `max_length` (padrão 128) e fp16 no CUDA são o limite da RTX 4060 8GB, não hiperparâmetro medido. Smoke local nessa placa, 1 época, batch 16, `max_length` 128, fp16: pico cerca de 3,4 GB alocados / 3,7 GB reservados. Se a placa recusar, baixe `--batch-size` ou `--max-length`, ou use `--grad-accum`.

`official_result` sai `false`. Copiar o macro F1 para o artigo pede registro do grupo.

## O que o JSON de uma corrida não é

No baseline, no Transformer e no harness solto, `official_result` sai `false`. O arquivo descreve a corrida local e inclui a lista de predições. Essa lista fica em `runs/` e fora do git. A nota [docs/cardiff-baseline-2026-11-02.md](docs/cardiff-baseline-2026-11-02.md) copia as métricas do baseline Cardiff de 2026-11-02, com semente, fração de teste, `per_class` 5000 e checkpoint. O flag dessa nota continua falso: a nota não é o registro do artigo. A revisão do dataset no Hub não entrou nesse resumo. A nota [docs/transformer-mbert-5000-strict-2026-11-07.md](docs/transformer-mbert-5000-strict-2026-11-07.md) copia as métricas do mBERT de 2026-11-07. O resultado oficial é `--per-class 5000 --strict-per-class --val-size 15000`. A corrida `--per-class 10000` sem strict é ablação. O flag do JSON continua falso.

No LLM, a decisão de 2026-11-03, emendada em 2026-11-04, aceita um arranjo como avaliação do braço. `official_result` sai `true` só na chamada real descrita na seção LLM, sem recusa persistente. Dry-run, a base da Groq, recusa persistente contada como sem sinal e qualquer outro desvio ficam `false`, com `protocol_deviations`. O JSON continua fora do git. A nota [docs/llm-gptoss20b-sub1k-2026-11-06.md](docs/llm-gptoss20b-sub1k-2026-11-06.md) copia as métricas das seis corridas de 2026-11-06. Nas seis, `official_result` está `true` e `protocol_deviations` está vazio. A nota não é o registro do artigo.
