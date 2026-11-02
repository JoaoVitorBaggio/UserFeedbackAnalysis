# UserFeedbackAnalysis

Trabalho prático de PLN (INF01100, UFRGS 2026/2, Grupo 16: João Vítor Baggio e Nikolas Cunha Chemale).

O baseline Cardiff tem macro F1 medido em 2026-11-02, sementes 0, 1 e 2: 0.4800, 0.4754 e 0.4836. Transformer e LLM ainda não têm número. A nota está em [docs/cardiff-baseline-2026-11-02.md](docs/cardiff-baseline-2026-11-02.md).

- [Problema, escopo e decisões](docs/definicao.md)
- [Protocolo experimental](docs/protocolo-experimental.md)
- [Baseline Cardiff, 2026-11-02](docs/cardiff-baseline-2026-11-02.md)
- [Função de loss](docs/funcao-de-loss.md)
- [Referências](docs/referencias.md)

João Vítor Baggio: harness, baseline de sentimento, LLM zero-shot e LLM few-shot. Nikolas Cunha Chemale: fine-tuning do Transformer (braço núcleo). O código do treino fica de propósito sem loop.

## Layout

```
src/user_feedback_analysis/
  harness.py            avaliação compartilhada
  metrics.py            precisão, recall, macro F1
  data/                 carga e partição do sat_dsat_annotation
  baseline/             sentimento pré-treinado → três classes
  llm/                  zero-shot e few-shot
  transformer/          interfaces do Nikolas
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
2. Validação, se `--val-size` vier. O tamanho continua em aberto. É uma contagem inteira do que restou depois do teste, nunca do teste.
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

O mesmo `SplitConfig` (semente, fração ou contagem de teste, `per_class`, unidade, papel) é o que alinha as quatro corridas. Trocar um desses parâmetros troca o hold-out.

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

O texto classificado é o `Content` do turno do usuário, o mesmo texto do baseline. O prompt, em inglês, pede um rótulo em `{SAT, DSAT, sem sinal}`. Few-shot amostra `k` exemplos só do treino, em rodízio pelas três classes. Esse `k` (quantas demonstrações entram no prompt) continua a definir. Não é o K=3 do protocolo: K=3 é o número de inferências por exemplo. O teste não entra no prompt.

```bash
python -m user_feedback_analysis.llm.run \
  --data tests/fixtures/tiny_sat_dsat.jsonl \
  --shot zero --test-size 3 --per-class 1 --seed 0 --dry-run

python -m user_feedback_analysis.llm.run \
  --data tests/fixtures/tiny_sat_dsat.jsonl \
  --shot few --k 3 --test-size 3 --per-class 1 --seed 0 --dry-run
```

Chamada real:

| Variável | Papel |
| --- | --- |
| `OPENAI_API_KEY` | obrigatória |
| `LLM_MODEL` | obrigatória. O protocolo fecha Llama 3.1 8B Instruct, via Groq ou OpenRouter; o id da API vai nesta variável |
| `OPENAI_BASE_URL` | opcional; padrão `https://api.openai.com/v1` |

Sem `OPENAI_API_KEY`, a corrida cai em dry-run e não chama a rede. Chave presente sem `LLM_MODEL` é erro, para não gravar mock no lugar de uma chamada que parecia real. `--live` exige as duas. `--dry-run` não chama a rede mesmo com chave.

Cada exemplo do teste recebe `--inferences` chamadas (padrão K=3). O rótulo entregue ao harness é a moda. Sem moda única, o desempate é a primeira inferência entre as empatadas (`tie_break: first_inference` no JSON). Para K=3, isso é o caso 1–1–1; um placar 2–1 fica na moda. `--temperature` é livre. O padrão do cliente é 0.7, para as K amostras poderem divergir num endpoint determinístico; não é hiperparâmetro medido e o protocolo não exige 0. K, a regra do voto, a temperatura e os votos entram no JSON. `--k` do few-shot continua sendo o número de demonstrações, ainda em aberto, e não é este K.

Rota documentada, sem chave no código: `OPENAI_BASE_URL=https://openrouter.ai/api/v1` e `LLM_MODEL=meta-llama/llama-3.1-8b-instruct`. Na Groq, a base compatível é `https://api.groq.com/openai/v1`. A página de depreciação da Groq (consulta em 31/10/2026) lista o id `llama-3.1-8b-instant` como desligado em 16/08/2026 fora de contrato enterprise; o id que o provedor servir para Llama 3.1 8B Instruct vai em `LLM_MODEL`. Esta nota não é uma chamada de API.

Cada texto no prompt é cortado em `--max-chars` (padrão 4000; `0` desliga). O corte fica no JSON (`n_test_truncated`, `n_demo_truncated`). Não é decisão do protocolo. Turnos do corpus podem ser longos; o card chega a centenas de milhares de caracteres.

O dry-run marca `dry_run: true`, `model: mock` e `official_result: false`.

## Transformer (Nikolas)

Não há loop de treino. O que já dá para reutilizar:

```python
from user_feedback_analysis.data import SplitConfig, load_jsonl, make_splits
from user_feedback_analysis.harness import evaluate
from user_feedback_analysis.transformer import (
    LABEL_TO_ID,
    SatisfactionDataset,
    SatisfactionModel,
    build_loss,
    train_transformer,
)
```

- `LABEL_TO_ID`: 0 SAT, 1 DSAT, 2 sem sinal.
- `SatisfactionDataset`: texto e índice, a partir de `split.train` ou `split.val`.
- `build_loss()`: `CrossEntropyLoss` sem peso. Peso de classe é recusado. A decisão está em [docs/funcao-de-loss.md](docs/funcao-de-loss.md).
- `SatisfactionModel.forward` e `train_transformer` levantam `NotImplementedError`.

O teste oficial não entra no treino nem no early stopping. O encoder fechado é BERT-base multilíngue ou E5-base; qualquer um dos dois serve, e o id entra no manifesto. A GPU do treino é a RTX 4060 8GB de Nikolas. Batch e comprimento de sequência nesta placa continuam limite de hardware, não hiperparâmetro medido; o pico de VRAM não foi medido aqui.

O relatório final do braço passa pelo mesmo `evaluate`.

## O que o JSON de uma corrida não é

`official_result` sai `false`. O arquivo descreve a corrida local e inclui a lista de predições. Essa lista fica em `runs/` e fora do git. A nota [docs/cardiff-baseline-2026-11-02.md](docs/cardiff-baseline-2026-11-02.md) copia as métricas do baseline Cardiff de 2026-11-02, com semente, fração de teste, `per_class` 5000 e checkpoint. O flag continua falso: a nota não é o registro do artigo. A revisão do dataset no Hub não entrou nesse resumo.
