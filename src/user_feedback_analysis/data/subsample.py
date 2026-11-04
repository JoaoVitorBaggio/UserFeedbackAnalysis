"""Subamostra estratificada do teste oficial, depois do split.

``make_splits`` continua devolvendo o teste inteiro. Esta função não
refaz o treino e não devolve turno ao lado de treino. Quem avalia um
recorte (os braços LLM) chama isto em cima de ``split.test``.

A subamostra não rebalanceia. Os alvos acompanham a mistura do teste
recebido, pelo método dos maiores restos (Hamilton):

* ``quota(classe) = n * contagem_da_classe / tamanho_do_teste``, em
  ``Decimal``, para a fração não oscilar no float binário.
* Cada classe começa com o piso da quota.
* As vagas que faltam para fechar ``n`` vão para as maiores partes
  fracionárias. Empate de fração segue SAT, DSAT, sem sinal.
* Nenhuma classe recebe mais exemplos do que tem nesse teste.

Seleção: ``random.Random(seed)`` próprio, independente do RNG do split.
Para cada classe, na ordem SAT, DSAT, sem sinal, embaralha o pool na
ordem em que o teste oficial o entregou e fica com o prefixo do alvo.
O embaralhamento não depende de ``n``; ``n`` só define o tamanho do
prefixo. A lista devolvida preserva a ordem do teste oficial.

A unidade da subamostra é o turno. Se o teste oficial foi separado por
conversa, a conversa continua fora do treino. A subamostra pode deixar
de fora parte dos turnos dessa conversa.

Trocar ``n`` pode, em tese, mover uma vaga entre classes (o método de
Hamilton não é monótono de casa). A corrida fechada pede um ``n`` só.
"""

from __future__ import annotations

import decimal
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from user_feedback_analysis.labels import LABELS, Label
from user_feedback_analysis.records import Example

SUBSAMPLE_RULE = (
    "subamostra estratificada do teste oficial, depois do split, sem rebalancear. "
    "Alvos pelo método dos maiores restos: quota = n * contagem / tamanho do teste, "
    "piso, e as vagas restantes vão para as maiores partes fracionárias. "
    "Empate de fração segue SAT, DSAT, sem sinal. "
    "Dentro de cada classe, nessa ordem, Random(seed) próprio embaralha o pool "
    "e fica com o prefixo do alvo. O embaralhamento não depende de n. "
    "A lista devolvida preserva a ordem do teste oficial. "
    "O treino não muda. A unidade é o turno: uma conversa que estava inteira "
    "no teste pode ficar só em parte na subamostra, e não volta ao treino."
)

SUBSAMPLE_RNG_ORDER = (
    "Random(seed) só da subamostra, depois do split fechado: "
    "para cada classe na ordem SAT, DSAT, sem sinal, "
    "shuffle do pool da classe na ordem do teste oficial"
)


@dataclass(frozen=True)
class SubsampleResult:
    examples: list[Example]
    requested: int
    actual: int
    seed: int
    rule: str
    rng_order: str
    class_targets: dict[str, int]
    class_counts: dict[str, int]
    source_size: int
    source_class_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "applied": True,
            "scope": "stratified_subsample",
            "requested": self.requested,
            "actual": self.actual,
            "seed": self.seed,
            "rule": self.rule,
            "rng_order": self.rng_order,
            "class_targets": dict(self.class_targets),
            "class_counts": dict(self.class_counts),
            "official_test_size": self.source_size,
            "official_test_class_counts": dict(self.source_class_counts),
            "ids": [example.example_id for example in self.examples],
        }


def subsample_class_targets(counts: Mapping[str, int], n: int) -> dict[str, int]:
    """Alvos por classe para uma subamostra de tamanho ``n``.

    ``counts`` usa o nome canônico (``SAT``, ``DSAT``, ``sem sinal``).
    Classe ausente conta 0. A soma dos alvos é ``n`` quando ``1 <= n``
    e ``n`` não passa do total.
    """
    parsed = _parse_counts(counts)
    total = sum(parsed.values())
    _require_subsample_n(n, total=total)
    floors: dict[Label, int] = {}
    fracs: list[tuple[decimal.Decimal, int, Label]] = []
    for index, label in enumerate(LABELS):
        quota = (decimal.Decimal(n) * decimal.Decimal(parsed[label])) / decimal.Decimal(total)
        floor = int(quota.to_integral_value(rounding=decimal.ROUND_FLOOR))
        floors[label] = floor
        fracs.append((quota - decimal.Decimal(floor), index, label))
    remaining = n - sum(floors.values())
    fracs.sort(key=lambda item: (-item[0], item[1]))
    targets = dict(floors)
    for _, _, label in fracs:
        if remaining == 0:
            break
        if targets[label] >= parsed[label]:
            continue
        targets[label] += 1
        remaining -= 1
    if remaining != 0:
        raise RuntimeError("A subamostra não fechou os alvos por classe.")
    return {label.value: targets[label] for label in LABELS}


def subsample_stratified(
    examples: Sequence[Example],
    n: int,
    seed: int,
) -> SubsampleResult:
    """Tira ``n`` turnos de ``examples``, estratificados pela mistura deles."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("A semente da subamostra é um inteiro >= 0.")
    items = list(examples)
    ids = [example.example_id for example in items]
    if len(ids) != len(set(ids)):
        raise ValueError(
            "O teste oficial tem example_id repetido. A subamostra não escolhe nesse caso."
        )
    source_counts = _counts_from_examples(items)
    named_source = {label.value: source_counts[label] for label in LABELS}
    targets = subsample_class_targets(named_source, n)
    rng = random.Random(seed)
    pools: dict[Label, list[Example]] = {label: [] for label in LABELS}
    for example in items:
        pools[example.label].append(example)
    chosen: set[str] = set()
    for label in LABELS:
        pool = list(pools[label])
        rng.shuffle(pool)
        take = targets[label.value]
        chosen.update(example.example_id for example in pool[:take])
    selected = [example for example in items if example.example_id in chosen]
    class_counts = {
        label.value: sum(1 for example in selected if example.label is label)
        for label in LABELS
    }
    return SubsampleResult(
        examples=selected,
        requested=n,
        actual=len(selected),
        seed=seed,
        rule=SUBSAMPLE_RULE,
        rng_order=SUBSAMPLE_RNG_ORDER,
        class_targets=targets,
        class_counts=class_counts,
        source_size=len(items),
        source_class_counts=named_source,
    )


def format_subsample(result: SubsampleResult) -> str:
    counts = ", ".join(
        f"{label.value}={result.class_counts[label.value]}" for label in LABELS
    )
    return (
        f"subamostra: {result.actual} de {result.source_size} "
        f"(pedido {result.requested}; {counts}; semente {result.seed})"
    )


def absent_subsample_record(official_test_class_counts: Mapping[str, int], official_test_size: int) -> dict[str, object]:
    """JSON de quando ``--test-subsample`` não veio. O teste oficial inteiro entra."""
    return {
        "applied": False,
        "scope": "full_official_test",
        "requested": None,
        "actual": None,
        "seed": None,
        "rule": None,
        "rng_order": None,
        "class_targets": None,
        "class_counts": None,
        "official_test_size": official_test_size,
        "official_test_class_counts": dict(official_test_class_counts),
        "ids": None,
    }


def _parse_counts(counts: Mapping[str, int]) -> dict[Label, int]:
    unknown = set(counts) - {label.value for label in LABELS}
    if unknown:
        raise ValueError(
            "Classe fora do protocolo na subamostra: " + ", ".join(sorted(unknown)) + "."
        )
    parsed: dict[Label, int] = {}
    for label in LABELS:
        value = counts.get(label.value, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Contagem de {label.value} é um inteiro >= 0.")
        parsed[label] = value
    return parsed


def _counts_from_examples(examples: Sequence[Example]) -> dict[Label, int]:
    counts = {label: 0 for label in LABELS}
    for example in examples:
        if example.label not in counts:
            raise ValueError(f"Rótulo fora do protocolo na subamostra: {example.label!r}.")
        counts[example.label] += 1
    return counts


def _require_subsample_n(n: int, *, total: int) -> None:
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("test_subsample é um inteiro >= 1.")
    if total < 1:
        raise ValueError("Não há teste oficial para subamostrar.")
    if n > total:
        raise ValueError(
            f"test_subsample={n} passa do teste oficial ({total})."
        )
