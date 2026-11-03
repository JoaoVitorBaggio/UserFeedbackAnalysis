"""Partição na ordem fechada do protocolo.

1. Separar o teste. O protocolo usa 12% estratificado (``test_fraction``).
   ``test_size`` inteiro continua disponível para smoke e fixtures, sem estratificar.
2. Do restante, opcionalmente separar validação (tamanho ainda em aberto; contagem inteira).
3. Do que sobrou, treino balanceado. A corrida de relatório pede 5000 por
   classe, com strict. O default numérico de ``per_class`` continua 10000.
4. Os exemplos de sem sinal do treino são amostra aleatória do pool, não o pool inteiro.

A semente desta invocação entra no manifesto. O protocolo pede três sementes,
as mesmas nas quatro corridas; cada chamada recebe uma. Este módulo não escolhe
os três inteiros. Few-shot e early stopping ficam de fora do teste: esta função
não coloca exemplo de teste no treino nem na validação.

Estratificação (só quando ``test_fraction`` vem preenchido):

* ``turn``: de cada classe saem ``round_half_up(n_classe * fração)`` exemplos.
  As três amostras são independentes. O teste não é rebalanceado: a mistura
  realizada acompanha a mistura elegível, com o arredondamento de cada classe.
* ``conversation``: a conversa inteira fica de um lado só (sem vazamento).
  A chave é o rótulo majoritário dos turnos elegíveis. Empate de contagem
  segue a ordem SAT, DSAT, sem sinal. Dentro de cada chave, nessa mesma ordem,
  conversas entram até o número de exemplos *daquela classe* no teste atingir
  o alvo. A última conversa pode ultrapassar o alvo. Turnos de outra classe
  que viajam com a conversa contam para o alvo delas.
"""

from __future__ import annotations

import decimal
import random
from dataclasses import dataclass, field

from user_feedback_analysis.labels import LABELS, Label, label_from_flags
from user_feedback_analysis.records import Example, LoadedCorpus

PROTOCOL_TEST_FRACTION = 0.12

SPLIT_RNG_ORDER = (
    "shuffle do teste; shuffle da validação se val_size; "
    "amostra SAT; amostra DSAT; amostra sem sinal "
    "(shuffle da classe se o pool for menor que per_class)"
)

SPLIT_FRACTION_RNG_ORDER = (
    "estratificado: para cada classe na ordem SAT, DSAT, sem sinal, "
    "shuffle do pool da classe (turno) ou das conversas dessa maioria (conversa); "
    "shuffle da validação se val_size; "
    "amostra SAT; amostra DSAT; amostra sem sinal "
    "(shuffle da classe se o pool for menor que per_class)"
)

STRATIFY_RULE_COUNT = (
    "contagem inteira, sem estratificar; smoke e fixtures. "
    "O protocolo usa test_fraction 0.12."
)

STRATIFY_RULE_TURN = (
    "turno: cada classe entra com round half-up de n_classe * fração exemplos, "
    "sorteados à parte. O teste não é rebalanceado."
)

STRATIFY_RULE_CONVERSATION = (
    "conversa, sem vazamento: a conversa inteira fica de um lado só. "
    "A chave é o rótulo majoritário dos turnos elegíveis; "
    "empate de contagem segue a ordem SAT, DSAT, sem sinal. "
    "Em cada chave, na mesma ordem, conversas entram até o teste atingir "
    "o alvo daquela classe (round half-up de n_classe * fração). "
    "A última conversa pode ultrapassar o alvo. "
    "Exemplos de outras classes que viajam com a conversa contam para o alvo delas."
)


@dataclass(frozen=True)
class SplitConfig:
    seed: int
    test_size: int | None = None
    test_fraction: float | None = None
    per_class: int = 10_000
    val_size: int | None = None
    split_unit: str = "conversation"
    role: str | None = "User"
    strict_per_class: bool = False

    def __post_init__(self) -> None:
        _require_count("seed", self.seed, minimum=0)
        has_size = self.test_size is not None
        has_fraction = self.test_fraction is not None
        if has_size == has_fraction:
            raise ValueError("Informe só um entre test_size e test_fraction.")
        if has_size:
            _require_count("test_size", self.test_size, minimum=1)
        else:
            fraction = self.test_fraction
            if (
                fraction is None
                or isinstance(fraction, bool)
                or not isinstance(fraction, (int, float))
                or not 0 < float(fraction) < 1
            ):
                raise ValueError("test_fraction é um número em (0, 1).")
        _require_count("per_class", self.per_class, minimum=1)
        if self.val_size is not None:
            _require_count("val_size", self.val_size, minimum=1)
        if self.split_unit not in {"conversation", "turn"}:
            raise ValueError("split_unit é 'conversation' ou 'turn'.")


def _require_count(name: str, value: int, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} é um inteiro >= {minimum}.")


@dataclass
class PreparedCorpus:
    examples: list[Example]
    n_rows: int
    n_dropped_role: int
    n_dropped_ambiguous: int
    n_dropped_empty: int
    n_dropped_missing_flags: int


@dataclass
class SplitManifest:
    seed: int
    test_size_requested: int | None
    test_fraction_requested: float | None
    test_target_size: int
    test_size_actual: int
    stratified: bool
    stratify_rule: str
    test_class_targets: dict[str, int] | None
    test_class_counts: dict[str, int]
    eligible_class_counts: dict[str, int]
    per_class_requested: int
    per_class_actual: dict[str, int]
    per_class_pool_before_sample: dict[str, int]
    val_size_requested: int | None
    val_size_actual: int
    split_unit: str
    role: str | None
    n_rows: int
    n_dropped_role: int
    n_dropped_ambiguous: int
    n_dropped_empty: int
    n_dropped_missing_flags: int
    n_eligible: int
    n_train: int
    n_val: int
    n_test: int
    n_unused: int
    no_signal_pool_before_sample: int
    no_signal_sampled: int
    conversation_note: str
    source: str
    subset_limit: int | None
    full_file: bool
    rng_order: str
    rng_injected: bool
    notes: list[str] = field(default_factory=list)
    train_ids: list[str] = field(default_factory=list)
    val_ids: list[str] = field(default_factory=list)
    test_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "test_size_requested": self.test_size_requested,
            "test_fraction_requested": self.test_fraction_requested,
            "test_target_size": self.test_target_size,
            "test_size_actual": self.test_size_actual,
            "stratified": self.stratified,
            "stratify_rule": self.stratify_rule,
            "test_class_targets": (
                None
                if self.test_class_targets is None
                else dict(self.test_class_targets)
            ),
            "test_class_counts": dict(self.test_class_counts),
            "eligible_class_counts": dict(self.eligible_class_counts),
            "per_class_requested": self.per_class_requested,
            "per_class_actual": self.per_class_actual,
            "per_class_pool_before_sample": self.per_class_pool_before_sample,
            "val_size_requested": self.val_size_requested,
            "val_size_actual": self.val_size_actual,
            "split_unit": self.split_unit,
            "role": self.role,
            "n_rows": self.n_rows,
            "n_dropped_role": self.n_dropped_role,
            "n_dropped_ambiguous": self.n_dropped_ambiguous,
            "n_dropped_empty": self.n_dropped_empty,
            "n_dropped_missing_flags": self.n_dropped_missing_flags,
            "n_eligible": self.n_eligible,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_test": self.n_test,
            "n_unused": self.n_unused,
            "no_signal_pool_before_sample": self.no_signal_pool_before_sample,
            "no_signal_sampled": self.no_signal_sampled,
            "conversation_note": self.conversation_note,
            "source": self.source,
            "subset_limit": self.subset_limit,
            "full_file": self.full_file,
            "rng_order": self.rng_order,
            "rng_injected": self.rng_injected,
            "notes": list(self.notes),
            "train_ids": list(self.train_ids),
            "val_ids": list(self.val_ids),
            "test_ids": list(self.test_ids),
            "label_set": [label.value for label in LABELS],
        }


@dataclass
class DatasetSplit:
    train: list[Example]
    val: list[Example]
    test: list[Example]
    unused: list[Example]
    manifest: SplitManifest


def prepare_examples(
    corpus: LoadedCorpus,
    *,
    role: str | None,
) -> PreparedCorpus:
    """Filtra texto vazio, papel e o par de booleanos verdadeiro.

    O par verdadeiro sai do treino e da avaliação: a definição não fecha
    precedência. ``role`` None mantém User e Agent.
    """
    role_key = role.casefold() if role is not None else None
    examples: list[Example] = []
    dropped_role = 0
    dropped_ambiguous = 0
    dropped_empty = 0
    dropped_missing = 0
    for row in corpus.rows:
        if row.content is None or not row.content.strip():
            dropped_empty += 1
            continue
        if role_key is not None and row.role.casefold() != role_key:
            dropped_role += 1
            continue
        if row.satisfaction is None or row.dissatisfaction is None:
            dropped_missing += 1
            continue
        label = label_from_flags(row.satisfaction, row.dissatisfaction)
        if label is None:
            dropped_ambiguous += 1
            continue
        examples.append(
            Example(
                example_id=f"r{row.source_index}",
                conversation_id=row.conversation_id,
                text=row.content,
                label=label,
                role=row.role,
                source_index=row.source_index,
                turn_id=row.turn_id,
                utterance_id=row.utterance_id,
                state=row.state,
                domain=row.domain,
            )
        )
    return PreparedCorpus(
        examples=examples,
        n_rows=len(corpus.rows),
        n_dropped_role=dropped_role,
        n_dropped_ambiguous=dropped_ambiguous,
        n_dropped_empty=dropped_empty,
        n_dropped_missing_flags=dropped_missing,
    )


def fraction_count(n: int, fraction: float) -> int:
    """Quantos exemplos de uma classe o teste pede.

    Arredondamento half-up (0,5 sobe), em ``Decimal``, para ``fração * n``
    não oscilar no float binário. ``fraction`` fica em (0, 1).
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError("n é um inteiro >= 0.")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
        raise ValueError("test_fraction é um número em (0, 1).")
    if not 0 < float(fraction) < 1:
        raise ValueError("test_fraction é um número em (0, 1).")
    product = decimal.Decimal(n) * decimal.Decimal(str(fraction))
    return int(product.to_integral_value(rounding=decimal.ROUND_HALF_UP))


def conversation_majority(examples: list[Example]) -> Label:
    """Chave de estratificação da conversa.

    Conta os rótulos dos turnos elegíveis. A maioria vence. Se duas ou três
    classes empatam no máximo, vence a primeira em ``LABELS``: SAT, depois
    DSAT, depois sem sinal. A conversa inteira usa essa chave e não é
    cortada entre teste e o restante.
    """
    if not examples:
        raise ValueError("Conversa sem turnos elegíveis.")
    counts = {label: 0 for label in LABELS}
    for example in examples:
        counts[example.label] += 1
    top = max(counts.values())
    for label in LABELS:
        if counts[label] == top:
            return label
    raise RuntimeError("Maioria sem candidato.")


def _class_counts(examples: list[Example]) -> dict[str, int]:
    counts = {label.value: 0 for label in LABELS}
    for example in examples:
        counts[example.label.value] += 1
    return counts


def _class_targets(examples: list[Example], fraction: float) -> dict[Label, int]:
    totals = {label: 0 for label in LABELS}
    for example in examples:
        totals[example.label] += 1
    return {label: fraction_count(count, fraction) for label, count in totals.items()}


def take_stratified_holdout(
    examples: list[Example],
    fraction: float,
    rng: random.Random,
    split_unit: str,
) -> tuple[list[Example], list[Example], dict[str, int], str]:
    """Separa o teste estratificado antes de qualquer amostra de treino.

    O lado de teste não é rebalanceado. Em ``conversation`` o tamanho
    realizado pode passar da soma dos alvos para não cortar conversa.
    """
    if not examples:
        raise ValueError("Não há exemplos elegíveis para partir.")
    if split_unit not in {"conversation", "turn"}:
        raise ValueError("split_unit é 'conversation' ou 'turn'.")
    targets = _class_targets(examples, fraction)
    if sum(targets.values()) < 1:
        raise ValueError(
            "A fração não coloca nenhum exemplo no teste. "
            "Neste arquivo pequeno use test_size, ou aumente a fração."
        )
    if split_unit == "turn":
        holdout, rest = _stratify_turns(examples, targets, rng)
        rule = STRATIFY_RULE_TURN
    else:
        holdout, rest = _stratify_conversations(examples, targets, rng)
        rule = STRATIFY_RULE_CONVERSATION
    named = {label.value: targets[label] for label in LABELS}
    return holdout, rest, named, rule


def _stratify_turns(
    examples: list[Example],
    targets: dict[Label, int],
    rng: random.Random,
) -> tuple[list[Example], list[Example]]:
    pools: dict[Label, list[Example]] = {label: [] for label in LABELS}
    for example in examples:
        pools[example.label].append(example)
    for label in LABELS:
        rng.shuffle(pools[label])
        target = targets[label]
        if target > len(pools[label]):
            raise ValueError(
                f"Alvo {target} da classe {label.value} passa do pool {len(pools[label])}."
            )
    chosen_ids: set[str] = set()
    holdout: list[Example] = []
    for label in LABELS:
        taken = pools[label][: targets[label]]
        holdout.extend(taken)
        chosen_ids.update(example.example_id for example in taken)
    rest = [example for example in examples if example.example_id not in chosen_ids]
    return holdout, rest


def _stratify_conversations(
    examples: list[Example],
    targets: dict[Label, int],
    rng: random.Random,
) -> tuple[list[Example], list[Example]]:
    groups: dict[int, list[Example]] = {}
    for example in examples:
        groups.setdefault(example.conversation_id, []).append(example)
    by_key: dict[Label, list[int]] = {label: [] for label in LABELS}
    for conversation_id, members in groups.items():
        by_key[conversation_majority(members)].append(conversation_id)
    for label in LABELS:
        rng.shuffle(by_key[label])
    held = {label: 0 for label in LABELS}
    hold_ids: list[int] = []
    for label in LABELS:
        target = targets[label]
        if target <= 0:
            continue
        for conversation_id in by_key[label]:
            if held[label] >= target:
                break
            hold_ids.append(conversation_id)
            for example in groups[conversation_id]:
                held[example.label] += 1
    if not hold_ids:
        raise ValueError(
            "A estratificação por conversa não selecionou nenhum grupo. "
            "A chave é o rótulo majoritário dos turnos elegíveis."
        )
    hold_set = set(hold_ids)
    holdout: list[Example] = []
    for conversation_id in hold_ids:
        holdout.extend(groups[conversation_id])
    rest: list[Example] = []
    for conversation_id, members in groups.items():
        if conversation_id not in hold_set:
            rest.extend(members)
    return holdout, rest


def take_holdout(
    examples: list[Example],
    n: int,
    rng: random.Random,
    split_unit: str,
) -> tuple[list[Example], list[Example]]:
    """Tira o hold-out antes de qualquer amostra de treino.

    Em ``conversation``, a conversa inteira vai para um lado só. O tamanho
    realizado pode passar de ``n`` para não cortar a conversa.
    """
    if n < 1:
        raise ValueError("O tamanho do hold-out é um inteiro >= 1.")
    if not examples:
        raise ValueError("Não há exemplos elegíveis para partir.")
    if split_unit == "turn":
        items = list(examples)
        rng.shuffle(items)
        if n > len(items):
            raise ValueError(
                f"test_size={n} é maior que os {len(items)} exemplos elegíveis."
            )
        return items[:n], items[n:]
    if split_unit != "conversation":
        raise ValueError("split_unit é 'conversation' ou 'turn'.")

    groups: dict[int, list[Example]] = {}
    for example in examples:
        groups.setdefault(example.conversation_id, []).append(example)
    conversation_ids = list(groups)
    rng.shuffle(conversation_ids)
    hold_ids: list[int] = []
    count = 0
    for conversation_id in conversation_ids:
        if count >= n:
            break
        hold_ids.append(conversation_id)
        count += len(groups[conversation_id])
    if count < n:
        raise ValueError(
            f"Hold-out pedido de {n} exemplos, mas só há {count} "
            f"na unidade {split_unit}."
        )
    hold_set = set(hold_ids)
    holdout: list[Example] = []
    rest: list[Example] = []
    for conversation_id in conversation_ids:
        target = holdout if conversation_id in hold_set else rest
        target.extend(groups[conversation_id])
    return holdout, rest


def _sample_label(
    pool: list[Example],
    label: Label,
    k: int,
    rng: random.Random,
) -> tuple[list[Example], int]:
    items = [example for example in pool if example.label == label]
    if not items:
        raise ValueError(
            f"A classe {label.value} ficou sem exemplos depois do hold-out. "
            "O treino balanceado precisa das três classes."
        )
    if len(items) <= k:
        taken = list(items)
        rng.shuffle(taken)
        return taken, len(items)
    return rng.sample(items, k), len(items)


def make_splits(
    corpus: LoadedCorpus,
    config: SplitConfig,
    *,
    rng: random.Random | None = None,
) -> DatasetSplit:
    prepared = prepare_examples(corpus, role=config.role)
    rng_injected = rng is not None
    draw = rng if rng is not None else random.Random(config.seed)
    eligible_counts = _class_counts(prepared.examples)
    if config.test_fraction is not None:
        test, after_test, class_targets, stratify_rule = take_stratified_holdout(
            prepared.examples,
            float(config.test_fraction),
            draw,
            config.split_unit,
        )
        stratified = True
        rng_order = SPLIT_FRACTION_RNG_ORDER
        target_size = sum(class_targets.values())
    else:
        if config.test_size is None:
            raise ValueError("Informe só um entre test_size e test_fraction.")
        test, after_test = take_holdout(
            prepared.examples,
            config.test_size,
            draw,
            config.split_unit,
        )
        class_targets = None
        stratify_rule = STRATIFY_RULE_COUNT
        stratified = False
        rng_order = SPLIT_RNG_ORDER
        target_size = config.test_size
    test_counts = _class_counts(test)
    if config.val_size is None:
        val: list[Example] = []
        pool = after_test
    else:
        val, pool = take_holdout(after_test, config.val_size, draw, config.split_unit)

    train: list[Example] = []
    per_class_actual: dict[str, int] = {}
    per_class_pool: dict[str, int] = {}
    for label in LABELS:
        taken, pool_size = _sample_label(pool, label, config.per_class, draw)
        train.extend(taken)
        per_class_actual[label.value] = len(taken)
        per_class_pool[label.value] = pool_size

    shortfalls = {
        name: (got, per_class_pool[name])
        for name, got in per_class_actual.items()
        if got < config.per_class
    }
    if config.strict_per_class and shortfalls:
        detail = ", ".join(
            f"{name}={got}/{config.per_class} (pool {pool_size})"
            for name, (got, pool_size) in shortfalls.items()
        )
        raise ValueError(
            "strict_per_class: alguma classe ficou abaixo de per_class "
            f"depois do hold-out: {detail}."
        )

    train_ids = {example.example_id for example in train}
    unused = [example for example in pool if example.example_id not in train_ids]
    notes = _notes(
        config=config,
        corpus=corpus,
        test_actual=len(test),
        target_size=target_size,
        class_targets=class_targets,
        test_counts=test_counts,
        shortfalls=shortfalls,
        rng_injected=rng_injected,
    )
    manifest = SplitManifest(
        seed=config.seed,
        test_size_requested=config.test_size,
        test_fraction_requested=(
            None if config.test_fraction is None else float(config.test_fraction)
        ),
        test_target_size=target_size,
        test_size_actual=len(test),
        stratified=stratified,
        stratify_rule=stratify_rule,
        test_class_targets=class_targets,
        test_class_counts=test_counts,
        eligible_class_counts=eligible_counts,
        per_class_requested=config.per_class,
        per_class_actual=per_class_actual,
        per_class_pool_before_sample=per_class_pool,
        val_size_requested=config.val_size,
        val_size_actual=len(val),
        split_unit=config.split_unit,
        role=config.role,
        n_rows=prepared.n_rows,
        n_dropped_role=prepared.n_dropped_role,
        n_dropped_ambiguous=prepared.n_dropped_ambiguous,
        n_dropped_empty=prepared.n_dropped_empty,
        n_dropped_missing_flags=prepared.n_dropped_missing_flags,
        n_eligible=len(prepared.examples),
        n_train=len(train),
        n_val=len(val),
        n_test=len(test),
        n_unused=len(unused),
        no_signal_pool_before_sample=per_class_pool[Label.NO_SIGNAL.value],
        no_signal_sampled=per_class_actual[Label.NO_SIGNAL.value],
        conversation_note=corpus.conversation_note,
        source=corpus.source,
        subset_limit=corpus.subset_limit,
        full_file=corpus.subset_limit is None,
        rng_order=rng_order,
        rng_injected=rng_injected,
        notes=notes,
        train_ids=[example.example_id for example in train],
        val_ids=[example.example_id for example in val],
        test_ids=[example.example_id for example in test],
    )
    return DatasetSplit(train=train, val=val, test=test, unused=unused, manifest=manifest)


def _notes(
    *,
    config: SplitConfig,
    corpus: LoadedCorpus,
    test_actual: int,
    target_size: int,
    class_targets: dict[str, int] | None,
    test_counts: dict[str, int],
    shortfalls: dict[str, tuple[int, int]],
    rng_injected: bool,
) -> list[str]:
    notes = [
        "O protocolo fecha o teste em 12% estratificado e três sementes compartilhadas. "
        "Esta partição registra a fração ou a contagem, a estratificação e a semente desta invocação. "
        "Os três inteiros das sementes não são escolhidos neste código. "
        "O tamanho da validação continua em aberto; quando pedida, é uma contagem inteira depois do teste.",
        "Linhas com Satisfaction e Disatisfaction verdadeiros foram descartadas (sem regra de precedência).",
        corpus.conversation_note,
    ]
    if config.test_fraction is None:
        notes.append(
            "Esta invocação usou test_size inteiro, sem estratificar. "
            "Smoke e fixtures. A corrida do protocolo usa test_fraction 0.12."
        )
    else:
        notes.append(
            f"test_fraction {config.test_fraction} estratificado. "
            "Alvos e contagens realizadas estão neste manifesto."
        )
    if config.role is None:
        notes.append(
            "Nenhum filtro de Role. O ponto em aberto da definição continua em aberto."
        )
    elif config.role.casefold() == "user":
        notes.append(
            "Filtro Role==User é hipótese de trabalho: a pergunta é sobre a mensagem do usuário. "
            "O blueprint ainda deixa o filtro em aberto."
        )
    else:
        notes.append(f"Filtro de Role aplicado: {config.role}.")
    if config.split_unit == "turn":
        notes.append(
            "Split por turno pode colocar turnos da mesma conversa em treino e teste."
        )
    else:
        notes.append(
            "Split por conversa inferida: uma conversa não cruza o teste nem a validação. "
            "No lado restante, a amostra balanceada pode deixar turno da mesma conversa fora do treino."
        )
    if test_actual != target_size:
        if config.split_unit == "conversation":
            notes.append(
                f"O teste realizado tem {test_actual} exemplos; o alvo era {target_size}. "
                "A conversa inteira mudou de lado para não ser cortada."
            )
        else:
            notes.append(
                f"O teste realizado tem {test_actual} exemplos; o alvo era {target_size}."
            )
    if class_targets is not None:
        for label in LABELS:
            got = test_counts[label.value]
            want = class_targets[label.value]
            if got != want:
                notes.append(
                    f"Classe {label.value}: alvo {want}, realizado {got}."
                )
    if shortfalls:
        detail = ", ".join(
            f"{name}={got} (pedido {config.per_class}, pool {pool_size})"
            for name, (got, pool_size) in shortfalls.items()
        )
        notes.append(f"Amostra de treino abaixo de per_class: {detail}.")
    if corpus.subset_limit is not None:
        notes.append(
            "Há limit de linhas. Isto não é a distribuição real do arquivo e não é o hold-out do experimento."
        )
    if rng_injected:
        notes.append("Foi passado um rng externo; a semente do config não sorteou esta partição.")
    return notes


def format_split(manifest: SplitManifest) -> str:
    per_class = ", ".join(
        f"{name}={manifest.per_class_actual[name]}"
        for name in (label.value for label in LABELS)
    )
    test_classes = ", ".join(
        f"{name}={manifest.test_class_counts[name]}"
        for name in (label.value for label in LABELS)
    )
    if manifest.test_fraction_requested is None:
        asked = f"pedido {manifest.test_size_requested}"
    else:
        asked = (
            f"fração {manifest.test_fraction_requested} "
            f"(alvo {manifest.test_target_size}, estratificado)"
        )
    return "\n".join(
        [
            f"fonte: {manifest.source}",
            (
                f"teste: {manifest.n_test} ({asked}; {test_classes})  "
                f"val: {manifest.n_val}  treino: {manifest.n_train} ({per_class})  "
                f"fora do treino: {manifest.n_unused}"
            ),
            (
                f"sem sinal amostrados: {manifest.no_signal_sampled}/"
                f"{manifest.no_signal_pool_before_sample} no pool depois do hold-out"
            ),
            (
                f"semente: {manifest.seed}  unidade: {manifest.split_unit}  "
                f"role: {manifest.role or 'any'}"
            ),
            (
                "descartes: "
                f"role={manifest.n_dropped_role}  "
                f"ambiguos={manifest.n_dropped_ambiguous}  "
                f"vazios={manifest.n_dropped_empty}  "
                f"sem flag={manifest.n_dropped_missing_flags}"
            ),
        ]
    )
