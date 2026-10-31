"""Partição na ordem fechada do protocolo.

1. Separar o teste na distribuição real (tamanho a definir; parâmetro obrigatório).
2. Do restante, opcionalmente separar validação (tamanho a definir).
3. Do que sobrou, treino balanceado: cerca de 10 mil por classe.
4. Os exemplos de sem sinal do treino são amostra aleatória do pool, não o pool inteiro.

A semente entra no manifesto. O valor segue a definir no protocolo.
Few-shot e early stopping ficam de fora do teste: esta função não coloca exemplo
de teste no treino nem na validação.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from user_feedback_analysis.labels import LABELS, Label, label_from_flags
from user_feedback_analysis.records import Example, LoadedCorpus

SPLIT_RNG_ORDER = (
    "shuffle do teste; shuffle da validação se val_size; "
    "amostra SAT; amostra DSAT; amostra sem sinal "
    "(shuffle da classe se o pool for menor que per_class)"
)


@dataclass(frozen=True)
class SplitConfig:
    seed: int
    test_size: int
    per_class: int = 10_000
    val_size: int | None = None
    split_unit: str = "conversation"
    role: str | None = "User"
    strict_per_class: bool = False

    def __post_init__(self) -> None:
        _require_count("seed", self.seed, minimum=0)
        _require_count("test_size", self.test_size, minimum=1)
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
    test_size_requested: int
    test_size_actual: int
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
            "test_size_actual": self.test_size_actual,
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
    test, after_test = take_holdout(
        prepared.examples,
        config.test_size,
        draw,
        config.split_unit,
    )
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
        shortfalls=shortfalls,
        rng_injected=rng_injected,
    )
    manifest = SplitManifest(
        seed=config.seed,
        test_size_requested=config.test_size,
        test_size_actual=len(test),
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
        rng_order=SPLIT_RNG_ORDER,
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
    shortfalls: dict[str, tuple[int, int]],
    rng_injected: bool,
) -> list[str]:
    notes = [
        "Tamanho de teste, validação e semente não estão fechados no protocolo; ficam registrados aqui.",
        "Linhas com Satisfaction e Disatisfaction verdadeiros foram descartadas (sem regra de precedência).",
        corpus.conversation_note,
    ]
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
    if test_actual != config.test_size:
        notes.append(
            f"O teste realizado tem {test_actual} exemplos; o pedido era {config.test_size}. "
            "A conversa inteira mudou de lado para não ser cortada."
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
    return "\n".join(
        [
            f"fonte: {manifest.source}",
            (
                f"teste: {manifest.n_test} (pedido {manifest.test_size_requested})  "
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
