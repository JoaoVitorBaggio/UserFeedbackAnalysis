# Função de loss

Nada disto foi treinado. Não há valor de loss.

A loss abaixo é a do Transformer fine-tunado (braço núcleo, Nikolas Chemale). O baseline não treina neste corpus. O LLM zero-shot / few-shot não recebe gradiente aqui. O espaço de rótulo, esse sim, é o mesmo na avaliação dos três braços.

## Duas opções

### BCE binário

Uma logística. O rótulo colapsa para “satisfeito ou não”.

Esse colapso apaga a distinção SAT versus DSAT, que é a polaridade do `sat_dsat_annotation`. **Sem sinal** também precisa de uma regra explícita: sair do treino, virar a classe negativa, ou virar um segundo problema binário (sinal versus sem sinal). Cada regra responde uma pergunta diferente da pergunta do trabalho.

### Cross-entropy categórica

Três classes: SAT, DSAT, sem sinal. Uma distribuição sobre o espaço inteiro. **Sem sinal** permanece classe própria. Não sai da loss.

## Decisão

A loss de treino do Transformer fine-tunado é a cross-entropy categórica com 3 classes (SAT, DSAT, sem sinal); o mesmo espaço de rótulo vale para a avaliação dos três braços. A call de 31/10/2026 reafirmou este quadro. BCE binária permanece só ablação ou apêndice.

| Item | Escolha |
| --- | --- |
| Loss de treino do Transformer | Cross-entropy categórica, 3 classes |
| Classes | SAT, DSAT, sem sinal |
| Sem sinal na loss | Classe própria; não é descartada |
| Ponderação de classe no caminho mínimo | Nenhuma. O treino já mira o balanceamento de 5000 por classe, com `--strict-per-class` |
| Avaliação dos três braços | O mesmo espaço de três classes |

A pergunta e o `sat_dsat_annotation` são de três vias. Precisão, recall e macro F1 já são métricas multiclasse. Juntar SAT e DSAT num binário “satisfeito?” apaga a polaridade que o arquivo rotula.

BCE binária (SAT versus DSAT apenas, ou sinal versus sem sinal) pode aparecer depois como ablação ou apêndice. Não é o protocolo principal.
