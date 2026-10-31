# Referências

Nenhum número abaixo é resultado deste repositório. São a citação do corpus, o ponteiro do arquivo e a nota de trabalho próximo.

## Corpus

Shi, Taiwei; Wang, Zhuoer; Yang, Longqi; Lin, Ying-Chun; He, Zexue; Wan, Mengting; Zhou, Pei; Jauhar, Sujay; Chen, Sihao; Xia, Shan; Zhang, Hongfei; Zhao, Jieyu; Xu, Xiaofeng; Song, Xia; Neville, Jennifer. **WildFeedback: Aligning LLMs With In-situ User Interactions And Feedback.** arXiv:2408.15549, 2024. <https://arxiv.org/abs/2408.15549>

Versão de anais, posterior ao preprint: ACL Anthology, 2026.acl-long.1701. <https://aclanthology.org/2026.acl-long.1701/>

Dataset: Microsoft, `microsoft/WildFeedback`, config `sat_dsat_annotation`. <https://huggingface.co/datasets/microsoft/WildFeedback>

O card do dataset (consulta em out/2026) declara licença Open Data Commons Attribution (ODC-By), curadoria Microsoft Research, e o subconjunto de anotações SAT/DSAT ao lado dos pares de preferência. O artigo descreve a classificação de enunciados por GPT-4 a partir de rubricas SAT/DSAT. A checagem humana deste grupo (~200 exemplos) continua em aberto; ver [definicao.md](definicao.md).

## Trabalho próximo

Nota do grupo (call e chat de 31/10/2026): não há modelo aberto treinado no `sat_dsat_annotation` com as classes SAT, DSAT e sem sinal. Esta seção não é uma busca exaustiva da literatura, e não há número de acerto deste repositório.

O modelo aberto mais próximo que o grupo cita é o **feedback-detector**: classificador ModernBERT, cerca de 0,1B parâmetros no card, [llm-semantic-router/feedback-detector](https://huggingface.co/llm-semantic-router/feedback-detector) (consulta em 31/10/2026). O card lista quatro classes: `SAT`, `NEED_CLARIFICATION`, `WRONG_ANSWER`, `WANT_DIFFERENT`. A taxonomia não é a deste trabalho (não há DSAT nem sem sinal; insatisfação aparece partida em esclarecimento, resposta errada e pedido de outra resposta). Esse modelo não esvazia a contribuição daqui: o alvo, o corpus e as três classes continuam outros.

Não localizamos artigo para citar no lugar do card. A fonte desta nota é o card do Hub.
