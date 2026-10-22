# Registo de experiências

Todas as comparações são feitas na **validação cruzada do dev** (5 folds; WA ao nível do paciente,
média ± desvio-padrão). O teste só é usado no fim, com o modelo já escolhido.

Gerar a tabela atualizada: `python -m murmur.report` → `outputs/report/cv_table.md`.
Diferenças menores do que o desvio-padrão entre folds (≈ 0.02–0.04) não devem levar a mudanças.

Para cada experiência: o que mudou, o que esperávamos (hipótese), o resultado, a interpretação
e a decisão tomada. É isto que os slides pedem ("show intermediate results, how you interpreted
them, and what you changed").

## Referências

| Classificador | WA (dev) |
|---|---|
| Sempre *Absent* | 0.387 |
| Sempre *Present* | 0.499 |

## Experiências

| # | Experiência | Alteração | Hipótese | WA CV | Observações / decisão |
|---|---|---|---|---|---|
| 1 | `cnn` | CNN base (configs/cnn.yaml) | — | | |
| 2 | `rnn` | RNN base (configs/rnn.yaml) | — | | |
| 3 | `cnn_w-none` | Perda sem pesos | Prevê quase sempre Absent: recall de Present baixo, WA pior | | |
| 4 | `cnn_w-balanced` | Pesos inverso da frequência (Unknown pesa ~10× Absent) | Sobre-prevê Unknown (a métrica só lhe dá peso 3) | | |
| 5 | `cnn_lab-patient` | Gravações herdam a label do paciente (+ agregação mean) | Labels mais ruidosas (focos sem sopro marcados Present): pior | | |
| 6 | `cnn_win2s` | Janelas de 2 s | Menos batidas por janela, menos contexto: pior? | | |
| 7 | `cnn_win8s` | Janelas de 8 s | Mais contexto mas menos recortes distintos: ? | | |
| 8 | `cnn_noaug` | Sem *augmentation* | Mais sobreajuste (gap treino/validação maior) | | |
| 9 | `cnn_small` | Canais 8-16-32-64 (91 k parâmetros) | Pode chegar com tão poucos dados | | |
| 10 | `cnn_big` | Canais 32-64-128-256 (1.2 M parâmetros) | Sobreajuste sem ganho | | |
| 11 | `rnn_logmel` | RNN com log-mel 64 em vez de MFCC | Log-mel guarda mais detalhe espectral: ? | | |
| 12 | `rnn_lstm` | LSTM em vez de GRU | Semelhante, mais lento | | |
| 13 | `rnn_uni` | GRU unidirecional | Pior: perde o contexto futuro | | |
| 14 | `cnn+rnn` | Ensemble (média das probabilidades) | Erros pouco correlacionados: ganho pequeno | | |

## Notas

- Hiperparâmetros afinados a partir do melhor da tabela (lr, dropout, weight decay): registar aqui também.
- Se uma alteração ganha com o argmax mas perde com os limiares afinados (ou vice-versa), anotar: a regra de decisão também é escolhida na CV.
