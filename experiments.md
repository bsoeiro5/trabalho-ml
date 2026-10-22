# Experiências e plano dos slides

Todas as comparações são feitas na **validação cruzada do dev** (5 folds; WA ao nível do paciente,
média ± desvio-padrão). O teste só é usado no fim, com o modelo já escolhido.

Tabela atualizada: `python -m murmur report` → `outputs/report/cv_table.md`. Todas as corridas: `bash scripts/experiments.sh`.
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

---

## Plano dos slides (10 minutos)

O enunciado diz que a parte mais importante é a **implementação**: motivar as escolhas, mostrar
resultados intermédios e o que se mudou. Tempo indicativo entre parênteses.

1. **Problema** (0:30). Sopros cardíacos, classificação por paciente em 3 classes, métrica WA (5/3/1).
2. **Dados** (1:00). 942 pacientes / 3163 gravações; desequilíbrio; sempre-Present 0.50 vs sempre-Absent 0.39;
   70 crianças com 2 IDs; sopro só em alguns focos (38% dos Present). Figura: barras por classe + exemplo de sinal.
3. **Protocolo de avaliação** (1:00). Diagrama: teste 20% por sujeito → dev 5 folds → OOF → regra de decisão → teste uma vez.
4. **Pré-processamento e entradas** (1:00). Filtro 25–800 Hz, 2 kHz, janelas de 4 s; figura log-mel Present vs Absent; MFCC para o RNN.
5. **CNN** (0:45). Diagrama da arquitetura + porquê.
6. **RNN** (0:45). Diagrama BiGRU + atenção + porquê.
7. **Treino** (1:00). Perda pesada (5,3,1) e argumento de Bayes; augmentation; early stopping na WA.
8. **Uma previsão por paciente** (0:45). Labels por foco, agregação max vs mean, regra de decisão com cross-fitting.
9. **Resultados intermédios** (1:30). Tabela/gráfico das ablações (tabela acima): o que mudou e porquê.
10. **Resultados no teste** (1:00). WA + matriz de confusão: CNN vs RNN (vs ensemble), IC 95%.
11. **Interpretação e conclusões** (1:00). Atenção/saliência na sístole; erros por grau do sopro; limitações
    (labels ruidosas, Unknown mal definido, teste pequeno); trabalho futuro.

**Apêndice.** Hiperparâmetros completos, curvas de aprendizagem por fold, WA por fold, análise de erros
por campanha/idade, resultados do bónus.

Figuras prontas: `outputs/report/*.png` (`python -m murmur report`), `outputs/<exp>/test/confusion_matrix.png`,
notebooks `01_eda.ipynb` e `02_results.ipynb`.
