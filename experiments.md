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

## Fase 1: receita de treino

Valores: WA de CV com agregação *max* + argmax (a regra base), média ± dp nos 5 folds.

| # | Experiência | Alteração | WA CV | Observações / decisão |
|---|---|---|---|---|
| 1 | `cnn_es` | CNN, até 60 épocas, *early stopping* na WA de validação (paciência 12) | 0.812 ± 0.022 | Curvas de validação muito ruidosas (WA 0.61–0.85 entre épocas). A época escolhida é um pico, o que torna as previsões OOF **otimistas**. |
| 2 | `rnn_es` | RNN, mesma receita | 0.806 ± 0.022 | Mesmo padrão de erros que a CNN (Present 126/143, Unknown 0/54). |
| 3 | `cnn_es-ema` | + média móvel exponencial dos pesos (0.995) | 0.808 ± 0.017 | Igual dentro do ruído, com menos variância entre folds. A comparação favorece injustamente o nº 1 (curvas mais ruidosas dão picos mais altos). |
| 4 | `cnn_fixed30` | 30 épocas fixas (cosine até 0) + EMA, **sem escolher a época** no fold de validação | 0.794 ± 0.028 | OOF honesto: o otimismo do *early stopping* era ~2 pontos. Curvas suaves; a WA estabiliza a partir da época ~9. |
| 5 | `rnn_fixed30` | idem para o RNN | (a correr) | |
| 6 | `cnn` | **20 épocas fixas** + EMA (receita base da fase 2) | 0.785 ± 0.023 | ~1 ponto abaixo de 30 épocas (melhor ou igual em todos os folds). As ablações usam 20 épocas (comparação relativa, mais baratas); o modelo final usa 30. Nunca prevê Unknown (0/54). |
| 7 | `rnn` | idem para o RNN | (em fila) | |

**Decisão:** épocas fixas + EMA (20 nas ablações, 30 no modelo final). Com estimativas honestas, as ablações comparam-se com justiça.
O teste dirá o valor real, sem ter servido para escolher nada.

### Diário

- **Agregação e regra de decisão (`cnn_es`, dev OOF).**
  - *max* + argmax dá 0.812, contra 0.778 da *mean* + argmax: +3.4 pontos, como previsto (basta um foco com sopro).
  - Afinar os multiplicadores **não** ajuda com *max* (0.802 cross-fitted), mas ajuda com *mean* (0.807, P×1.68, que compensa a diluição). Isto confirma o argumento de Bayes: com a perda pesada (5,3,1), o argmax já está perto do ótimo.
  - Outras agregações testadas nas mesmas previsões OOF (U pelo máximo, A pelo mínimo, média dos 2 maiores P) não ganharam, e mantém-se *P max, U mean*.
- **Onde erra (`cnn_es`, OOF pooled).**
  - Recall: Present 128/143 (89.5%), Absent 509/556 (91.5%), Unknown **5/54** (9%).
  - Os Unknown vão quase todos para Absent. Mesmo assim têm AUC 0.84 contra o resto (o Present tem 0.95): o modelo ordena-os razoavelmente, mas o score de Unknown nunca ganha.
- **Ensemble CNN+RNN (`cnn_es` + `rnn_es`).** 0.808, sem ganho: as redes concordam em 93% das decisões (correlação 0.88 dos scores de Present).
- **Regra escolhida por ruído.** Em `cnn_fixed30`, "mean + multiplicadores" ganhava a "max + argmax" por 0.0005. Passou a haver uma tolerância de 0.005: fica a regra mais simples, a não ser que outra seja claramente melhor.
- **Paralelismo.** Duas experiências ao mesmo tempo na GPU (MPS) ficam 2–4× mais lentas cada uma. Corre-se uma na GPU (CNN) e uma no CPU (RNN).

## Fase 2: ablações (com a receita escolhida na fase 1)

| # | Experiência | Alteração | Hipótese | WA CV | Observações / decisão |
|---|---|---|---|---|---|
| 6 | `cnn_w-none` | Perda sem pesos | Prevê quase sempre Absent: recall de Present baixo, WA pior | | |
| 7 | `cnn_w-balanced` | Pesos inverso da frequência (Unknown pesa ~10× Absent) | Sobre-prevê Unknown (a métrica só lhe dá peso 3) | | |
| 8 | `cnn_lab-patient` | Gravações herdam a label do paciente (+ agregação mean) | Labels mais ruidosas (focos sem sopro marcados Present): pior | | |
| 9 | `cnn_win2s` | Janelas de 2 s | Menos batidas por janela, menos contexto: pior? | | |
| 10 | `cnn_win8s` | Janelas de 8 s | Mais contexto mas menos recortes distintos: ? | | |
| 11 | `cnn_noaug` | Sem *augmentation* | Mais sobreajuste (gap treino/validação maior) | | |
| 12 | `cnn_small` | Canais 8-16-32-64 (91 k parâmetros) | Pode chegar com tão poucos dados | | |
| 13 | `cnn_big` | Canais 32-64-128-256 (1.2 M parâmetros) | Sobreajuste sem ganho | | |
| 14 | `rnn_logmel` | RNN com log-mel 64 em vez de MFCC | Log-mel guarda mais detalhe espectral: ? | | |
| 15 | `rnn_lstm` | LSTM em vez de GRU | Semelhante, mais lento | | |
| 16 | `rnn_uni` | GRU unidirecional | Pior: perde o contexto futuro | | |
| 17 | `cnn+rnn` | Ensemble (média das probabilidades) | Erros pouco correlacionados: ganho pequeno | | |

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
