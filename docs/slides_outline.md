# Estrutura dos slides (10 minutos)

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
9. **Resultados intermédios** (1:30). Tabela/gráfico das ablações (docs/experiment_log.md): o que mudou e porquê.
10. **Resultados no teste** (1:00). WA + matriz de confusão: CNN vs RNN (vs ensemble), IC 95%.
11. **Interpretação e conclusões** (1:00). Atenção/saliência na sístole; erros por grau do sopro; limitações
    (labels ruidosas, Unknown mal definido, teste pequeno); trabalho futuro.

**Apêndice.** Hiperparâmetros completos, curvas de aprendizagem por fold, WA por fold, análise de erros
por campanha/idade, resultados do bónus.

Figuras prontas: `outputs/report/*.png` (`python -m murmur.report`), `outputs/<exp>/test/confusion_matrix.png`,
notebooks `01_eda.ipynb` e `02_results.ipynb`.
