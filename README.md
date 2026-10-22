# Deteção de sopros cardíacos em fonocardiogramas

Trabalho prático de **Aprendizagem Computacional II** (CC3043, DCC/FCUP, 2026/2027).

O objetivo é classificar cada **paciente** como *Present*, *Unknown* ou *Absent* (sopro cardíaco)
a partir das gravações de auscultação do dataset **CirCor DigiScope**. É a tarefa *murmur* do
PhysioNet Challenge 2022. Implementamos dois classificadores profundos:

- **CNN 2D** sobre espectrogramas log-mel de janelas de 4 s;
- **RNN**: GRU bidirecional com *attention pooling*, sobre sequências de MFCC (+Δ, +ΔΔ).

A métrica principal é a *weighted accuracy* ao nível do paciente, com pesos 5 (Present),
3 (Unknown) e 1 (Absent).

---

## Estrutura do repositório

```
configs/          configurações YAML (base.yaml, cnn.yaml, rnn.yaml, smoke.yaml)
murmur/           código (pacote Python)
  data.py           metadados, labels por gravação, sujeitos (IDs duplicados)
  splits.py         splits por sujeito, estratificados
  audio.py          leitura, filtragem, reamostragem, normalização, cache
  features.py       log-mel / MFCC / SpecAugment em PyTorch (calculados na GPU)
  models.py         CNN2D, RNN (GRU/LSTM + atenção)
  engine.py         treino (recortes aleatórios, augmentation, early stopping) e inferência
  aggregation.py    gravações -> paciente, regra de decisão
  metrics.py        weighted accuracy, matriz de confusão, métricas adicionais, bootstrap
  interpret.py      atenção do RNN / saliência da CNN vs segmentação S1-sístole-S2-diástole
  prepare.py        CLI: splits + cache de áudio
  cv.py             CLI: validação cruzada no dev (um modelo por fold + previsões out-of-fold)
  evaluate.py       CLI: avaliação final no teste (ou só CV de um ensemble com --dev-only)
  predict.py        CLI: previsões para dados novos (bónus) -> CSV patient_id,murmur
  report.py         CLI: tabela comparativa e figuras de todas as experiências
notebooks/        01_eda.ipynb (análise exploratória), 02_results.ipynb (resultados e interpretação)
scripts/          download_data.sh, run_all.sh, run_experiments.sh, make_synthetic_data.py
splits/           splits.csv (fixo, seed 42; igual para todos os membros do grupo)
docs/             experiment_log.md (registo de experiências), slides_outline.md
tests/            testes (pytest)
```

## Instalação

Python ≥ 3.10. Com conda:

```bash
conda create -n ml2 python=3.12 -y
conda activate ml2
pip install -r requirements.txt
```

No portátil do Bernardo, o env `datasci` já tem tudo (`conda activate datasci`).

## Dados

```bash
bash scripts/download_data.sh          # zip de ~470 MB; se a ligação cair, correr de novo (retoma)
```

No fim deve existir `data/circor/training_data.csv` e `data/circor/training_data/` com os ficheiros
`.wav/.hea/.tsv/.txt`. Em alternativa, descarregar o zip de
<https://physionet.org/content/circor-heart-sound/1.0.3/> e extrair para `data/circor/`.
Os dados **não** vão para o git.

## Como correr

**Teste rápido sem descarregar nada** (dataset sintético no mesmo formato, ~2 min):

```bash
python scripts/make_synthetic_data.py --out data/synthetic
python -m murmur.prepare --config configs/base.yaml configs/smoke.yaml
python -m murmur.cv --config configs/cnn.yaml configs/smoke.yaml
python -m murmur.cv --config configs/rnn.yaml configs/smoke.yaml
python -m murmur.report --output-dir outputs/smoke
pytest -q
```

**Pipeline real** (o mesmo que `scripts/run_all.sh`):

```bash
python -m murmur.prepare --config configs/base.yaml        # cache de áudio (splits já existem)
python -m murmur.cv --config configs/cnn.yaml              # ~45 min (GPU/MPS)
python -m murmur.cv --config configs/rnn.yaml              # ~60 min (CPU); pode correr em paralelo com o anterior
python -m murmur.evaluate --exp cnn rnn --dev-only         # CV do ensemble, sem tocar no teste
python -m murmur.report                                    # tabela + figuras em outputs/report/
python -m murmur.evaluate --exp cnn                        # TESTE: só no fim, uma vez
python -m murmur.evaluate --exp rnn
```

Experiências / ablações: `bash scripts/run_experiments.sh` (ou uma a uma com
`python -m murmur.cv --config configs/cnn.yaml --name cnn_lr3e-4 --set train.lr=3e-4`).
Os resultados vão para `docs/experiment_log.md`.

**Bónus (dados escondidos):** depois da avaliação no teste, retreinar com todos os pacientes e prever:

```bash
python -m murmur.cv --config configs/cnn.yaml --all-data   # -> outputs/cnn_all
python -m murmur.cv --config configs/rnn.yaml --all-data   # -> outputs/rnn_all
python -m murmur.predict --exp cnn_all rnn_all --data-dir data/hidden --out predictions.csv
```

Cada experiência fica em `outputs/<nome>/`: `config.yaml`, `train.log`, `fold<k>/model.pt`,
`fold<k>/history.csv`, `oof_recordings.csv`, `cv_summary.json` e, depois da avaliação, `test/`.

---

## Protocolo de avaliação

| Passo | Detalhe |
|---|---|
| Unidade de divisão | **Sujeito**, não `Patient ID`: 140 IDs são 70 crianças gravadas nas duas campanhas (campo `Additional ID`). Dividir só por ID poria a mesma criança em treino e teste. |
| Teste (hold-out) | 20%: **189 pacientes** (36 P / 14 U / 139 A), estratificado por label × campanha (`splits/splits.csv`, seed 42). |
| Desenvolvimento | 753 pacientes, **5 folds** estratificados por sujeito (~150 pacientes/fold). |
| Treino / validação | Em cada fold: treino em 4 folds, *early stopping* no 5.º. Isto dá previsões *out-of-fold* (OOF) para todos os pacientes dev. |
| Escolhas | Arquitetura, hiperparâmetros, agregação e regra de decisão: **só com as previsões OOF do dev**. |
| WA de CV reportada | Média ± desvio-padrão nos 5 folds. Quando a regra usa multiplicadores afinados, os do fold *k* são afinados nos outros 4 folds (*cross-fitting*), e por isso nenhum paciente afina a sua própria decisão. |
| Teste | Usado **uma vez**, no fim: ensemble dos 5 modelos dos folds + regra de decisão do OOF. Reportamos WA, matriz de confusão, IC 95% por bootstrap e WA de cada modelo individual (média ± dp). |
| Referências triviais | Sempre *Absent*: WA = 0.387. Sempre *Present*: WA = 0.499. |

## Decisões de desenho (e porquê)

**Pré-processamento.** Passa-banda Butterworth 25–800 Hz (S1/S2 < ~200 Hz, sopros até ~600 Hz;
corta movimento/respiração e ruído de alta frequência) → reamostragem 4 kHz → 2 kHz (metade do
custo, Nyquist 1 kHz acima da banda útil) → normalização por gravação (média 0, variância 1).

**Entradas de tamanho fixo.** As gravações duram 5–80 s. As redes recebem janelas de **4 s**, o que
dá várias batidas mesmo a 60 bpm:
- em treino, 4 recortes aleatórios por gravação e por época (aumento de dados, e as gravações longas não dominam);
- em avaliação, janelas deslizantes com 50% de sobreposição, e a média das probabilidades dá a gravação.

**Features (calculadas dentro do modelo, na GPU).**
- CNN: log-mel com 64 bandas (25–800 Hz), janela de 64 ms e passo de 16 ms → imagem 64 × 251.
- RNN: 20 MFCC + Δ + ΔΔ, passo de 32 ms → sequência de 126 vetores de 60 features.

Calcular as features no modelo permite fazer *augmentation* no sinal (recorte, *time-stretch*,
ruído) e mantém um único caminho de código para treino, avaliação e predição.

**Labels por gravação.** A label é do paciente, mas em 38% dos *Present* o sopro não se ouve em
todos os focos. Uma gravação de um paciente *Present* só é *Present* se o foco estiver em
`Murmur locations`; caso contrário é *Absent*. *Unknown* e *Absent* passam a todas as gravações.
A alternativa "copiar a label do paciente" é testada como ablação.

**Uma previsão por paciente.**
- Agregação **max**: P(Present) do paciente é o máximo sobre as gravações (basta um foco com sopro); *Unknown* e *Absent* são a média.
- Agregação **mean**: média de tudo.
- A escolha entre as duas é feita na CV.

**Perda e desequilíbrio.** Como o denominador da WA só depende das labels verdadeiras, maximizar a
WA equivale a maximizar $\sum_i w_{y_i}\,[\hat y_i = y_i]$. A regra de Bayes é
$\hat y = \arg\max_c w_c\, p(c\mid x)$ com $w = (5, 3, 1)$. Treinar com **entropia cruzada pesada
por $w$** leva a rede a estimar $q_c \propto w_c\, p(c\mid x)$, e por isso o argmax da rede já
aproxima a decisão ótima para a métrica. Os pesos "inverso da frequência" e "sem pesos" são
comparados como ablação.

**Regra de decisão.** Opcionalmente, multiplicadores para os scores *Present*/*Unknown* (grelha
2^[-3, 3]), para corrigir a calibração e o efeito da agregação. Só são usados se baterem o argmax
em *cross-fitting*: nos dados sintéticos, um modelo mal treinado piorava com a afinação
(0.78 → 0.59), e o protocolo protege contra isso.

**CNN 2D** (327 k parâmetros, inspirada nas PANNs, Kong et al. 2020).
- Entrada: log-mel 64×251, com BatchNorm por banda.
- 4 blocos [Conv 3×3 → BN → ReLU] × 2 → AvgPool 2×2 → Dropout 0.1, com 16-32-64-128 canais.
- Média na frequência, depois média + máximo no tempo (vetor de 256).
- FC 128 (ReLU, dropout 0.3) → 3 classes.
- Porquê: o sopro é um padrão tempo-frequência local que se repete a cada batida (energia na sístole), o tipo de padrão que as convoluções detetam. O *pooling* global torna a rede independente da duração.

**RNN** (140 k parâmetros).
- Entrada: MFCC+Δ+ΔΔ (60 × 126), com BatchNorm por feature.
- GRU bidirecional, 2 camadas × 64 unidades, dropout 0.3 entre camadas.
- *Attention pooling* no tempo, $a_t = \mathrm{softmax}(v^\top \tanh(W h_t))$.
- FC 64 → 3 classes.
- Porquê: o sinal é uma sequência quase periódica (S1, sístole, S2, diástole). O GRU bidirecional usa o contexto antes e depois de cada instante, e a atenção escolhe os instantes relevantes. Dá também interpretabilidade: verificamos se a atenção cai na sístole.
- O GRU tem menos parâmetros do que o LSTM e treina mais depressa, e o LSTM e o unidirecional são ablações. No Mac, o GRU treina ~2.5× mais depressa no CPU do que em MPS.

**Treino.**
- AdamW (lr 1e-3, weight decay 1e-2), batch 64, 1 época de *warm-up* + *cosine decay*, *gradient clipping* 5.
- Até 60 épocas, com *early stopping* (paciência 12) na WA de validação ao nível do paciente, a métrica que interessa.
- *Augmentation*: recortes aleatórios, *time-stretch* ±10% (simula frequência cardíaca), ganho ±6 dB, ruído branco com SNR 10–40 dB, SpecAugment (máscaras de frequência e de tempo).
- Os valores iniciais vêm da literatura (PANNs, SpecAugment, equipas do Challenge) e são ajustados com as ablações da CV (`docs/experiment_log.md`).

## Calendário

| Data | O quê |
|---|---|
| 5 nov 2026 | *Checkpoint* nas aulas práticas (10% da nota): explicar o que está feito |
| 30 nov 2026, 23:59 | Entrega no Moodle: código, slides (PDF, 10 min), `predictions.csv` do bónus, autoavaliação |
| 3 e 10 dez 2026 | Apresentações (presença obrigatória de todos) |

Estrutura proposta para os slides: `docs/slides_outline.md`.
