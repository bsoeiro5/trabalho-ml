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
murmur/                 código (pacote Python)
  data.py                 metadados, labels por gravação, sujeitos, splits, leitura e pré-processamento do áudio
  features.py             log-mel, MFCC, SpecAugment (em PyTorch, calculados na GPU)
  models.py               CNN 2D e RNN (GRU/LSTM + atenção)
  training.py             treino (recortes aleatórios, augmentation, EMA dos pesos) e inferência por janelas
  evaluation.py           weighted accuracy, matriz de confusão, agregação gravações -> paciente, regra de decisão
  interpret.py            atenção do RNN / saliência da CNN vs fases do ciclo cardíaco
  plots.py                figuras
  utils.py                configuração YAML, seed, device, logging
  pipeline.py             comandos: prepare, cv, evaluate, predict, report
configs/                base.yaml, cnn.yaml, rnn.yaml, splits.csv (splits fixos, iguais para todo o grupo)
notebooks/              01_eda.ipynb (análise exploratória), 02_results.ipynb (resultados e interpretação)
scripts/                download_data.sh, experiments.sh (todas as experiências por ordem)
tests/                  testes unitários + teste de ponta a ponta com dados sintéticos
experiments.md          registo das experiências e plano dos slides
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

O código espera o dataset **ao lado do repositório**, em `../circor-dataset` (`data.dir` em
`configs/base.yaml`):

```
ac2/
├── circor-dataset/        training_data.csv + training_data/*.wav|.hea|.tsv|.txt
└── trabalho-ml/           este repositório
```

Para o descarregar: `bash scripts/download_data.sh` (zip de ~470 MB, retoma se a ligação cair), ou
manualmente em <https://physionet.org/content/circor-heart-sound/1.0.3/>.

## Como correr

```bash
python -m murmur prepare                                   # cache de áudio (os splits já existem)
python -m murmur cv --config configs/cnn.yaml              # 5 folds, ~25 min (GPU/MPS)
python -m murmur cv --config configs/rnn.yaml              # 5 folds, ~60 min (CPU); pode correr em paralelo
python -m murmur evaluate --exp cnn rnn --dev-only         # CV do ensemble, sem tocar no teste
python -m murmur report                                    # tabela + figuras em outputs/report/
python -m murmur evaluate --exp cnn                        # TESTE: só no fim, uma vez
```

- `python -m murmur --help` mostra todos os comandos.
- Em paralelo: **uma** experiência na GPU (CNN) + **uma** no CPU (RNN). Duas na GPU (MPS) ao mesmo tempo ficam cada uma 2–4× mais lentas, o que é pior do que corrê-las seguidas.
- Para mudar um hiperparâmetro sem criar ficheiros: `python -m murmur cv --config configs/cnn.yaml --name cnn_lr3e-4 --set train.lr=3e-4`.
- Todas as experiências do relatório: `bash scripts/experiments.sh` (as que já acabaram são saltadas).
- Testes (~1 min, não precisam dos dados reais): `pytest -q`.

Cada experiência fica em `outputs/<nome>/`: `config.yaml`, `train.log`, `fold<k>/model.pt`,
`fold<k>/history.csv`, `oof_recordings.csv`, `cv_summary.json` e, depois da avaliação, `test/`.

**Bónus (dados escondidos).** Depois da avaliação no teste, retreinar com todos os pacientes e prever:

```bash
python -m murmur cv --config configs/cnn.yaml --all-data   # -> outputs/cnn_all
python -m murmur cv --config configs/rnn.yaml --all-data   # -> outputs/rnn_all
python -m murmur predict --exp cnn_all rnn_all --data-dir ../hidden-data --out predictions.csv
```

---

## Protocolo de avaliação

| Passo | Detalhe |
|---|---|
| Unidade de divisão | **Sujeito**, não `Patient ID`: 140 IDs são 70 crianças gravadas nas duas campanhas (campo `Additional ID`). Dividir só por ID poria a mesma criança em treino e teste. |
| Teste (hold-out) | 20%: **189 pacientes** (36 P / 14 U / 139 A), estratificado por label × campanha (`configs/splits.csv`, seed 42). |
| Desenvolvimento | 753 pacientes, **5 folds** estratificados por sujeito (~150 pacientes/fold). |
| Treino / validação | Em cada fold: treino em 4 folds com um número **fixo** de épocas, e previsão no 5.º. Isto dá previsões *out-of-fold* (OOF) para todos os pacientes dev. A época **não** é escolhida no fold de validação, senão as previsões OOF ficam otimistas (ver Treino). |
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
2^[-3, 3]), para corrigir a calibração e o efeito da agregação. A regra (agregação × argmax/afinada)
é escolhida pela WA de CV com *cross-fitting*. Fica a mais simples (argmax, agregação *max*), a não
ser que outra ganhe por mais de 0.005, para não escolher regras por ruído.
- Nos dados sintéticos, um modelo mal treinado piorava com a afinação (0.78 → 0.59), e o protocolo protege contra isso.
- Nos dados reais, com *max*, a afinação não ajuda: a perda pesada já põe o argmax perto do ótimo, como previa o argumento de Bayes.

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
- **Número fixo de épocas** (cosine até 0) e avaliação dos **pesos EMA** (média móvel exponencial, 0.995).
- Primeiro usámos *early stopping* na WA de validação. As curvas eram muito ruidosas (WA 0.61–0.85 entre épocas consecutivas) e a época escolhida era um pico de sorte, o que deixava as previsões OOF ~2 pontos otimistas (0.812 → 0.794 com épocas fixas).
- Com a EMA as curvas ficam suaves e a WA estabiliza a partir da época ~9. Com 30 épocas fica ~1 ponto acima de 20 (0.794 vs 0.785, melhor ou igual em todos os folds): as ablações usam 20 épocas (mais baratas, comparações relativas) e o modelo final 30. Ver `experiments.md`.
- *Augmentation*: recortes aleatórios, *time-stretch* ±10% (simula frequência cardíaca), ganho ±6 dB, ruído branco com SNR 10–40 dB, SpecAugment (máscaras de frequência e de tempo).
- Os valores iniciais vêm da literatura (PANNs, SpecAugment, equipas do Challenge) e são ajustados com as ablações da CV (`experiments.md`).

## Calendário

| Data | O quê |
|---|---|
| 5 nov 2026 | *Checkpoint* nas aulas práticas (10% da nota): explicar o que está feito |
| 30 nov 2026, 23:59 | Entrega no Moodle: código, slides (PDF, 10 min), `predictions.csv` do bónus, autoavaliação |
| 3 e 10 dez 2026 | Apresentações (presença obrigatória de todos) |

Registo das experiências e estrutura dos slides: `experiments.md`.
