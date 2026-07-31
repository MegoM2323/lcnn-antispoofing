# Voice Anti-Spoofing: LCNN на ASVspoof2019 LA

Система детекции синтезированной и преобразованной речи (countermeasure, CM) на
логическом доступе (Logical Access) датасета
[ASVspoof2019](https://datashare.ed.ac.uk/handle/10283/3336).
Модель: Light CNN ([arXiv:1511.02683](https://arxiv.org/abs/1511.02683),
[arXiv:1904.05576](https://arxiv.org/abs/1904.05576)), рецепт обучения взят из
[arXiv:2103.11326](https://arxiv.org/abs/2103.11326).

Проект построен на
[PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template):
конфигурация через [Hydra](https://hydra.cc/), логирование экспериментов через
[Comet ML](https://www.comet.com/docs/v2/).

## Содержание

- [Задача и метрика](#задача-и-метрика)
- [Архитектура решения](#архитектура-решения)
- [Установка](#установка)
- [Данные](#данные)
- [Обучение](#обучение)
- [Инференс и сабмит](#инференс-и-сабмит)
- [Структура репозитория](#структура-репозитория)
- [Credits](#credits)

## Задача и метрика

Требуется бинарно классифицировать запись: `bonafide` (настоящая речь человека)
или `spoof` (запись, синтезированная TTS или полученная voice conversion).
Партиция LA содержит 19 алгоритмов атак, причём атаки в `eval` (A07-A19) не
пересекаются с атаками в `train`/`dev` (A01-A06), поэтому модель обязана
обобщаться на не виденные типы спуфинга.

| Партиция | Записей | bonafide | spoof  |
| -------- | ------- | -------- | ------ |
| train    | 25 380  | 2 580    | 22 800 |
| dev      | 24 844  | 2 548    | 22 296 |
| eval     | 71 237  | 7 355    | 63 882 |

Основная метрика здесь **EER** (Equal Error Rate): порог, при котором доля
ложных отклонений bonafide равна доле принятых spoof. Скором системы служит
логарифмическое отношение правдоподобий `logits[:, 1] - logits[:, 0]` (чем
больше, тем вероятнее bonafide). Реализация EER в `src/metrics/eer_utils.py` совпадает с официальным
`calculate_eer.py` курса, поэтому локальные числа сравнимы с числами грейдера.

EER не раскладывается по батчам: среднее по-батчевых EER не равно EER корпуса.
Поэтому тренер накапливает скоры всей валидационной партиции и считает метрику
один раз за эпоху. Она логируется как `dev_EER`, и по ней выбирается лучший
чекпоинт.

## Архитектура решения

**Front-end.** По умолчанию считается логарифмическая STFT-спектрограмма
(окно Блэкмана, `n_fft = 1724`, `hop = 130`), она даёт вход `863 × 600`.
Альтернатива: LFCC (20 коэффициентов + Δ + ΔΔ), вход `60 × 750`.
Все записи приводятся к 64 600 отсчётам (≈4 с при 16 кГц): случайная обрезка на
обучении, повтор сигнала при паддинге. Признаки считаются батчем на GPU
(`batch_transforms`), а не в даталоадере, иначе front-end упирается в CPU.

**Модель.** LCNN: 5 свёрточных блоков с активацией Max-Feature-Map (MFM берёт
поэлементный максимум двух половин карт признаков, то есть учит фильтр
конкурировать сам с собой и работает как обучаемый ReLU), 4 max-pooling слоя,
батч-нормализации, затем dropout `0.75` перед финальным полносвязным слоем;
порядок «сначала dropout, потом BatchNorm» взят из статьи STC. Около 10M
параметров.
Варианты голов (`src/model/lcnn_heads.py`): усреднение по времени,
attention-пулинг, BLSTM-суммирование.

**Обучение.** Adam, `lr = 3e-4`, `weight_decay = 1e-4`, StepLR с уменьшением
lr вдвое каждые 10 эпох. Cross-Entropy с весами классов `[1.0, 8.84]`,
компенсирующими дисбаланс train (22 800 spoof против 2 580 bonafide).
Обучение идёт в bfloat16 (`torch.autocast`): вход `863 × 600` делает активации
основным потребителем памяти, а bf16 сохраняет диапазон fp32, поэтому
`GradScaler` не нужен.

## Установка

Нужен Python ≥ 3.11 и CUDA 12.8 (для GPU-сборки PyTorch).

С [uv](https://docs.astral.sh/uv/) (быстрее):

```bash
uv venv --python 3.13
uv pip install -r requirements.txt \
    --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu128
source .venv/bin/activate
```

Либо через pip:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

Для авто-форматирования кода перед коммитом:

```bash
pre-commit install
```

### Настройка трекера экспериментов

Логи пишутся в [Comet ML](https://www.comet.com/). Нужен API-ключ: либо
переменная окружения, либо файл `~/.comet.config`, который создаётся при первом
`comet login` и подхватывается автоматически.

```bash
export COMET_API_KEY=<ваш ключ>
# либо ~/.comet.config:
# [comet]
# api_key = <ваш ключ>
```

Workspace берётся из того же файла; при работе в команде его можно задать явно
через `writer.workspace=<workspace>`. Проект по умолчанию называется
`asvspoof-lcnn`.

Без интернета запускайте с `writer.mode=offline`: эксперимент сложится в
`.cometml-runs/<id>.zip` рядом с запуском и загрузится позже командой
`comet upload <файл>.zip`. Если вместо Comet нужен WandB, добавьте к запуску
`writer=wandb` (потребуется `wandb login`).

## Данные

Скачайте архив LA (≈7.6 ГБ) и распакуйте:

```bash
curl -L -o LA.zip "https://datashare.ed.ac.uk/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/download"
unzip LA.zip -d data/
```

Ожидаемая структура (`data_dir` указывает на каталог, содержащий эти папки):

```
LA/
├── ASVspoof2019_LA_train/flac/
├── ASVspoof2019_LA_dev/flac/
├── ASVspoof2019_LA_eval/flac/
└── ASVspoof2019_LA_cm_protocols/
    ├── ASVspoof2019.LA.cm.train.trn.txt
    ├── ASVspoof2019.LA.cm.dev.trl.txt
    └── ASVspoof2019.LA.cm.eval.trl.txt
```

Путь задаётся переменной окружения (иначе берётся значение по умолчанию из
`src/configs/datasets/asvspoof.yaml`):

```bash
export ASVSPOOF_DIR=/path/to/LA/LA
```

Индекс партиции строится по протоколу один раз и кэшируется в `data/`.

## Обучение

Сначала sanity-check на 64 фиксированных записях: пайплайн исправен, если лосс
уходит почти в ноль, а EER на тех же данных обращается в ноль.

```bash
python3 train.py -cn=one_batch
```

Полное обучение на STFT-признаках:

```bash
python3 train.py -cn=lcnn
```

Вариант с LFCC:

```bash
python3 train.py -cn=lcnn_lfcc
```

Любой параметр переопределяется из командной строки:

```bash
python3 train.py -cn=lcnn dataloader.batch_size=16 trainer.n_epochs=50 \
    writer.run_name=lcnn_bs16 trainer.override=True
```

Полезные ключи `trainer`:

| Ключ            | Значение                                                       |
| --------------- | -------------------------------------------------------------- |
| `use_amp`       | обучение в bf16 (по умолчанию `True`)                           |
| `override`      | перезаписать каталог `saved/${writer.run_name}`                 |
| `resume_from`   | продолжить обучение с чекпоинта внутри каталога запуска         |
| `from_pretrained` | инициализировать веса из произвольного `.pth`                 |
| `monitor`       | метрика для лучшего чекпоинта, по умолчанию `min dev_EER`       |
| `early_stop`    | сколько эпох без улучшения ждать до остановки                   |
| `cudnn_benchmark` | автоподбор алгоритмов свёрток, по умолчанию `False`           |

Про `cudnn_benchmark` отдельно: размер входа здесь фиксирован (863×600), поэтому
автоподбор алгоритмов cuDNN окупается: замер даёт ускорение в 2.1 раза
(с 70 до 145 утт/с). Платить приходится воспроизводимостью: алгоритм выбирается
по замерам времени и может отличаться от запуска к запуску, так что при одном
и том же `trainer.seed` результаты совпадают лишь приблизительно. По умолчанию стоит
`False` (`cudnn.deterministic=True`), для ускорения:

```bash
python3 train.py -cn=lcnn trainer.cudnn_benchmark=True
```

Тот же ключ есть у `inferencer` в `inference.yaml`.

Длина окна, до которой доводится каждая запись в батче, задаётся одним
параметром `collate_max_len` (по умолчанию 64600 отсчётов, ~4.04 с при 16 кГц):
на него ссылается и `max_len` датасетов (обрезка при чтении файла), и
`collate_fn`. Меняя его, не забудьте про `model.in_frames`.

Чекпоинты и конфиг запуска пишутся в `saved/${writer.run_name}/`, лучший из них
`model_best.pth`. В Comet ML логируются лосс на train/dev, `dev_EER`, learning
rate, норма градиента и гистограммы скоров отдельно для bonafide и spoof: их
расхождение наглядно показывает, насколько классы разделимы.

## Инференс и сабмит

Инференс на eval-партиции с лучшим чекпоинтом:

```bash
python3 inference.py inferencer.from_pretrained=saved/lcnn_stft/model_best.pth
```

Результат сохраняется в `data/saved/eval/`:

- `eval_scores.csv` в сабмит-формате: без заголовка, `utterance_id,score`;
- `eval_outputs.pth` с логитами, скорами и метками всей партиции одним файлом
  (для анализа распределений и фьюжна нескольких систем).

Если у партиции есть разметка, EER печатается сразу после инференса.

Перед отправкой файл проверяется скриптом: он ловит всё, на чём падает или
занижает оценку официальный `grading.py` (пропущенные utterance_id, дубликаты,
NaN, лишние колонки), печатает EER и ожидаемую оценку, после чего копирует
результат под нужным именем:

```bash
python3 scripts/make_submission.py data/saved/eval/eval_scores.csv
```

```
protocol: 71237 trials from ASVspoof2019.LA.cm.eval.trl.txt
scores:   71237 unique ids from data/saved/eval/eval_scores.csv

bonafide trials: 7355, spoof trials: 63882
EER: 5.1234%
expected performance grade: 10.00 / 10
submission saved to /path/to/mppanin.csv
```

Путь к протоколу задаётся флагом `-p` или переменной `ASVSPOOF_EVAL_PROTOCOL`,
а имя выходного файла флагом `-o`.

## Структура репозитория

```
├── train.py                  # точка входа обучения
├── inference.py              # точка входа инференса
├── scripts/make_submission.py# проверка и подготовка сабмита
├── src/
│   ├── configs/              # Hydra-конфиги
│   │   ├── lcnn.yaml         # основной конфиг обучения (STFT)
│   │   ├── lcnn_lfcc.yaml    # то же с LFCC front-end
│   │   ├── one_batch.yaml    # sanity-check на одном батче
│   │   ├── inference.yaml    # конфиг инференса
│   │   └── {model,datasets,dataloader,metrics,transforms,writer}/
│   ├── datasets/             # ASVspoofDataset, индекс по CM-протоколам, collate
│   ├── model/                # LCNN, MFM, варианты голов
│   ├── transforms/           # STFT/LFCC front-end, аугментации, нормализация
│   ├── loss/                 # CE и margin-based лоссы
│   ├── metrics/              # EER (официальная реализация), accuracy
│   ├── trainer/              # Trainer и Inferencer
│   ├── logger/               # Comet ML / WandB
│   └── utils/                # инициализация, seed, io
└── requirements.txt
```

## Credits

Репозиторий основан на
[PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template)
(MIT License). Расчёт EER взят из официального evaluation-пакета ASVspoof2019.

## License

[MIT](LICENSE)
