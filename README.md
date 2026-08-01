# Voice Anti-Spoofing: LCNN на ASVspoof2019 LA

Детекция синтезированной и преобразованной речи на партиции Logical Access
[ASVspoof2019](https://datashare.ed.ac.uk/handle/10283/3336). Модель — Light CNN
([arXiv:1904.05576](https://arxiv.org/abs/1904.05576)), рецепт обучения из
[arXiv:2103.11326](https://arxiv.org/abs/2103.11326), шаблон проекта —
[Blinorot/pytorch_project_template](https://github.com/Blinorot/pytorch_project_template).

**Итог: EER 3,0704 % на полной eval-партиции (71 237 записей), 10 / 10 по
официальному `grading.py`** — фьюжн пяти наборов скоров: четыре чекпоинта
LFCC-модели и один модели на спектрограмме. Логи обучения:
[comet.com/a-ern/asvspoof-lcnn](https://www.comet.com/a-ern/asvspoof-lcnn).

## Решение

Скор системы — `logits[:, 1] - logits[:, 0]`. Реализация EER в
`src/metrics/eer_utils.py` взята из официального evaluation-пакета, поэтому
локальные числа сравнимы с числами грейдера; по батчам EER не раскладывается, и
метрика считается раз в эпоху по всей валидации (`dev_EER`). Два front-end, оба
участвуют в итоге:

| Front-end | Конфиг | Вход модели | Параметров |
| --- | --- | --- | --- |
| лог-спектрограмма (окно Блэкмана, `n_fft = 1724`, `hop = 130`) | `-cn=lcnn` | 863 × 600 | 10 198 818 |
| LFCC (20 коэффициентов + Δ + ΔΔ) | `-cn=lcnn_lfcc` | 60 × 750 | 865 058 |

Записи приводятся к 77 870 отсчётам (4,87 с): случайная обрезка на обучении,
циклический повтор при паддинге — тишина в конце работает как ложная подсказка.
Модель: 9 свёрточных слоёв с Max-Feature-Map, 4 max-pooling, dropout 0.75.
Обучение: Adam, StepLR, Cross-Entropy с весами классов `[1.0, 8.84]` под
дисбаланс train, bfloat16-autocast.

## Установка

Python 3.12, CUDA 12.8 для GPU-сборки PyTorch.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

Логи пишутся в [Comet ML](https://www.comet.com/): нужен `COMET_API_KEY` в
окружении или `~/.comet.config` (без интернета — `writer.mode=offline`).

## Данные

```bash
curl -L -o LA.zip "https://datashare.ed.ac.uk/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/download"
unzip LA.zip -d data/
export ASVSPOOF_DIR=/path/to/LA/LA
export ASVSPOOF_EVAL_PROTOCOL=$ASVSPOOF_DIR/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt
```

Первая переменная указывает на каталог с `ASVspoof2019_LA_train/` и остальными,
вторая — на eval-протокол, по которому проверяется полнота сабмита.

## Готовые веса

Чекпоинты не входят в репозиторий. Пять файлов, из которых собран итог,
выложены релизом
[v1.0](https://github.com/MegoM2323/lcnn-antispoofing/releases/tag/v1.0); конфиг
обучающего прогона лежит внутри самого `.pth`, поэтому скачанный файл работает
сам по себе.

```bash
mkdir -p checkpoints && cd checkpoints
base=https://github.com/MegoM2323/lcnn-antispoofing/releases/download/v1.0
for f in lfcc_epoch12.pth lfcc_epoch15.pth lfcc_epoch18.pth lfcc_epoch21.pth \
         stft_epoch15.pth SHA256SUMS.txt; do
  curl -L -o $f $base/$f
done
sha256sum -c SHA256SUMS.txt
cd ..
```

## Обучение

Sanity-check на 64 фиксированных записях (лосс должен уйти почти в ноль):

```bash
python3 train.py -cn=one_batch
```

Команды, которыми получен итоговый результат:

```bash
# модель на спектрограмме, прогон lcnn_stft_600f
python3 train.py -cn=lcnn writer.run_name=lcnn_stft_600f trainer.seed=1 \
    trainer.cudnn_benchmark=True dataloader.batch_size=32 \
    dataloader.num_workers=6 trainer.n_epochs=30

# модель на LFCC, прогон lcnn_lfcc_seed1
python3 train.py -cn=lcnn_lfcc writer.run_name=lcnn_lfcc_seed1 trainer.seed=1 \
    trainer.cudnn_benchmark=True dataloader.batch_size=32 \
    dataloader.num_workers=5 trainer.n_epochs=30 trainer.save_period=3
```

`trainer.save_period=3` во второй команде обязателен: по умолчанию он равен 5,
и тогда чекпоинтов эпох 12, 18 и 21, из которых собран итог, не существует.
`cudnn_benchmark=True` ускоряет обучение в 2,1 раза, но алгоритм свёртки
выбирается по замерам времени, поэтому **оба финальных прогона побитово не
повторяются**: при том же сиде веса получатся близкие, но не идентичные (для
точной воспроизводимости оставьте `False`). Чекпоинты и конфиг прогона пишутся
в `saved/${writer.run_name}/`.

## Предсказания и сабмит

`scripts/predict_eval.py` прогоняет чекпоинт по всей eval-партиции, проверяет,
что предсказана каждая запись протокола, и пишет сабмит:

```bash
python3 scripts/predict_eval.py checkpoints/lfcc_epoch21.pth -o mppanin.csv
```

Входной тракт берётся из конфига обучающего прогона, а не из конфигов проекта:
`load_state_dict` сверяет только формы весов, поэтому чужой front-end загрузился
бы молча, а посторонний `collate_max_len` сдвигает eval EER на 0,33 пункта.
Скрипт печатает итоговый EER, ожидаемую оценку и EER каждой атаки; из флагов
есть `--save-dir`, `-b` (размер батча) и `-d` (устройство).
Шаблонный `inference.py` тоже работает, но берёт front-end из
конфигов проекта, поэтому для LFCC его надо задавать явно
(`transforms=lfcc model.in_freq=60 model.in_frames=750`).

Готовый csv проверяется отдельно — скрипт ловит всё, на чём падает или занижает
оценку `grading.py` (пропущенные id, дубликаты, NaN, лишние колонки):

```bash
python3 scripts/make_submission.py data/saved/lfcc21/eval_scores.csv -o mppanin.csv
```

### Итоговый файл

Пять прогонов и фьюжн; последовательность даёт побайтово тот же `mppanin.csv`,
что был отправлен:

```bash
for e in 12 15 18 21; do
  python3 scripts/predict_eval.py checkpoints/lfcc_epoch$e.pth \
      --save-dir data/saved/lfcc$e -o data/saved/lfcc$e.csv
done
python3 scripts/predict_eval.py checkpoints/stft_epoch15.pth \
    --save-dir data/saved/stft15 -o data/saved/stft15.csv
python3 scripts/fuse_scores.py \
    data/saved/{lfcc12,lfcc15,lfcc18,lfcc21,stft15}/eval_scores.csv -o mppanin.csv
```

`--save-dir` обязателен: по умолчанию все прогоны пишут сырые предсказания в
один каталог и затирают друг друга. Размер батча (`-b`) на скоры не влияет,
только на скорость и потребление памяти. Скоры разных систем несопоставимы по
масштабу, поэтому перед усреднением заменяются на ранги — нормировка монотонная
и EER отдельной системы не меняет.

## Результаты

Все числа — на полной eval-партиции той же реализацией EER, что у грейдера.

| Система | Чекпоинт | EER, % |
| --- | --- | --- |
| LFCC (`lcnn_lfcc_seed1`) | эпохи 12 / 15 / 18 / 21 | 4,0653 / 4,0922 / 4,1052 / 3,6978 |
| LFCC (`lcnn_lfcc_seed1`) | лучший по `dev_EER` | 3,6678 |
| спектрограмма (`lcnn_stft_600f`) | эпоха 15 | 5,5222 |
| фьюжн четырёх LFCC | — | 3,6571 |
| фьюжн LFCC + спектрограмма | — | 3,4672 |
| фьюжн всех пяти | — | **3,0704** |

Модель на спектрограмме слабее LFCC почти на два пункта, хотя параметров в ней
в 12 раз больше. Объединение эпох одного прогона почти ничего не даёт —
чекпоинты одной модели ошибаются на одних записях, — а второй front-end даёт
основной выигрыш.

## Тесты

Фикстуры собирают во временном каталоге синтетический LA, настоящий корпус не
нужен. Покрыты расчёт EER и разбивка по атакам, формат сабмита и фьюжн скоров,
датасет и приведение записей к фиксированной длине, front-end, forward модели и
загрузка чекпоинта.

```bash
python3 -m pytest tests/ -q
black --check . && isort --profile black --check-only . && flake8 .
```

## Структура

- `train.py`, `inference.py` — точки входа шаблона;
- `scripts/` — `predict_eval`, `fuse_scores`, `make_submission`;
- `src/configs/` — Hydra-конфиги: `lcnn`, `lcnn_lfcc`, `one_batch`, `inference`;
- `src/datasets/`, `src/transforms/` — датасет по CM-протоколу, collate,
  STFT / LFCC front-end;
- `src/model/`, `src/loss/`, `src/metrics/` — LCNN с Max-Feature-Map,
  взвешенная кросс-энтропия, EER и фьюжн скоров;
- `src/trainer/`, `src/logger/`, `src/utils/` — Trainer, Inferencer, Comet ML;
- `tests/` — pytest на синтетическом корпусе.

## Credits

Репозиторий основан на
[PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template)
(MIT License). Расчёт EER взят из официального evaluation-пакета ASVspoof2019.

## License

[MIT](LICENSE)
