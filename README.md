# Voice Anti-Spoofing: LCNN на ASVspoof2019 LA

Детекция синтезированной и преобразованной речи на партиции Logical Access
[ASVspoof2019](https://datashare.ed.ac.uk/handle/10283/3336). Модель: Light CNN
([arXiv:1904.05576](https://arxiv.org/abs/1904.05576)) с рецептом обучения из
[arXiv:2103.11326](https://arxiv.org/abs/2103.11326).

Итог: **EER 3,0704 %** на полной eval-партиции (71 237 записей). Результат
получен фьюжном пяти наборов скоров: четыре чекпоинта LFCC-модели и один
чекпоинт модели на спектрограмме. Логи обучения:
[comet.com/a-ern/asvspoof-lcnn](https://www.comet.com/a-ern/asvspoof-lcnn).

## Установка

Нужен Python 3.12 и CUDA 12.8 для GPU-сборки PyTorch.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

Логи пишутся в [Comet ML](https://www.comet.com/), поэтому нужен `COMET_API_KEY`
в окружении или `~/.comet.config`. Без интернета работает `writer.mode=offline`.

## Данные

```bash
curl -L -o LA.zip "https://datashare.ed.ac.uk/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/download"
unzip LA.zip -d data/
export ASVSPOOF_DIR=/path/to/LA/LA
export ASVSPOOF_EVAL_PROTOCOL=$ASVSPOOF_DIR/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt
```

`ASVSPOOF_DIR` указывает на каталог с `ASVspoof2019_LA_train/` и остальными
партициями, `ASVSPOOF_EVAL_PROTOCOL` задаёт протокол для проверки сабмита.

## Обучение

Сначала sanity-check на 64 записях, лосс должен уйти почти в ноль:

```bash
python3 train.py -cn=one_batch
```

Команды, которыми получен итоговый результат:

```bash
# модель на лог-спектрограмме (вход 863 x 600)
python3 train.py -cn=lcnn writer.run_name=lcnn_stft_600f trainer.seed=1 \
    trainer.cudnn_benchmark=True dataloader.batch_size=32 \
    dataloader.num_workers=6 trainer.n_epochs=30

# модель на LFCC (вход 60 x 750)
python3 train.py -cn=lcnn_lfcc writer.run_name=lcnn_lfcc_seed1 trainer.seed=1 \
    trainer.cudnn_benchmark=True dataloader.batch_size=32 \
    dataloader.num_workers=5 trainer.n_epochs=30 trainer.save_period=3
```

Чекпоинты и конфиг прогона пишутся в `saved/${writer.run_name}/`. С
`cudnn_benchmark=True` обучение быстрее в 2,1 раза, но прогоны перестают
повторяться побитово: при том же сиде веса получатся близкие, но не идентичные.

## Предсказания

Готовые веса лежат в релизе
[v1.0](https://github.com/MegoM2323/lcnn-antispoofing/releases/tag/v1.0), конфиг
обучающего прогона хранится внутри самого `.pth`.

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

Один чекпоинт по всей eval-партиции:

```bash
python3 scripts/predict_eval.py checkpoints/lfcc_epoch21.pth -o mppanin.csv
```

Скрипт берёт входной тракт из конфига обучающего прогона, проверяет, что
предсказана каждая запись протокола, и печатает EER. Флаги: `-b` (размер
батча), `-d` (устройство), `--save-dir` (куда положить сырые скоры).

Итоговый файл собирается из пяти прогонов:

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

`--save-dir` здесь обязателен, иначе прогоны затрут скоры друг друга. Скоры
разных систем несопоставимы по масштабу, поэтому перед усреднением заменяются
на ранги. Проверить уже готовый csv отдельно можно скриптом
`scripts/make_submission.py`.

## Результаты

Все числа посчитаны на полной eval-партиции той же реализацией EER, что
у грейдера.

| Система | EER, % |
| --- | --- |
| LFCC, эпоха 21 | 3,6978 |
| LFCC, лучшая эпоха по `dev_EER` | 3,6678 |
| спектрограмма, эпоха 15 | 5,5222 |
| фьюжн четырёх LFCC | 3,6571 |
| фьюжн всех пяти наборов скоров | **3,0704** |

Модель на спектрограмме слабее LFCC почти на два пункта, хотя параметров в ней
в 12 раз больше. Объединение эпох одного прогона почти ничего не даёт, основной
выигрыш приносит второй front-end.

## Credits

Проект собран на шаблоне
[PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template)
(MIT License). Расчёт EER взят из официального evaluation-пакета ASVspoof2019.
Лицензия: [MIT](LICENSE).
