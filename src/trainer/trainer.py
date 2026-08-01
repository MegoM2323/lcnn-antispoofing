import torch
from tqdm.auto import tqdm

from src.metrics.eer_utils import epoch_eer, logits_to_scores
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer

SCORE_HIST_BINS = 64


class Trainer(BaseTrainer):
    """
    Тренер антиспуфинг-контрмеры.

    Добавляет к базовому тренеру две вещи:

    1. Смешанную точность вокруг прямого прохода, см. BaseTrainer._setup_amp.
    2. Корректный EER на уровне эпохи. EER это свойство всего распределения
       скоров, и по батчам он не раскладывается: среднее побатчевых EER не
       равно EER партиции. Поэтому скоры всей эвалюационной партиции
       накапливаются, а EER считается один раз за эпоху, ровно так же, как
       это делает официальный проверяющий скрипт.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._setup_amp()

        # буферы для EER на уровне эпохи (заполняются только при оценке)
        self._epoch_scores = []
        self._epoch_labels = []

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Прогоняет батч через модель, считает метрики и лосс, а на стадии
        обучения делает шаг оптимизации.

        Функция рассчитывает, что criterion сводит все лоссы (если их
        несколько) к одному, лежащему по ключу 'loss'.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # трансформы на устройстве, так быстрее

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        with self._autocast():
            outputs = self.model(**batch)
            batch.update(outputs)

            all_losses = self.criterion(**batch)
            batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()  # сумма всех лоссов всегда лежит в loss
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        else:
            self._accumulate_scores(batch)

        # обновление метрик по каждому лоссу (на случай нескольких лоссов)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Оценивает модель на партиции после эпохи обучения.

        Повторяет логику базового метода и дополнительно считает EER по всей
        партиции (см. докстринг класса). Значение логируется как скаляр "EER"
        и кладётся в возвращаемые логи, поэтому в общих логах оно появляется
        как "{part}_EER" и годится для мониторинга.
        """
        self.is_train = False
        self.model.eval()
        self.evaluation_metrics.reset()
        self._epoch_scores = []
        self._epoch_labels = []

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    metrics=self.evaluation_metrics,
                )
            self.writer.set_step(epoch * self.epoch_len, part)
            self._log_scalars(self.evaluation_metrics)
            self._log_batch(
                batch_idx, batch, part
            )  # при инференсе логируется только последний батч

        logs = self.evaluation_metrics.result()

        scores, labels = self._collected_scores()
        eer = epoch_eer(scores, labels, warn=self.logger.warning)
        if eer is not None:
            logs["EER"] = eer
            if self.writer is not None:
                self.writer.add_scalar("EER", eer)

        return logs

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Логирует данные батча, вызывая self.writer.add_* для отправки их
        в трекер экспериментов.

        Для эвалюационных партиций распределения скоров двух классов
        логируются гистограммами: их перекрытие и есть то, что измеряет EER,
        так что график показывает, насколько классы разделимы.
        """
        if mode == "train" or self.writer is None:
            # на train-партиции ничего тяжёлого: метод вызывается каждые
            # log_step шагов, а картинки и гистограммы замедляют обучение
            return

        scores, labels = self._collected_scores()
        if scores is None:
            return

        bonafide_scores = scores[labels == 1]
        spoof_scores = scores[labels == 0]
        if bonafide_scores.numel() > 0:
            self.writer.add_histogram(
                "scores_bonafide", bonafide_scores, bins=SCORE_HIST_BINS
            )
        if spoof_scores.numel() > 0:
            self.writer.add_histogram(
                "scores_spoof", spoof_scores, bins=SCORE_HIST_BINS
            )

    def _accumulate_scores(self, batch):
        """
        Сохраняет детекционные скоры и метки батча для EER на уровне эпохи.
        """
        self._epoch_scores.append(logits_to_scores(batch["logits"]).cpu())
        self._epoch_labels.append(batch["labels"].detach().reshape(-1).cpu())

    def _collected_scores(self):
        """
        Склеивает скоры и метки, накопленные за эпоху; (None, None), если
        накопить ничего не успели.
        """
        if not self._epoch_scores:
            return None, None
        return torch.cat(self._epoch_scores), torch.cat(self._epoch_labels)
