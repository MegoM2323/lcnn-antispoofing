import csv

import torch
from tqdm.auto import tqdm

from src.metrics.eer_utils import epoch_eer, logits_to_scores
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    """
    Инференсер: то же, что тренер, но для инференса.

    Класс обрабатывает данные без оптимизаторов, логгеров и прочего. Нужен,
    чтобы оценить модель на датасете, сохранить предсказания и так далее.

    Предсказания сохраняются в формате, который ждёт официальный проверяющий
    скрипт: csv без заголовка со строками "utterance_id,score", где скор это
    логарифм отношения правдоподобий класса bonafide.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        save_path,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
    ):
        """
        Аргументы:
            model (nn.Module): модель PyTorch.
            config (DictConfig): конфиг запуска, содержащий конфиг
                инференсера.
            device (str): устройство для тензоров и модели.
            dataloaders (dict[DataLoader]): даталоадеры, которые надо
                прогнать.
            save_path (Path): каталог, куда пишутся предсказания.
            metrics (dict | None): metrics[inference], каждая из них
                экземпляр src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): трансформы, которые
                применяются ко всему батчу, в зависимости от имени тензора.
            skip_model_load (bool): если False, в модель загружается чекпоинт
                из config.inferencer.from_pretrained.
        """
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms

        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}
        self.save_path = save_path

        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

        self._setup_amp()

        # буферы с предсказаниями текущей партиции
        self._utt_ids = []
        self._scores = []
        self._labels = []

        if not skip_model_load:
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        """
        Прогоняет инференс на каждой партиции и возвращает её логи по имени
        партиции.
        """
        return {
            part: self._inference_part(part, dataloader)
            for part, dataloader in self.evaluation_dataloaders.items()
        }

    def move_batch_to_device(self, batch):
        """
        Переносит на устройство все нужные тензоры.

        В отличие от базовой реализации, отсутствующие в батче тензоры молча
        пропускаются: партиции без истинных меток не должны ломать инференс.
        """
        for tensor_for_device in self.cfg_trainer.device_tensors:
            if tensor_for_device in batch:
                batch[tensor_for_device] = batch[tensor_for_device].to(self.device)
        return batch

    def process_batch(self, batch, metrics):
        """
        Прогоняет батч через модель, считает метрики и накапливает
        предсказания. На диск всё пишется один раз на партицию в методе
        '_inference_part': отдельный файл на запись означал бы 71237 файлов
        для эвалюационной партиции.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # трансформы на устройстве, так быстрее

        with self._autocast():
            outputs = self.model(**batch)
            batch.update(outputs)

        if metrics is not None:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))

        self._scores.append(logits_to_scores(batch["logits"]).cpu())
        self._utt_ids.extend(batch["utt_id"])
        if batch.get("labels") is not None:
            self._labels.append(batch["labels"].detach().reshape(-1).cpu())

        return batch

    def _inference_part(self, part, dataloader):
        """
        Прогоняет инференс на заданной партиции, сохраняет предсказания и
        считает EER, если доступны истинные метки.
        """
        self.is_train = False
        self.model.eval()

        if self.evaluation_metrics is not None:
            self.evaluation_metrics.reset()

        self._utt_ids = []
        self._scores = []
        self._labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=part, total=len(dataloader)):
                self.process_batch(batch=batch, metrics=self.evaluation_metrics)

        scores = torch.cat(self._scores) if self._scores else torch.empty(0)
        labels = torch.cat(self._labels) if self._labels else None

        self._save_scores(part, scores)

        logs = (
            self.evaluation_metrics.result()
            if self.evaluation_metrics is not None
            else {}
        )
        eer = epoch_eer(scores, labels, warn=print)
        if eer is not None:
            logs["EER"] = eer
            print(f"{part} EER: {eer:.4f}%")

        return logs

    def _save_scores(self, part, scores):
        """
        Пишет '{part}_scores.csv' в формате посылки: без заголовка,
        "utterance_id,score", по строке на каждую запись партиции.
        """
        if self.save_path is None:
            return

        self.save_path.mkdir(exist_ok=True, parents=True)
        csv_path = self.save_path / f"{part}_scores.csv"
        try:
            with csv_path.open("w", newline="") as file:
                writer = csv.writer(file)
                for utt_id, score in zip(self._utt_ids, scores.tolist()):
                    writer.writerow([utt_id, repr(float(score))])
        except OSError as e:
            print(f"Failed to write scores to {csv_path}: {e}")
        else:
            print(f"Saved {scores.numel()} scores to {csv_path}")
