"""
Контрмера на базе Light CNN: девять свёрточных блоков с активациями
Max-Feature-Map и полносвязная голова поверх всей карты признаков.
"""

import torch
from torch import nn

from src.model.mfm import MFM


def init_weights(module: nn.Module) -> None:
    """
    Инициализация Kaiming normal для свёрточных и линейных слоёв
    (arXiv:1904.05576, разд. 2.3). Смещения обнуляются.
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class LCNNBackbone(nn.Module):
    """
    Свёрточная часть Light CNN (слои 1-28 таблицы 1 из arXiv:1904.05576).

    Блоки построены как Conv -> MFM -> (MaxPool) -> (BatchNorm); сеть целиком
    свёрточная, поэтому выходное разрешение зависит от входного только через
    четыре слоя max-pooling 2x2.
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()

        # Номера слоёв ниже это номера строк таблицы 1 из arXiv:1904.05576,
        # так что блок читается прямо по статье.
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, stride=1, padding=2),  # Conv 1
            MFM(32),  # MFM 2
            nn.MaxPool2d(kernel_size=2, stride=2),  # MaxPool 3
            nn.Conv2d(32, 64, kernel_size=1, stride=1),  # Conv 4
            MFM(32),  # MFM 5
            nn.BatchNorm2d(32),  # BatchNorm 6
            nn.Conv2d(32, 96, kernel_size=3, stride=1, padding=1),  # Conv 7
            MFM(48),  # MFM 8
            nn.MaxPool2d(kernel_size=2, stride=2),  # MaxPool 9
            nn.BatchNorm2d(48),  # BatchNorm 10
            nn.Conv2d(48, 96, kernel_size=1, stride=1),  # Conv 11
            MFM(48),  # MFM 12
            nn.BatchNorm2d(48),  # BatchNorm 13
            nn.Conv2d(48, 128, kernel_size=3, stride=1, padding=1),  # Conv 14
            MFM(64),  # MFM 15
            nn.MaxPool2d(kernel_size=2, stride=2),  # MaxPool 16
            nn.Conv2d(64, 128, kernel_size=1, stride=1),  # Conv 17
            MFM(64),  # MFM 18
            nn.BatchNorm2d(64),  # BatchNorm 19
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # Conv 20
            MFM(32),  # MFM 21
            nn.BatchNorm2d(32),  # BatchNorm 22
            nn.Conv2d(32, 64, kernel_size=1, stride=1),  # Conv 23
            MFM(32),  # MFM 24
            nn.BatchNorm2d(32),  # BatchNorm 25
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Conv 26
            MFM(32),  # MFM 27
            nn.MaxPool2d(kernel_size=2, stride=2),  # MaxPool 28
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LCNN(nn.Module):
    """
    Light CNN для голосового антиспуфинга, в точности как в таблице 1 из
    arXiv:1904.05576: карта признаков backbone вытягивается в вектор и проходит
    FC_29 -> MFM_30 -> Dropout -> BatchNorm_31 -> FC_32.
    """

    def __init__(
        self,
        in_freq: int = 863,
        in_frames: int = 600,
        n_class: int = 2,
        dropout: float = 0.75,
        embedding_dim: int = 160,
    ):
        """
        Аргументы:
            in_freq (int): число частотных бинов входного признака.
            in_frames (int): число временных фреймов входного признака.
            n_class (int): число классов (2 для bonafide/spoof).
            dropout (float): вероятность dropout перед финальным BatchNorm,
                как требует условие задания.
            embedding_dim (int): число выходных признаков FC_29. Итоговый
                эмбеддинг вдвое меньше из-за MFM_30.
        """
        super().__init__()

        self.backbone = LCNNBackbone()
        pooled_dim = self._flattened_dim(in_freq, in_frames)

        # Строки 29-32 таблицы 1 из arXiv:1904.05576; dropout вставлен там, где
        # его требует условие задания (в самой таблице 1 строки с ним нет).
        self.fc = nn.Linear(pooled_dim, embedding_dim)  # FC 29
        self.mfm_fc = MFM(embedding_dim // 2)  # MFM 30
        self.dropout = nn.Dropout(p=dropout)  # Dropout, перед финальным BatchNorm
        self.bn = nn.BatchNorm1d(embedding_dim // 2)  # BatchNorm 31
        self.classifier = nn.Linear(embedding_dim // 2, n_class)  # FC 32

        self.apply(init_weights)

    def _flattened_dim(self, in_freq: int, in_frames: int) -> int:
        """
        Число признаков на входе FC_29, измеренное прогоном фиктивного тензора
        через backbone.

        Реальный прогон вместо формулы даёт верный размер головы для любого
        фронт-энда и, в отличие от ленивых слоёв, известен до первого forward,
        поэтому чекпоинт можно загрузить сразу после создания модели.
        """
        was_training = self.backbone.training
        self.backbone.eval()
        with torch.no_grad():
            shape = self.backbone(torch.zeros(1, 1, in_freq, in_frames)).shape
        self.backbone.train(was_training)

        return int(shape[1] * shape[2] * shape[3])

    def forward(self, data_object: torch.Tensor, **batch) -> dict:
        """
        Превращает входной признак, (B, F, T) или (B, 1, F, T), в логиты двух
        классов.
        """
        if data_object.dim() == 3:
            data_object = data_object.unsqueeze(1)

        features = self.backbone(data_object)
        embedding = self.mfm_fc(self.fc(torch.flatten(features, start_dim=1)))
        embedding = self.bn(self.dropout(embedding))

        return {"logits": self.classifier(embedding)}

    def __str__(self):
        """
        Модель печатается вместе с числом параметров.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
