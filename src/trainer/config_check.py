"""
Проверка того, что чекпоинт загружают с тем же входным конвейером, с которым
его обучали.

Метод 'load_state_dict' сверяет только формы весов, поэтому чужая длина
сигнала, окно или шаг принимаются молча: замер на обученной модели показал,
что один лишь чужой 'collate_max_len' сдвигает эвалюационный EER на 0,33
пункта. Поэтому фронт-энд чекпоинта сравнивается с текущим, и о каждом
расхождении сообщается.
"""

from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

# настройки фронт-энда, которые меняют вход модели, не меняя формы её весов
FRONTEND_KEYS = ("n_fft", "win_length", "hop_length", "window", "n_frames", "crop")

FRONTEND_PATH = ("transforms", "batch_transforms", "inference")


def _collect(node: Any, params: dict[str, Any]) -> None:
    """
    Поиск FRONTEND_KEYS в описании трансформа в глубину: фронт-энд это
    Sequential со списком подмодулей, поэтому обойти нужно всё поддерево.
    Аккумулятор 'params' меняется на месте.
    """
    if isinstance(node, (DictConfig, ListConfig)):
        node = OmegaConf.to_container(node, resolve=False)

    if isinstance(node, Mapping):
        for key, value in node.items():
            if key in FRONTEND_KEYS and not isinstance(value, (Mapping, list)):
                params.setdefault(key, value)
            else:
                _collect(value, params)
    elif isinstance(node, list):
        for item in node:
            _collect(item, params)


def frontend_params(config: Mapping | None) -> dict[str, Any]:
    """
    Собирает всё, что определяет вход модели: длину, к которой приводится
    каждый сигнал, и параметры фронт-энда инференса.
    """
    if not isinstance(config, Mapping):
        return {}

    params: dict[str, Any] = {"collate_max_len": config.get("collate_max_len")}
    node: Any = config
    for key in FRONTEND_PATH:
        node = node.get(key) if isinstance(node, Mapping) else None
    _collect(node, params)
    return params


def config_mismatches(
    saved_config: Mapping | None, current_config: Mapping | None
) -> list[str]:
    """
    Сравнивает входные конвейеры двух запусков, по строке на каждый
    разошедшийся параметр.
    """
    saved = frontend_params(saved_config)
    current = frontend_params(current_config)
    return [
        f"{key}: trained with {saved.get(key)}, now {current.get(key)}"
        for key in sorted(set(saved) | set(current))
        if saved.get(key) != current.get(key)
    ]


def format_mismatch_warning(mismatches: list[str], checkpoint_path: str) -> str:
    """
    Предупреждение, которое печатается при расхождении двух конвейеров.
    """
    return (
        f"'{checkpoint_path}' was trained with another input pipeline "
        f"({'; '.join(mismatches)}), so its scores are not comparable with the "
        "ones of the training run."
    )
