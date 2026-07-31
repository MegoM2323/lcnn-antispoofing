class MetricTracker:
    """
    Class to aggregate metrics from many batches.

    The state is kept in plain dicts instead of a pandas DataFrame: since
    pandas 3.0 copy-on-write makes ``DataFrame[col].values`` a read-only view,
    which broke the in-place reset of the original implementation. Dicts also
    keep the logged values as python floats instead of numpy scalars.
    """

    def __init__(self, *keys: str, writer=None):
        """
        Args:
            *keys (list[str]): list (as positional arguments) of metric
                names (may include the names of losses)
            writer (WandBWriter | CometMLWriter | None): experiment tracker.
                Not used in this code version. Can be used to log metrics
                from each batch.
        """
        self.writer = writer
        self._keys: tuple[str, ...] = tuple(keys)
        self._total: dict[str, float] = {}
        self._counts: dict[str, float] = {}
        self._average: dict[str, float] = {}
        self.reset()

    def reset(self) -> None:
        """
        Reset all metrics after epoch end.
        """
        for key in self._keys:
            self._total[key] = 0.0
            self._counts[key] = 0.0
            self._average[key] = 0.0

    def update(self, key: str, value: float, n: int = 1) -> None:
        """
        Update metrics with new value.

        Args:
            key (str): metric name.
            value (float): metric value on the batch.
            n (int): how many times to count this value.
        """
        # if self.writer is not None:
        #     self.writer.add_scalar(key, value)
        if key not in self._total:
            raise KeyError(f"Metric '{key}' is not defined in the MetricTracker")
        self._total[key] += float(value) * n
        self._counts[key] += n
        self._average[key] = self._total[key] / self._counts[key]

    def avg(self, key: str) -> float:
        """
        Return average value for a given metric.

        Args:
            key (str): metric name.
        Returns:
            average_value (float): average value for the metric. Zero if the
                metric has not been updated since the last reset.
        """
        return self._average[key]

    def result(self) -> dict[str, float]:
        """
        Return average value of each metric.

        Returns:
            average_metrics (dict): dict, containing average metrics
                for each metric name.
        """
        return dict(self._average)

    def keys(self) -> list[str]:
        """
        Return all metric names defined in the MetricTracker.

        Returns:
            metric_keys (list[str]): all metric names in the tracker.
        """
        return list(self._keys)
