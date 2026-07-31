from abc import abstractmethod
from contextlib import nullcontext

import torch
from numpy import inf
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from src.datasets.data_utils import inf_loop
from src.metrics.tracker import MetricTracker
from src.trainer.config_check import config_mismatches, format_mismatch_warning
from src.utils.io_utils import ROOT_PATH


class BaseTrainer:
    """
    Base class for all trainers.
    """

    def __init__(
        self,
        model,
        criterion,
        metrics,
        optimizer,
        lr_scheduler,
        config,
        device,
        dataloaders,
        logger,
        writer,
        epoch_len=None,
        skip_oom=True,
        batch_transforms=None,
    ):
        """
        Args:
            model (nn.Module): PyTorch model.
            criterion (nn.Module): loss function for model training.
            metrics (dict): dict with the definition of metrics for training
                (metrics[train]) and inference (metrics[inference]). Each
                metric is an instance of src.metrics.BaseMetric.
            optimizer (Optimizer): optimizer for the model.
            lr_scheduler (LRScheduler): learning rate scheduler for the
                optimizer.
            config (DictConfig): experiment config containing training config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            logger (Logger): logger that logs output.
            writer (WandBWriter | CometMLWriter): experiment tracker.
            epoch_len (int | None): number of steps in each epoch for
                iteration-based training. If None, use epoch-based
                training (len(dataloader)).
            skip_oom (bool): skip batches with the OutOfMemory error.
            batch_transforms (dict[Callable] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
        """
        self.is_train = True

        self.config = config
        self.cfg_trainer = self.config.trainer

        self.device = device
        self.skip_oom = skip_oom

        self.logger = logger
        self.log_step = config.trainer.get("log_step", 50)

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.batch_transforms = batch_transforms

        # define dataloaders
        self.train_dataloader = dataloaders["train"]
        if epoch_len is None:
            # epoch-based training
            self.epoch_len = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.epoch_len = epoch_len

        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"
        }

        # define epochs
        self._last_epoch = 0  # required for saving on interruption
        self.start_epoch = 1
        self.epochs = self.cfg_trainer.n_epochs

        # configuration to monitor model performance and save best

        self.save_period = (
            self.cfg_trainer.save_period
        )  # checkpoint each save_period epochs
        self.monitor = self.cfg_trainer.get(
            "monitor", "off"
        )  # format: "mnt_mode mnt_metric"

        self._setup_monitoring()

        # setup visualization writer instance
        self.writer = writer

        # define metrics
        self.metrics = metrics
        self.train_metrics = MetricTracker(
            *self.config.writer.loss_names,
            "grad_norm",
            *[m.name for m in self.metrics["train"]],
            writer=self.writer,
        )
        self.evaluation_metrics = MetricTracker(
            *self.config.writer.loss_names,
            *[m.name for m in self.metrics["inference"]],
            writer=self.writer,
        )

        # define checkpoint dir and init everything if required

        self.checkpoint_dir = (
            ROOT_PATH / config.trainer.save_dir / config.writer.run_name
        )

        if config.trainer.get("resume_from") is not None:
            resume_path = self.checkpoint_dir / config.trainer.resume_from
            self._resume_checkpoint(resume_path)

        if config.trainer.get("from_pretrained") is not None:
            self._from_pretrained(config.trainer.get("from_pretrained"))

    def train(self):
        """
        Wrapper around training process to save model on keyboard interrupt.
        """
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info("Saving model on keyboard interrupt")
            self._save_checkpoint(self._last_epoch, save_best=False)
            raise e

    def _train_process(self):
        """
        Full training logic:

        Training model for an epoch, evaluating it on non-train partitions,
        and monitoring the performance improvement (for early stopping
        and saving the best checkpoint).
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            result = self._train_epoch(epoch)

            # save logged information into logs dict
            logs = {"epoch": epoch}
            logs.update(result)

            # print logged information to the screen
            for key, value in logs.items():
                self.logger.info(f"    {key:15s}: {value}")

            # evaluate model performance according to configured metric,
            # save best checkpoint as model_best
            best, stop_process, not_improved_count = self._monitor_performance(
                logs, not_improved_count
            )

            # A periodic checkpoint is kept even when the epoch is the best one:
            # on ASVspoof2019 LA the dev EER saturates at 0.0 and almost every
            # epoch counts as "best", so only_best=True would leave a single
            # overwritten model_best.pth and no history to fall back on.
            periodic = epoch % self.save_period == 0
            if periodic or best:
                self._save_checkpoint(epoch, save_best=best, only_best=not periodic)

            if stop_process:  # early_stop
                break

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch, including logging and evaluation on
        non-train partitions.

        Args:
            epoch (int): current training epoch.
        Returns:
            logs (dict): logs that contain the average loss and metric in
                this epoch.
        """
        self.is_train = True
        self.model.train()
        self.train_metrics.reset()
        self.writer.set_step((epoch - 1) * self.epoch_len)
        self.writer.add_scalar("epoch", epoch)
        # kept outside the loop: with skip_oom every batch of the epoch may be
        # skipped, and the logs still have to be defined
        last_train_metrics = {}
        done_steps = 0
        for batch_idx, batch in enumerate(
            tqdm(self.train_dataloader, desc="train", total=self.epoch_len)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    metrics=self.train_metrics,
                )
            except torch.cuda.OutOfMemoryError as e:
                if self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    torch.cuda.empty_cache()  # free some memory
                    continue
                else:
                    raise e

            done_steps += 1
            self.train_metrics.update("grad_norm", self._get_grad_norm())

            # log current results
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.epoch_len + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "learning rate", self.lr_scheduler.get_last_lr()[0]
                )
                self._log_scalars(self.train_metrics)
                self._log_batch(batch_idx, batch)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx + 1 >= self.epoch_len:
                break

        if done_steps == 0:
            self.logger.warning(
                f"Epoch {epoch}: every batch ran out of GPU memory and was skipped "
                "(skip_oom=True), the model was not updated. Reduce "
                "dataloader.batch_size or enable trainer.use_amp."
            )
        elif not last_train_metrics:
            # log_step is larger than the epoch: report what has been accumulated
            last_train_metrics = self.train_metrics.result()

        logs = last_train_metrics

        # Run val/test
        for part, dataloader in self.evaluation_dataloaders.items():
            val_logs = self._evaluation_epoch(epoch, part, dataloader)
            logs.update(**{f"{part}_{name}": value for name, value in val_logs.items()})

        return logs

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Evaluate model on the partition after training for an epoch.

        Args:
            epoch (int): current training epoch.
            part (str): partition to evaluate on
            dataloader (DataLoader): dataloader for the partition.
        Returns:
            logs (dict): logs that contain the information about evaluation.
        """
        self.is_train = False
        self.model.eval()
        self.evaluation_metrics.reset()
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
            )  # log only the last batch during inference

        return self.evaluation_metrics.result()

    def _setup_monitoring(self):
        """
        Read the model selection settings from the config: the monitored
        metric ('trainer.monitor'), the optional secondary metric that breaks
        its ties ('trainer.monitor_tiebreak', off by default) and the patience
        of the early stopping. Both metrics are given as "mode metric", e.g.
        "min dev_EER".
        """
        # secondary metric that breaks the ties of the monitored one,
        # see _check_improvement
        self.tiebreak_mode = None
        self.tiebreak_metric = None
        self.tiebreak_worst = None
        self.tiebreak_best = None
        # criterion the last saved model_best.pth won by, for the logs
        self.best_criterion = ""

        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
            return

        self.mnt_mode, self.mnt_metric = self.monitor.split()
        assert self.mnt_mode in ["min", "max"]

        self.mnt_best = inf if self.mnt_mode == "min" else -inf
        self.early_stop = self.cfg_trainer.get("early_stop", inf)
        if self.early_stop <= 0:
            self.early_stop = inf

        tiebreak = self.cfg_trainer.get("monitor_tiebreak", None)
        if tiebreak in (None, "off"):
            return

        self.tiebreak_mode, self.tiebreak_metric = tiebreak.split()
        assert self.tiebreak_mode in ["min", "max"]
        self.tiebreak_worst = inf if self.tiebreak_mode == "min" else -inf
        self.tiebreak_best = self.tiebreak_worst

    def _monitor_performance(self, logs, not_improved_count):
        """
        Check if there is an improvement in the metrics. Used for early
        stopping and saving the best checkpoint.

        Args:
            logs (dict): logs after training and evaluating the model for
                an epoch.
            not_improved_count (int): the current number of epochs without
                improvement.
        Returns:
            best (bool): if True, the monitored metric has improved.
            stop_process (bool): if True, stop the process (early stopping).
                The metric did not improve for too much epochs.
            not_improved_count (int): updated number of epochs without
                improvement.
        """
        best = False
        stop_process = False
        if self.mnt_mode != "off":
            try:
                improved = self._check_improvement(logs)
            except KeyError:
                self.logger.warning(
                    f"Warning: Metric '{self.mnt_metric}' is not found. "
                    "Model performance monitoring is disabled."
                )
                self.mnt_mode = "off"
                improved = False

            if improved:
                not_improved_count = 0
                best = True
            else:
                not_improved_count += 1

            if not_improved_count >= self.early_stop:
                self.logger.info(
                    "Validation performance didn't improve for {} epochs. "
                    "Training stops.".format(self.early_stop)
                )
                stop_process = True
        return best, stop_process, not_improved_count

    def _check_improvement(self, logs):
        """
        Decide whether the epoch is the new best one and remember its values.

        The comparison is strict. A non-strict one makes every epoch "best"
        as soon as the metric saturates (dev_EER hits 0.0 on ASVspoof2019 LA
        within a few epochs), so model_best.pth degenerates into the last
        epoch and early stopping never triggers. Strictness alone freezes the
        best checkpoint on the first epoch that reached the optimum, which is
        just as arbitrary, hence the optional secondary metric
        ('trainer.monitor_tiebreak', e.g. "min dev_loss"): while the primary
        metric stands still, the epochs are ranked by the secondary one.

        Args:
            logs (dict): logs after training and evaluating the model for
                an epoch.
        Returns:
            improved (bool): True if the epoch is the new best one.
        Raises:
            KeyError: the monitored metric is not in the logs.
        """
        value = logs[self.mnt_metric]

        if self._is_better(value, self.mnt_best, self.mnt_mode):
            previous = self.mnt_best
            self.mnt_best = value
            self.tiebreak_best = self._tiebreak_value(logs, self.tiebreak_worst)
            self.best_criterion = (
                f"{self.mnt_metric}={value:.6g} improved from {previous:.6g}"
            )
            return True

        if self.tiebreak_metric is None or value != self.mnt_best:
            return False

        tiebreak_value = self._tiebreak_value(logs, None)
        if tiebreak_value is None or not self._is_better(
            tiebreak_value, self.tiebreak_best, self.tiebreak_mode
        ):
            return False

        previous = self.tiebreak_best
        self.tiebreak_best = tiebreak_value
        self.best_criterion = (
            f"{self.mnt_metric}={value:.6g} unchanged, tiebreak "
            f"{self.tiebreak_metric}={tiebreak_value:.6g} improved from "
            f"{previous:.6g}"
        )
        return True

    def _tiebreak_value(self, logs, default):
        """
        Read the secondary metric of the epoch.

        Args:
            logs (dict): logs of the epoch.
            default (float | None): value to return when the tiebreak is off
                or its metric is not logged.
        Returns:
            value (float | None): value of the tiebreak metric.
        """
        if self.tiebreak_metric is None:
            return default
        return logs.get(self.tiebreak_metric, default)

    @staticmethod
    def _is_better(value, best, mode):
        """
        Compare a metric value with the best one seen so far.

        Args:
            value (float): value of the current epoch.
            best (float): best value so far.
            mode (str): "min" or "max".
        Returns:
            is_better (bool): True if value is strictly better than best.
        """
        return value < best if mode == "min" else value > best

    def move_batch_to_device(self, batch):
        """
        Move all necessary tensors to the device.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader with some of the tensors on the device.
        """
        for tensor_for_device in self.cfg_trainer.device_tensors:
            batch[tensor_for_device] = batch[tensor_for_device].to(self.device)
        return batch

    def transform_batch(self, batch):
        """
        Transforms elements in batch. Like instance transform inside the
        BaseDataset class, but for the whole batch. Improves pipeline speed,
        especially if used with a GPU.

        Each tensor in a batch undergoes its own transform defined by the key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform).
        """
        # do batch transforms on device
        transform_type = "train" if self.is_train else "inference"
        transforms = self.batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                batch[transform_name] = transforms[transform_name](
                    batch[transform_name]
                )
        return batch

    def _setup_amp(self):
        """
        Read the mixed precision settings from the config and check that the
        device supports them.

        The LCNN input is a 863x600 spectrogram, so activations dominate the
        memory footprint; bf16 halves it and speeds the forward pass up. bf16
        has the same exponent range as fp32, hence no GradScaler is required.
        Autocast is only available on CUDA, so on CPU the request is refused
        instead of silently changing the numerics.
        """
        self.use_amp = bool(self.cfg_trainer.get("use_amp", False))
        self.amp_dtype = getattr(
            torch, str(self.cfg_trainer.get("amp_dtype", "bfloat16"))
        )
        self.amp_device_type = torch.device(self.device).type

        if self.use_amp and self.amp_device_type != "cuda":
            self._report(
                f"AMP is requested but the device is '{self.device}'. "
                "Autocast is disabled.",
                level="warning",
            )
            self.use_amp = False
        elif self.use_amp:
            self._report(f"Running with autocast, dtype={self.amp_dtype}.")

    def _autocast(self):
        """
        Context manager for the forward pass.

        Returns:
            context (AbstractContextManager): autocast context if AMP is
                enabled, a no-op context otherwise.
        """
        if not self.use_amp:
            return nullcontext()
        return torch.autocast(device_type=self.amp_device_type, dtype=self.amp_dtype)

    def _report(self, message, level="info"):
        """
        Log a message, falling back to stdout: the inferencer is constructed
        without a logger.

        Args:
            message (str): text to report.
            level (str): name of the logger method, "info" or "warning".
        """
        if hasattr(self, "logger"):
            getattr(self.logger, level)(message)
        else:
            print(message)

    def _clip_grad_norm(self):
        """
        Clips the gradient norm by the value defined in
        config.trainer.max_grad_norm
        """
        if self.config["trainer"].get("max_grad_norm", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["max_grad_norm"]
            )

    @torch.no_grad()
    def _get_grad_norm(self, norm_type=2):
        """
        Calculates the gradient norm for logging.

        Args:
            norm_type (float | str | None): the order of the norm.
        Returns:
            total_norm (float): the calculated norm.
        """
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
            norm_type,
        )
        return total_norm.item()

    def _progress(self, batch_idx):
        """
        Calculates the percentage of processed batch within the epoch.

        Args:
            batch_idx (int): the current batch index.
        Returns:
            progress (str): contains current step and percentage
                within the epoch.
        """
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.epoch_len
        return base.format(current, total, 100.0 * current / total)

    @abstractmethod
    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Abstract method. Should be defined in the nested Trainer Class.

        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        return NotImplementedError()

    def _log_scalars(self, metric_tracker: MetricTracker):
        """
        Wrapper around the writer 'add_scalar' to log all metrics.

        Args:
            metric_tracker (MetricTracker): calculated metrics.
        """
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))

    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        """
        Save the checkpoints.

        Args:
            epoch (int): current epoch number.
            save_best (bool): if True, rename the saved checkpoint to 'model_best.pth'.
            only_best (bool): if True and the checkpoint is the best, save it only as
                'model_best.pth'(do not duplicate the checkpoint as
                checkpoint-epochEpochNumber.pth)
        """
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "monitor_best": self.mnt_best,
            "monitor_tiebreak_best": self.tiebreak_best,
            "config": self.config,
        }
        filename = str(self.checkpoint_dir / f"checkpoint-epoch{epoch}.pth")
        if not (only_best and save_best):
            torch.save(state, filename)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(filename, str(self.checkpoint_dir.parent))
            self.logger.info(f"Saving checkpoint: {filename} ...")
        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(best_path, str(self.checkpoint_dir.parent))
            # the criterion is logged with the checkpoint: with a saturated
            # metric it is the only way to tell afterwards why this very epoch
            # became the best one
            criterion = self.best_criterion or "monitoring is off"
            self.logger.info(f"Saving current best: model_best.pth ({criterion}) ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from a saved checkpoint (in case of server crash, etc.).
        The function loads state dicts for everything, including model,
        optimizers, etc.

        Notice that the checkpoint should be located in the current experiment
        saved directory (where all checkpoints are saved in '_save_checkpoint').

        Args:
            resume_path (str): Path to the checkpoint to be resumed.
        """
        resume_path = str(resume_path)
        self.logger.info(f"Loading checkpoint: {resume_path} ...")
        # weights_only=False: the checkpoint stores the hydra config object,
        # which the safe unpickler of torch>=2.6 refuses to load
        checkpoint = torch.load(
            resume_path, map_location=self.device, weights_only=False
        )
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]
        if self.tiebreak_metric is not None:
            # checkpoints written before the tiebreak existed (or with it off)
            # carry no value to restore
            restored = checkpoint.get("monitor_tiebreak_best")
            self.tiebreak_best = self.tiebreak_worst if restored is None else restored

        # load architecture params from checkpoint.
        if checkpoint["config"]["model"] != self.config["model"]:
            self.logger.warning(
                "Warning: Architecture configuration given in the config file is different from that "
                "of the checkpoint. This may yield an exception when state_dict is loaded."
            )
        self.model.load_state_dict(checkpoint["state_dict"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if (
            checkpoint["config"]["optimizer"] != self.config["optimizer"]
            or checkpoint["config"]["lr_scheduler"] != self.config["lr_scheduler"]
        ):
            self.logger.warning(
                "Warning: Optimizer or lr_scheduler given in the config file is different "
                "from that of the checkpoint. Optimizer and scheduler parameters "
                "are not resumed."
            )
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        self.logger.info(
            f"Checkpoint loaded. Resume training from epoch {self.start_epoch}"
        )

    def _from_pretrained(self, pretrained_path):
        """
        Init model with weights from pretrained pth file.

        Notice that 'pretrained_path' can be any path on the disk. It is not
        necessary to locate it in the experiment saved dir. The function
        initializes only the model.

        Args:
            pretrained_path (str): path to the model state dict.
        """
        pretrained_path = str(pretrained_path)
        self._report(f"Loading model weights from: {pretrained_path} ...")
        # weights_only=False: '_save_checkpoint' stores the hydra config object
        checkpoint = torch.load(
            pretrained_path, map_location=self.device, weights_only=False
        )

        if checkpoint.get("state_dict") is not None:
            self._check_input_pipeline(checkpoint.get("config"), pretrained_path)
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

    def _check_input_pipeline(self, saved_config, pretrained_path):
        """
        Compare the config the checkpoint was trained with against the current
        one and report every difference that changes the input of the model.

        'load_state_dict' checks the shapes of the weights only, so a changed
        front-end or a changed waveform length is accepted without a word and
        silently invalidates the scores.

        Args:
            saved_config (DictConfig | None): config stored in the checkpoint.
            pretrained_path (str): path of the loaded checkpoint, for the text
                of the warning.
        """
        current_config = getattr(self, "config", None)
        if current_config is None:
            return
        if saved_config is None:
            self._report(
                f"The checkpoint '{pretrained_path}' stores no config: the "
                "front-end it was trained with cannot be verified.",
                level="warning",
            )
            return

        mismatches = config_mismatches(saved_config, current_config)
        if mismatches:
            self._report(
                format_mismatch_warning(mismatches, pretrained_path), level="warning"
            )
        else:
            self._report("Checkpoint config matches the current input pipeline.")
