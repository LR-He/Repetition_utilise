import math
import matplotlib
import os
import torch

from enum import Enum
from matplotlib import pyplot as plt
from torch import Tensor, nn
from typing import Any, Dict, List, Literal, Optional, Tuple

from lib import config_utils, data_utils, utils, visutils
from lib.models import MODELS
from lib.visutils import COLORMAPS


class Method(Enum):
    UTILISE = 'utilise'
    TRIVIAL = 'trivial'

    
class Mode(Enum):
    LAST = 'last'
    NEXT = 'next'
    CLOSEST = 'closest'
    LINEAR_INTERPOLATION = 'linear_interpolation'
    NONE = None

"""
上面的代码定义了两个枚举类（Enum）：Method 和 Mode，它们用于创建有限集合的枚举类型，每个枚举成员都有与之关联的唯一值。
在这里，定义了两个枚举类，Method 和 Mode，并为每个枚举成员分配了相应的值。

Method 枚举类：
    Method 枚举类包含两个枚举成员：UTILISE 和 TRIVIAL。
    UTILISE 的关联值是 'utilise'，它表示某种方法被使用。
    TRIVIAL 的关联值是 'trivial'，它表示某种方法是平凡的或不复杂的。
Mode 枚举类：
    Mode 枚举类包含五个枚举成员：LAST、NEXT、CLOSEST、LINEAR_INTERPOLATION 和 NONE。
    LAST 的关联值是 'last'，它可能表示一种操作模式或方式。
    NEXT 的关联值是 'next'，它也可能表示一种操作模式或方式。
    CLOSEST 的关联值是 'closest'，它表示某种操作模式，可能与查找最接近的内容有关。
    LINEAR_INTERPOLATION 的关联值是 'linear_interpolation'，它表示一种插值方式。
    NONE 的关联值是 None，它表示不存在特定的操作模式或方式。
枚举类型是一种非常有用的方式来定义一组相关的常量，使代码更具可读性和维护性，因为每个枚举成员都有清晰的名称和相关值。
在实际编程中，可以使用这些枚举成员来表示不同的选项、状态或模式，以提高代码的可理解性和可维护性。


            Args:
                config_file_train:
                method:
                mode:
                checkpoint:
    
            method: Literal['utilise', 'trivial'] = 'utilise' 这行代码是Python中函数参数的定义，通常用于类的构造函数。这段代码中包含了以下信息：
                1. method：这是函数的参数名，用于在函数内部引用该参数的值。
                2. Literal['utilise', 'trivial']：这部分定义了参数的类型注释。
                    它使用了Literal类型，它是Python 3.8+引入的类型提示，
                    表示参数的值只能是在方括号中列出的特定值之一，即 'utilise' 或 'trivial'。
                3. =：这个等号表示在函数调用时，如果没有提供参数值，将使用等号右边的默认值。
                4. 'utilise'：这是默认值，如果在函数调用时未提供method参数的值，那么method将默认为 'utilise'。
            所以，method是一个函数参数，用于接收一个字符串值，但只能接受 'utilise' 或 'trivial' 之一，如果没有提供值，默认为 'utilise'。
    
            Method(method)：当执行 Method(method)时，它会尝试将传入的 method 值与枚举类 Method 中的成员进行比较，并返回匹配的枚举成员。
            例如，如果 method 的值是 'utilise'，那么 Method(method) 将返回 Method.UTILISE，因为 'utilise' 与 Method 中的 UTILISE 成员匹配。
            同样，如果 method 的值是 'trivial'，Method(method) 将返回 Method.TRIVIAL。
            这允许将字符串值映射到具体的枚举成员，以便在代码中使用更具语义的枚举值，而不是直接使用字符串。这有助于提高代码的可读性和可维护性。
            在代码中，可以使用返回的枚举成员进行比较、操作和其他操作，而不必担心字符串的拼写错误或大小写问题。
"""


class Imputation:
    def __init__(
            self,
            config_file_train: str | None,
            method: Literal['utilise', 'trivial'] = 'utilise',
            mode: Literal['last', 'next', 'closest', 'linear_interpolation'] | None = None,
            checkpoint: str | None = None
    ):

        # 属性的初始化：
        self.method = Method(method) # 根据传入的method参数创建一个Method枚举对象，用于表示操作方法。
        self.mode = Mode(mode) # 根据传入的mode参数创建一个Mode枚举对象，用于表示操作模式。
        self.checkpoint = checkpoint # 将传入的checkpoint参数存储在对象属性中。
        self.config_file_train = config_file_train # 将传入的config_file_train参数存储在对象属性中。

        # 错误检查：
        if self.method == Method.TRIVIAL and self.mode == Mode.NONE:
            raise ValueError(f'No mode specified. Choose among {[mode.value for mode in Mode]}.')
        
        if self.method == Method.UTILISE:
            if self.checkpoint is None:
                raise ValueError('No checkpoint specified.\n')
                
            if self.config_file_train is None:
                raise ValueError('No training configuration file specified.\n')
            
            if not os.path.isfile(self.config_file_train):
                raise FileNotFoundError(f'Cannot find the configuration file used during training: {self.config_file_train}\n')

            if not os.path.isfile(self.checkpoint):
                raise FileNotFoundError(f'Cannot find the model weights: {self.checkpoint}\n')
                
            # Read the configuration file used during training
            self.config = config_utils.read_config(self.config_file_train) # 返回一个字典，以自定义DictConfig类表示。

            # Extract the temporal window size and the number of channels used during training
            self.temporal_window = self.config.data.max_seq_length
            self.num_channels = data_utils.get_dataset(self.config, phase=self.config.misc.run_mode).num_channels
        

        self.device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
        _ = torch.set_grad_enabled(False) # 禁止计算梯度

        # Get the model
        if self.method == Method.UTILISE:
            # utils.get_model(self.config, self.num_channels)返回模型和模型参数，返回的模型已经使用模型参数初始化了，并且权重也初始化了。
            self.model, _ = utils.get_model(self.config, self.num_channels)
            self._resume() # 该方法（定义在下方）用于加载指定的检查点文件，并将其模型权重加载到当前模型中，没有接收的参数，也没有返回的东西。
            self.model.to(self.device).eval()
        else:
            self.model = MODELS['ImageSeriesInterpolator'](mode=self.mode.value)
            # MODELS是一个字典，MODELS['ImageSeriesInterpolator']返回一个ImageSeriesInterpolator，是一个模型。
            # (mode=self.mode.value)表示传入“mode”参数，实例化ImageSeriesInterpolator模型，该参数接受一个字符串，表示需要实例化哪种模型结构。

    def  impute_sample(
            self,
            batch: Dict[str, Any],
            t_start: Optional[int] = None,
            t_end: Optional[int] = None,
            return_att: Optional[bool] = False
    ) -> Tuple[Dict[str, Any], Tensor, Tensor] | Tuple[Dict[str, Any], Tensor]:
    # 该方法对给定的卫星图像时间序列进行插补（impute）操作。
    # 它接受一系列参数，包括 batch（一个字典，包含要插补的卫星图像时间序列数据）、
    # t_start 和 t_end（指定时间序列的起始和结束索引，用于选择子序列），
    # 以及 return_att（一个布尔值，用于控制是否返回注意力权重）。

        if t_start is not None and t_end is not None:
            # Choose a subsequence
            batch['x'] = batch['x'][:, t_start:t_end, ...] # 根据 t_start 和 t_end从 batch 字典中选择时间序列的子序列。这将限制时间序列的时间范围。

            # 通过循环，对一些其他关键数据也执行类似的操作，以确保它们与选择的时间子序列保持一致。
            for key in ['y', 'masks', 'cloud_mask', 'masks_valid_obs']:
                if key in batch:
                    batch[key] = batch[key][:, t_start:t_end, ...]
                    
            for key in ['days', 'position_days']:
                if key in batch:
                    batch[key] = batch[key][:, t_start:t_end]

        # Impute the given satellite image time series
        if isinstance(self.model, MODELS['utilise']):
            batch = data_utils.to_device(batch, self.device)
            if return_att: # 如果 return_att 为 True，则获取注意力权重 att
                # 调用 impute_sequence 函数对时间序列数据进行插补，该函数使用了 self.model。
                y_pred, att = impute_sequence(self.model, batch, self.temporal_window, return_att=True)
                if att is not None:
                    att = att.cpu() # 将数据移回 CPU
            else:
                y_pred = impute_sequence(self.model, batch, self.temporal_window, return_att=False)
            batch = data_utils.to_device(batch, 'cpu') #  # 将数据移回 CPU
            y_pred = y_pred.cpu() #  # 将数据移回 CPU
        else:
            y_pred = self.model(batch['x'], cloud_mask=batch['masks'], days=batch['days'])

        if return_att: # 如果 return_att 为 True，则最后还会返回注意力权重 att。
            return batch, y_pred, att
        return batch, y_pred

    def _resume(self) -> None:
        # 这个方法用于加载指定的检查点文件，将其模型权重加载到当前模型中，
        # 然后打印一些有关加载过程的信息，包括检查点文件的路径和选择的训练轮数。
        # 这对于从之前的训练状态继续训练模型非常有用。

        checkpoint = torch.load(self.checkpoint) # 从指定的检查点文件中加载模型的状态。检查点文件通常包含模型的权重、优化器状态、训练进度等信息。
        self.model.load_state_dict(checkpoint['model_state_dict']) # 将从检查点文件中加载的模型状态中的模型权重（通常存储在 'model_state_dict' 键下）加载到当前的模型中，以恢复模型的权重。
        print(f'Checkpoint \'{self.checkpoint}\' loaded.')
        print(f"Chosen epoch: {checkpoint['epoch']}\n") # 删除了 checkpoint 变量，以释放检查点文件占用的内存。一旦检查点文件的信息已经加载到模型中，通常可以安全地删除检查点变量以释放内存。
        del checkpoint


def impute_sequence(
        model, batch: Dict[str, Any], temporal_window: int, return_att: bool = False
) -> Tensor | Tuple[Tensor, Tensor]:
    """
    Sliding-window imputation of satellite image time series.

    Assumption: `batch` consists of a single sample.
    """

    x = batch['x']
    positions = batch['position_days']
    y_pred: Tensor
    att: Tensor

    if temporal_window is None or x.shape[1] <= temporal_window:
        # Process the entire sequence in one go
        if return_att:
            y_pred, att = model(x, batch_positions=positions, return_att=True)
        else:
            y_pred = model(x, batch_positions=positions)
    else:
        if return_att:
            att = None
            
        t_start = 0
        t_end = temporal_window
        t_max = x.shape[1]
        cloud_coverage = torch.mean(batch['masks'], dim=(0, 2, 3, 4))
        reached_end = False

        while not reached_end:
            # 处理滑动窗口中的数据。
            y_pred_chunk = model(x[:, t_start:t_end], batch_positions=positions[:, t_start:t_end])

            if t_start == 0:
                # Initialize the full-length output sequence
                B, T, _, H, W = x.shape
                C = y_pred_chunk.shape[2]
                y_pred = torch.zeros((B, T, C, H, W), device=x.device)

                y_pred[:, t_start:t_end] = y_pred_chunk

                # Move the temporal window
                t_start_old = t_start
                t_end_old = t_end
                t_start, t_end = move_temporal_window_next(t_start, t_max, temporal_window, cloud_coverage)
            else:
                # Find the indices of those frames that have been processed by both the previous and the current
                # temporal window
                # 找到那些已经在前一时窗和当前时窗中处理过的帧的索引，即重叠处理过的帧的索引。
                t_candidates = torch.Tensor(
                    list(set(torch.arange(t_start_old, t_end_old).tolist()) & set(
                        torch.arange(t_start, t_end).tolist()))
                ).long().to(x.device)

                # Find the frame for which the difference between the previous and the current prediction is
                # the lowest:
                # use this frame to switch from the previous imputation results to the current imputation results
                # 找到前一次和当前预测之间差异最小的帧：使用这一帧将上一次的填充结果替换为当前的填充结果。
                # t_candidates: tensor([5, 6, 7, 8, 9], device='cuda:4')
                # t_start: 5
                # t_candidates - t_start: tensor([0, 1, 2, 3, 4], device='cuda:4')，t_candidates - t_start表示t_candidates中的每个数都减去5，这里的减法用了tensor的广播规则去处理。
                # y_pred_chunk.shape: torch.Size([1, 10, 4, 128, 128])
                # y_pred_chunk[:, t_candidates - t_start].shape: torch.Size([1, 5, 4, 128, 128])，表示当前预测结果的前5帧。
                # y_pred[:, t_candidates].shape: torch.Size([1, 5, 4, 128, 128])，表示上一个预测结果的后5帧。
                # torch.abs(y_pred[:, t_candidates] - y_pred_chunk[:, t_candidates - t_start])求两者差异的绝对值。
                # 当前预测结果的前5帧与上一个预测结果的后5帧是相同的输入，也就是滑动窗口中重叠处理过的帧。
                error = torch.mean(
                    torch.abs(y_pred[:, t_candidates] - y_pred_chunk[:, t_candidates - t_start]),
                    dim=(0, 2, 3, 4)
                )
                # error.argmin().item()表示找到error中最小值的索引。
                # error.argmin(): tensor(3, device='cuda:4')
                # error.argmin().item(): 3
                # t_switch表示前一次和当前预测之间差异最小的帧在整个时间序列中的索引。
                t_switch = error.argmin().item() + t_start
                # 将当前预测的结果中从重叠差异最小处开始，替换部分与上一次预测的结果重叠的部分，同时，本次预测中不与上次重叠的部分也保存在最终预测结果中。
                y_pred[:, t_switch:t_end] = y_pred_chunk[:, (t_switch - t_start)::]

                if t_end == t_max:
                    reached_end = True
                else:
                    # Move the temporal window
                    t_start_old = t_start
                    t_end_old = t_end
                    t_start, t_end = move_temporal_window_next(
                        t_start_old, t_max, temporal_window, cloud_coverage
                    )

    if return_att:
        return y_pred, att
    return y_pred


def move_temporal_window_end(t_max: int, temporal_window: int) -> Tuple[int, int]:
    """
    Moves the temporal window for evaluation such that the last frame of the temporal window coincides with the
    last frame of the image sequence.

    Args:
        t_max:              int, sequence length of the image sequence
        temporal_window:    int, length of the subsequence passed to U-TILISE for processing

    Returns:
        t_start:            int, frame index, start of the subsequence
        t_end:              int, frame index, end of the subsequence
    """

    t_start = t_max - temporal_window
    t_end = t_max

    return t_start, t_end


def move_temporal_window_next(
        t_start: int, t_max: int, temporal_window: int, cloud_coverage: Tensor
) -> Tuple[int, int]:
    """
    Moves the temporal window for evaluation by half of the temporal window size (= stride).
    If the first frame within the new temporal window is cloudy (cloud coverage above 10%), the temporal window is
    shifted by at most half the stride (backward or forward) such that the first frame is as least cloudy as
    possible.

    Args:
        t_start:            int, frame index, start of the subsequence for processing
        t_max:              int, frame index, t_max - 1 is the last frame of the subsequence for processing
        temporal_window:    int, length of the subsequence passed to U-TILISE for processing
        cloud_coverage:     torch.Tensor, (T,), cloud coverage [-] per frame

    Returns:
        t_start:            int, frame index, start of the subsequence
        t_end:              int, frame index, end of the subsequence
    """

    stride = temporal_window // 2
    t_start += stride

    if t_start + temporal_window > t_max:
        # Reduce the stride such that the end of the temporal window coincides with the end of the entire sequence
        t_start, t_end = move_temporal_window_end(t_max, temporal_window)
    else:
        # Check if the start of the next temporal window is mostly cloud-free
        if cloud_coverage[t_start] <= 0.1:
            # Keep the default stride and ensure that the temporal window does not exceed the sequence length
            t_end = t_start + temporal_window
            if t_end > t_max:
                t_start, t_end = move_temporal_window_end(t_max, temporal_window)
        else:
            # 当滑动窗口的第一张图像的云量大于0.1时，寻找该张图象附近最小云量的图像作为第一张开始图像。
            # 下面对“附近”进行定义，是该张图象前后3张。
            # Find the least cloudy frame within [t_start + stride - dt, t_start + stride + dt]
            dt = math.ceil(stride / 2)
            left = max(0, t_start - dt)
            right = min(t_start + dt + 1, t_max)

            # Frame(s) with the lowest cloud coverage within [t_start + stride - dt, t_start + stride + dt]
            t_candidates = (cloud_coverage[left:right] == cloud_coverage[left:right].min()).nonzero(as_tuple=True)[
                               0] + left
            # cloud_coverage[left:right]表示找到cloud_coverage中该图像附近的6张图象的云覆盖率，
            # cloud_coverage[left:right].min()找到cloud_coverage[left:right]中云覆盖率最小的值，
            # cloud_coverage[left:right] == cloud_coverage[left:right].min()将值等于最小云覆盖率的位置为True，其他位置为False，返回一个布尔矩阵，
            # .nonzero(as_tuple=True)表示找到为True的位置（可能会有多个），以元组的形式返回，.nonzero(as_tuple=True)[0]表示返回该位置的索引，
            # 该位置的索引加上left后，就是云量最小图像在时间序列中的位置，即t_candidates表示云量最小的图像的索引，以元组的形式表示。

            # Take the frame closest to the standard stride
            # 找到离原t_start最近的位置的索引，并将该位置作为新t_start。
            t_start = t_candidates[torch.abs(t_candidates - t_start).argmin()].item()

            # Ensure that the temporal window does not exceed the sequence length
            t_end = t_start + temporal_window
            if t_end > t_max:
                t_start, t_end = move_temporal_window_end(t_max, temporal_window)

    return t_start, t_end


def upsample_att_maps(att: Tensor, target_shape: Tuple[int, int]) -> Tensor:
    """Upsamples the attention masks `att` to the spatial resolution `target_shape`."""

    n_heads, b, t_out, t_in, h, w = att.shape
    attn = att.view(n_heads * b * t_out, t_in, h, w)

    attn = nn.Upsample(
        size=target_shape, mode="bilinear", align_corners=False
    )(attn)

    return attn.view(n_heads, b, t_out, t_in, *target_shape)


def visualize_att_for_one_head_across_time(
        seq: Tensor,
        att: Tensor,
        head: int,
        batch: int = 0,
        upsample_att: bool = True,
        indices_rgb: List[int] | List[float] | Tensor | None = None,
        brightness_factor: float = 1,
        fontsize: int = 10,
        scale_individually: bool = False
) -> matplotlib.figure.Figure:
    """
    Visualizes the attention masks learned by the `head`.th attention head across time.

    Args:
        seq:                    torch.Tensor, B x T x C x H x W, satellite image time series.
        att:                    torch.Tensor, n_head x B x T x T x h x w, attention masks.
        head:                   int, index of the attention head to be visualized.
        batch:                  int, batch index to visualize.
        upsample_att:           bool, True to upsample the attention masks to the spatial resolution of the satellite
                                image time series; False to keep the native spatial resolution of the attention masks.
        indices_rgb:            list of int or list of float or torch.Tensor, indices of the RGB channels.
        brightness_factor:      float, brightness factor applied to all images in the sequence.
        figsize:                (float, float), figure size.
        fontsize:               int, font size.
        scale_individually:     bool, True to scale the attention masks for each time step individually; False to
                                maintain a common scale across all attention masks and time.

    Returns:
        matplotlib.pyplot.
    """

    indices_rgb = [0, 1, 2] if indices_rgb is None else indices_rgb

    if upsample_att:
        target_shape = seq.shape[-2:]
        att = upsample_att_maps(att, target_shape)

    seq_length = seq.shape[1]
    figsize = (7, 1 + seq_length)
    fig, axes = plt.subplots(nrows=seq_length + 1, ncols=1, figsize=figsize)

    # Plot satellite image time series
    grid = visutils.gallery(seq[batch, :, indices_rgb, :, :], brightness_factor=brightness_factor)
    axes[0].imshow(grid, COLORMAPS['rgb'])
    axes[0].set_title('Input sequence', fontsize=fontsize)

    if scale_individually:
        vmin = None
        vmax = None
    else:
        vmin = 0
        vmax = 1

    # Plot attention mask for attention head `head` across all time steps
    for t in range(seq_length):
        grid = visutils.gallery(att[head, batch, t, :, :, :].unsqueeze(1), brightness_factor=1)
        axes[t + 1].imshow(grid, COLORMAPS['att'], vmin=vmin, vmax=vmax)
        axes[t + 1].set_title(f'Attention mask, head {head}, target frame {t}', fontsize=fontsize)

    for ax in axes.ravel():
        ax.set_axis_off()
    plt.tight_layout()

    return fig


def visualize_att_for_target_t_across_heads(
        seq: Tensor,
        att: Tensor,
        t_target: int,
        batch: int = 0,
        upsample_att: bool = True,
        indices_rgb: List[int] | List[float] | Tensor | None = None,
        brightness_factor: float = 1,
        figsize: Tuple[float, float] = (10, 7),
        dpi: int = 200,
        fontsize: int = 10,
        scale_individually: bool = False,
        highlight_t_target: bool = True
) -> matplotlib.figure.Figure:
    """
    Visualizes the attention masks of all attention heads w.r.t. to the time step `t_target`.

    Args:
        seq:                    torch.Tensor, B x T x C x H x W, satellite image time series.
        att:                    torch.Tensor, n_head x B x T x T x h x w, attention masks.
        t_target:               int, time step (temporal coordinate) to visualize.
        batch:                  int, batch index to visualize.
        upsample_att:           bool, True to upsample the attention masks to the spatial resolution of the satellite
                                image time series; False to keep the native spatial resolution of the attention masks.
        indices_rgb:            list of int or list of float or torch.Tensor, indices of the RGB channels.
        brightness_factor:      float, brightness factor applied to all images in the sequence.
        figsize:                (float, float), figure size.
        dpi:                    int, dpi of the figure.
        fontsize:               int, font size.
        scale_individually:     bool, True to scale the attention masks for each time step individually; False to
                                maintain a common scale across all attention masks and time.
        highlight_t_target:     bool, True to highlight the target time step by drawing a red frame around the
                                respective image in the time series.

    Returns:
        matplotlib.pyplot.
    """

    indices_rgb = [0, 1, 2] if indices_rgb is None else indices_rgb

    if upsample_att:
        target_shape = seq.shape[-2:]
        att = upsample_att_maps(att, target_shape)

    n_heads = att.shape[0]
    fig, axes = plt.subplots(nrows=n_heads + 1, ncols=1, figsize=figsize, dpi=dpi)

    # Plot input sequence
    grid = visutils.gallery(seq[batch, :, indices_rgb, :, :], brightness_factor=brightness_factor)

    if highlight_t_target:
        # Create a red frame to highlight the target frame
        border_thickness = 2
        H, W = seq.shape[-2:]
        H += 2 * border_thickness
        W += 2 * border_thickness
        frame_color = torch.Tensor([1, 0, 0]).type(grid.dtype)

        if t_target < 0:
            t_target = seq.shape[1] - abs(t_target)

        grid[0:(2 * border_thickness + 1), t_target * W:(t_target + 1) * W, :] = frame_color
        grid[-2 * border_thickness::, t_target * W:(t_target + 1) * W, :3] = frame_color
        grid[:, t_target * W - border_thickness:t_target * W + border_thickness, :] = frame_color
        grid[:, ((t_target + 1) * W - border_thickness):(t_target + 1) * W + border_thickness, :] = frame_color

        if t_target == seq.shape[1] - 1:
            grid[:, ((t_target + 1) * W - 2 * border_thickness):(t_target + 1) * W + border_thickness, :] = frame_color
        elif t_target == 0:
            grid[:, 0:2 * border_thickness, :] = frame_color

    axes[0].imshow(grid, COLORMAPS['rgb'])
    axes[0].set_title('Input sequence', fontsize=fontsize)

    if scale_individually:
        vmin = None
        vmax = None
    else:
        vmin = 0
        vmax = 1

    # Plot attention masks per head for frame `t_target`
    for head in range(n_heads):
        grid = visutils.gallery(att[head, batch, t_target, :, :, :].unsqueeze(1), brightness_factor=1)
        axes[head + 1].imshow(grid, COLORMAPS['att'], vmin=vmin, vmax=vmax)
        axes[head + 1].set_title(f'Attention mask, head {head}, target frame {t_target}', fontsize=fontsize)

    for ax in axes.ravel():
        ax.set_axis_off()
    plt.tight_layout()

    return fig
