"""PixelForge 算法引擎（可独立开源复用）。

模块划分
--------
* device   计算设备层：CPU(numpy) / GPU(CuPy) 自动探测，数组运算类型保持
* core     基础库：OKLab 色彩空间、面积/双线性重采样、高斯/盒式/导向滤波、
           显著图（谱残差 / 中心-周边 / 颜色稀有度 / 纹理能量）、扫描线连通域、k-means
* segment  主体选择：双色彩模型 + 显著性先验 + 多尺度 MRF(ICM) +
           智能多方案搜索与客观打分 + 阴影剔除 + 三值图/导向滤波精修
* pixelize 像素化：主体等比入格（含铺满模式）+ 逐格候选色 +
           SSIM 结构坐标下降 + 感知调色板/Potts 去噪 + 抖动 + 描边

最简用法
--------
>>> from algorithms import core, to_pixel_art, PixelArtOptions
>>> rgb, alpha = core.load_image("input.png")
>>> small_rgb, small_a, info = to_pixel_art(rgb, alpha, PixelArtOptions(size=32))
>>> core.save_rgba("out.png", small_rgb, small_a)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import core, device, pixelize, segment

__version__ = "1.0.0"

__all__ = ["core", "device", "segment", "pixelize", "PixelArtOptions", "to_pixel_art",
           "segment_subject", "pixelize_image", "__version__"]


@dataclass
class PixelArtOptions:
    """像素化参数（默认值即推荐值）。"""

    size: int = 32                     # 输出边长（正方形），也可用 out_w/out_h
    out_w: int = 0
    out_h: int = 0
    remove_bg: bool = True             # 抠掉背景（输出透明底）
    smart: bool = True                 # 智能主体检测（多方案试算 + 客观打分）
    sensitivity: float = 0.5           # 背景移除强度 0..1
    center_bias: float = 0.25          # 主体居中先验 0..1
    saliency: float = 0.5              # 显著性先验强度 0..1
    subject: str = "auto"              # auto / largest / center / all
    keep_shadow: bool = False          # 保留投影阴影
    palette: int = 24                  # 调色板颜色数，0 = 不限制
    detail: float = 0.75               # 细节保留强度 0..1
    structure_passes: int = 3          # SSIM 结构优化迭代次数 0..5
    contrast: float = 0.45             # 对比度增强 0..1
    saturation: float = 0.5            # 饱和度增强 0..1
    margin: int = 1                    # 居中留白时的边距（像素）
    fill: str = "none"                 # none 居中留白 / cover 等比铺满 / stretch 拉伸铺满
    trim: bool = False                 # 裁掉透明边让主体占满
    outline: int = 0                   # 0 无 / 1 内部描边 / 2 外部描边
    dither: str = "none"               # none / ordered / floyd
    dither_amount: float = 0.6
    extra: dict = field(default_factory=dict)


def _pixel_options(opt: PixelArtOptions) -> pixelize.PixelOptions:
    ow = int(opt.out_w or opt.size)
    oh = int(opt.out_h or opt.size)
    kw = dict(opt.extra or {})
    return pixelize.PixelOptions(
        out_w=ow, out_h=oh, palette=int(opt.palette), detail=float(opt.detail),
        structure_passes=int(opt.structure_passes), margin=int(opt.margin),
        contrast=float(opt.contrast), saturation=float(opt.saturation),
        outline=int(opt.outline), dither=str(opt.dither),
        dither_amount=float(opt.dither_amount), trim=bool(opt.trim),
        fill=str(opt.fill), **kw,
    )


def segment_subject(rgb: np.ndarray, alpha: np.ndarray | None = None,
                    opt: PixelArtOptions | None = None):
    """只做主体选择，返回 (alpha, info)。"""
    opt = opt or PixelArtOptions()
    if alpha is None:
        alpha = np.ones(rgb.shape[:2], np.float32)
    return segment.segment(rgb, alpha, sensitivity=opt.sensitivity,
                           center_bias=opt.center_bias, keep_shadow=opt.keep_shadow,
                           saliency=opt.saliency, subject=opt.subject, smart=opt.smart)


def pixelize_image(rgb: np.ndarray, alpha: np.ndarray, opt: PixelArtOptions | None = None):
    """只做像素化（alpha 已给定时），返回 (rgb_small, alpha_small, info)。"""
    opt = opt or PixelArtOptions()
    return pixelize.pixelize(rgb, alpha, _pixel_options(opt))


def to_pixel_art(rgb: np.ndarray, alpha: np.ndarray | None = None,
                 opt: PixelArtOptions | None = None):
    """一张图 -> 像素画：返回 (rgb_small, alpha_small, info)。

    rgb 为 sRGB 0..1 的 (H,W,3) float32；alpha 为 (H,W) 0..1（None 视为全不透明）。
    """
    opt = opt or PixelArtOptions()
    if alpha is None:
        alpha = np.ones(rgb.shape[:2], np.float32)
    a_used = np.ones(rgb.shape[:2], np.float32)
    seg_info: dict = {"method": "off", "bg_ratio": 0.0}
    if opt.remove_bg:
        a_used, seg_info = segment.segment(
            rgb, alpha, sensitivity=opt.sensitivity, center_bias=opt.center_bias,
            keep_shadow=opt.keep_shadow, saliency=opt.saliency, subject=opt.subject,
            smart=opt.smart)
    small_rgb, small_a, info = pixelize.pixelize(rgb, a_used, _pixel_options(opt))
    for k, v in seg_info.items():
        if isinstance(v, (int, float, str, bool)):
            info["seg_" + k] = v
    return small_rgb, small_a, info
