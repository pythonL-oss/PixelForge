"""最小可运行示例：把一张图转成像素画（不依赖 GUI / 命令行工具）。

用法::

    python -m algorithms.demo 输入.png 输出.png --size 32
    python -m algorithms.demo 输入.png 输出目录 --size 16,32,64 --fill cover --colors 14
    python -m algorithms.demo 输入.png 输出.png --size 64 --no-bg --no-smart
"""
from __future__ import annotations

import argparse
import os

from . import PixelArtOptions, core, device, to_pixel_art


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m algorithms.demo",
                                description="PixelForge 算法引擎示例：图片 -> 像素画")
    p.add_argument("input", help="输入图片")
    p.add_argument("output", help="输出 PNG（单尺寸）或输出目录（多尺寸）")
    p.add_argument("-s", "--size", default="32", help="输出尺寸，逗号分隔，如 16,32,64")
    p.add_argument("-c", "--colors", type=int, default=16, help="调色板颜色数，0=不限制")
    p.add_argument("--fill", default="none", choices=["none", "cover", "stretch"],
                   help="none 居中留白 / cover 等比铺满（方块材质）/ stretch 拉伸铺满")
    p.add_argument("--no-bg", action="store_true", help="不抠背景（保留整张图）")
    p.add_argument("--no-smart", action="store_true", help="关闭智能主体检测")
    p.add_argument("--detail", type=float, default=0.75, help="细节保留强度 0..1")
    p.add_argument("--structure", type=int, default=3, help="SSIM 结构优化迭代次数 0..5")
    p.add_argument("--preview", type=int, default=0, help="额外输出放大 N 倍的预览图")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"], help="计算设备")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    os.environ["PIXELFORGE_DEVICE"] = args.device
    print("计算设备：", device.describe())

    rgb, alpha = core.load_image(args.input)
    sizes = [int(s) for s in str(args.size).replace("x", ",").split(",") if s.strip()]
    if not sizes:
        sizes = [32]
    base = os.path.splitext(os.path.basename(args.input))[0]
    as_file = args.output.lower().endswith(".png") and len(sizes) == 1

    for n in sizes:
        opt = PixelArtOptions(
            size=n, palette=int(args.colors), detail=float(args.detail),
            structure_passes=int(args.structure), fill=args.fill,
            remove_bg=not args.no_bg, smart=not args.no_smart,
        )
        small_rgb, small_a, info = to_pixel_art(rgb, alpha, opt)
        if as_file:
            out = args.output
        else:
            os.makedirs(args.output, exist_ok=True)
            out = os.path.join(args.output, f"{base}_{n}x{n}.png")
        core.save_rgba(out, small_rgb, small_a, 1)
        msg = (f"{out}  {n}x{n}  颜色 {info.get('palette_used')}  "
               f"不透明 {float((small_a > 0.5).mean()):.2f}")
        if info.get("seg_smart_choice"):
            msg += f"  智能检测：{info['seg_smart_choice']}({info.get('seg_smart_score')})"
        print(msg)
        if args.preview > 1:
            pv = os.path.splitext(out)[0] + f"_x{args.preview}.png"
            core.save_rgba(pv, small_rgb, small_a, int(args.preview))
            print("  预览图", pv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
