"""算法自检：用独立参考实现交叉验证核心算子（numpy + Pillow 即可运行）。

    python -m algorithms.selftest
"""
from __future__ import annotations

import numpy as np

from . import core, to_pixel_art, PixelArtOptions


def _ref_resize_area(img: np.ndarray, oh: int, ow: int) -> np.ndarray:
    """暴力面积均值（参考实现）。"""
    h, w = img.shape
    out = np.zeros((oh, ow), np.float64)
    ye, xe = np.linspace(0, h, oh + 1), np.linspace(0, w, ow + 1)
    for i in range(oh):
        for j in range(ow):
            y0, y1, x0, x1 = ye[i], ye[i + 1], xe[j], xe[j + 1]
            tot = 0.0
            for y in range(int(y0), min(int(np.ceil(y1)), h)):
                wy = max(0.0, min(y1, y + 1) - max(y0, y))
                for x in range(int(x0), min(int(np.ceil(x1)), w)):
                    wx = max(0.0, min(x1, x + 1) - max(x0, x))
                    tot += img[y, x] * wy * wx
            out[i, j] = tot / ((y1 - y0) * (x1 - x0))
    return out


def _ref_components(mask: np.ndarray, connectivity: int = 8):
    """逐像素 BFS（参考实现）。"""
    h, w = mask.shape
    offs = ([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            if connectivity == 8 else [(-1, 0), (1, 0), (0, -1), (0, 1)])
    seen = np.zeros((h, w), bool)
    sizes = []
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or seen[sy, sx]:
                continue
            stack, seen[sy, sx] = [(sy, sx)], True
            n = 0
            while stack:
                y, x = stack.pop()
                n += 1
                for dy, dx in offs:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            sizes.append(n)
    return sorted(sizes)


def main() -> int:
    rng = np.random.default_rng(0)
    fails: list[str] = []

    # 1) OKLab 往返
    rgb = rng.random((32, 32, 3)).astype(np.float32)
    err = float(np.abs(core.oklab_to_rgb(core.rgb_to_oklab(rgb)) - rgb).max())
    print(f"[1] OKLab 往返最大误差 {err:.2e}")
    if err > 1e-3:
        fails.append("oklab roundtrip")

    # 2) 面积重采样 vs 暴力
    img = rng.random((37, 29)).astype(np.float32)
    mine = core.resize_area(img, 8, 6)
    ref = _ref_resize_area(img, 8, 6)
    err = float(np.abs(mine - ref).max())
    print(f"[2] resize_area 与暴力实现最大误差 {err:.2e}")
    if err > 1e-4:
        fails.append("resize_area")

    # 3) 连通域 vs BFS（8/4 连通）
    for conn in (8, 4):
        for t in range(5):
            m = rng.random((23, 19)) > 0.6
            _lab, sizes = core.components(m, conn)
            if sorted(sizes) != _ref_components(m, conn):
                fails.append(f"components-{conn}")
                break
    print(f"[3] 连通域 8/4 连通与 BFS 参考一致: {'是' if not any(f.startswith('components') for f in fails) else '否'}")

    # 4) 填洞
    m = np.zeros((21, 21), bool)
    m[3:18, 3:18] = True
    m[8:12, 8:12] = False
    if not core.fill_holes(m)[9, 9]:
        fails.append("fill_holes")
    print(f"[4] 填洞: {'通过' if 'fill_holes' not in fails else '失败'}")

    # 5) 端到端：合成图 -> 像素画
    img2 = np.zeros((120, 120, 3), np.float32)
    img2[...] = 0.9
    img2[30:90, 30:90] = [0.2, 0.5, 0.85]
    g, a, info = to_pixel_art(img2, None, PixelArtOptions(size=32, palette=8))
    ok = g.shape == (32, 32, 3) and a.shape == (32, 32) and 0.0 <= float(a.min()) and float(a.max()) <= 1.0
    print(f"[5] 端到端 32x32: 形状 {g.shape} 不透明占比 {float((a > 0.5).mean()):.2f} "
          f"颜色 {info.get('palette_used')}")
    if not ok:
        fails.append("end2end")

    print("\n自检结果：", "全部通过" if not fails else f"失败项 {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
