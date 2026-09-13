"""底层工具：色彩空间、重采样、滤波、形态学、连通域、k-means、显著性、导向滤波。

所有重数组运算都写成"类型保持"的形式（numpy 进 numpy 出、cupy 进 cupy 出），
因此同一条代码在 CPU 与 GPU(CuPy) 上都能跑；掩膜/连通域等小数组操作用 numpy。
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from . import device


# --------------------------------------------------------------------------
# 色彩空间
# --------------------------------------------------------------------------

def srgb_to_linear(c):
    xp = device.xp_for(c)
    c = xp.clip(xp.asarray(c, dtype=xp.float32), 0.0, 1.0)
    return xp.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(xp.float32)


def linear_to_srgb(c):
    xp = device.xp_for(c)
    c = xp.clip(xp.asarray(c, dtype=xp.float32), 0.0, 1.0)
    return xp.where(c <= 0.0031308, c * 12.92, 1.055 * xp.power(c, 1 / 2.4) - 0.055)


def rgb_to_oklab(rgb):
    """sRGB(0..1) -> OKLab，输入 (..., 3)。"""
    xp = device.xp_for(rgb)
    lin = srgb_to_linear(rgb)
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = xp.cbrt(l), xp.cbrt(m), xp.cbrt(s)
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    A = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    B = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return xp.stack([L, A, B], -1).astype(xp.float32)


def oklab_to_rgb(lab):
    xp = device.xp_for(lab)
    lab = xp.asarray(lab, dtype=xp.float32)
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    l_ = L + 0.3963377774 * A + 0.2158037573 * B
    m_ = L - 0.1055613458 * A - 0.0638541728 * B
    s_ = L - 0.0894841775 * A - 1.2914855480 * B
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    b = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    return xp.clip(linear_to_srgb(xp.stack([r, g, b], -1)), 0.0, 1.0)


def luminance(rgb):
    xp = device.xp_for(rgb)
    lin = srgb_to_linear(rgb)
    return (0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]).astype(xp.float32)


def lab_dist(a, b):
    """a:(...,3)；b:(...,3) 逐元素距离（b:(K,3) 时用 nearest_center 更划算）。"""
    xp = device.xp_for(a, b)
    if b.ndim == 1:
        d = a - b
        return xp.sqrt((d * d).sum(-1))
    diff = a[..., None, :] - b
    return xp.sqrt((diff * diff).sum(-1))


def nearest_center(lab, centers):
    """返回 (最近中心索引, 最近距离)。用 |a|^2+|b|^2-2ab 矩阵化，比广播差分快得多。"""
    xp = device.xp_for(lab, centers)
    a = xp.asarray(lab, dtype=xp.float32).reshape(-1, 3)
    b = xp.asarray(centers, dtype=xp.float32).reshape(-1, 3)
    d2 = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * (a @ b.T)
    idx = xp.argmin(d2, -1)
    d = xp.sqrt(xp.maximum(xp.take_along_axis(d2, idx[:, None], 1)[:, 0], 0.0))
    shape = lab.shape[:-1] if lab.ndim > 1 else (lab.shape[0],)
    return idx.reshape(shape), d.reshape(shape)


def xp_take_along(a, idx):
    xp = device.xp_for(a)
    return xp.take_along_axis(a, idx, -1)


# --------------------------------------------------------------------------
# 重采样（面积均值 / 双线性）
# --------------------------------------------------------------------------

def _cs(img, axis, xp):
    lead = xp.zeros_like(xp.take(img, xp.asarray([0]), axis=axis))
    return xp.concatenate([lead, xp.cumsum(img, axis=axis)], axis=axis)


def _sample(cs, pos, axis, n, xp):
    i0 = xp.clip(xp.floor(pos).astype(xp.int64), 0, n - 1)
    f = (pos - i0).astype(xp.float32)
    a = xp.take(cs, i0, axis=axis)
    b = xp.take(cs, xp.minimum(i0 + 1, n), axis=axis)
    shape = [1] * cs.ndim
    shape[axis] = len(pos)
    return a + (b - a) * f.reshape(shape)


def resize_area(img, out_h: int, out_w: int):
    """面积均值重采样（降采样精确，升采样近似双线性）。支持 (H,W) 与 (H,W,C)。"""
    xp = device.xp_for(img)
    arr = xp.asarray(img, dtype=xp.float32)
    two_dim = arr.ndim == 2
    if two_dim:
        arr = arr[..., None]
    h, w, _ = arr.shape
    out_h, out_w = max(1, int(out_h)), max(1, int(out_w))
    if h == out_h and w == out_w:
        return arr[..., 0] if two_dim else arr
    res = arr
    if w != out_w:
        c = _cs(res, 1, xp)
        e = xp.linspace(0.0, w, out_w + 1)
        s = (_sample(c, e[1:], 1, w, xp) - _sample(c, e[:-1], 1, w, xp)) / (e[1:] - e[:-1]).reshape(1, -1, 1)
        res = s
    if h != out_h:
        c = _cs(res, 0, xp)
        e = xp.linspace(0.0, h, out_h + 1)
        s = (_sample(c, e[1:], 0, h, xp) - _sample(c, e[:-1], 0, h, xp)) / (e[1:] - e[:-1]).reshape(-1, 1, 1)
        res = s
    lo = min(float(xp.min(arr)), 0.0)
    hi = max(float(xp.max(arr)), 1.0)
    res = xp.clip(res, lo, hi)
    return res[..., 0] if two_dim else res


def resize_bilinear(img, out_h: int, out_w: int):
    xp = device.xp_for(img)
    arr = xp.asarray(img, dtype=xp.float32)
    two_dim = arr.ndim == 2
    if two_dim:
        arr = arr[..., None]
    h, w, _ = arr.shape
    out_h, out_w = max(1, int(out_h)), max(1, int(out_w))

    def axis_interp(a, n, out_n, axis):
        pos = (xp.arange(out_n, dtype=xp.float32) + 0.5) * (n / out_n) - 0.5
        i0 = xp.clip(xp.floor(pos).astype(xp.int64), 0, n - 1)
        f = xp.clip(pos - i0, 0.0, 1.0)
        v0 = xp.take(a, i0, axis=axis)
        v1 = xp.take(a, xp.minimum(i0 + 1, n - 1), axis=axis)
        shape = [1] * a.ndim
        shape[axis] = out_n
        return v0 + (v1 - v0) * f.reshape(shape)

    res = axis_interp(arr, w, out_w, 1)
    res = axis_interp(res, h, out_h, 0)
    return xp.clip(res, 0.0, 1.0)[..., 0] if two_dim else xp.clip(res, 0.0, 1.0)


def box_down2(img):
    """2 倍盒式降采样（奇数边裁掉最后一行/列）。"""
    xp = device.xp_for(img)
    arr = xp.asarray(img)
    h, w = arr.shape[:2]
    h2, w2 = h - h % 2, w - w % 2
    a = arr[:h2, :w2]
    if a.ndim == 2:
        return a.reshape(h2 // 2, 2, w2 // 2, 2).mean(axis=(1, 3))
    c = a.shape[2]
    return a.reshape(h2 // 2, 2, w2 // 2, 2, c).mean(axis=(1, 3))


# --------------------------------------------------------------------------
# 滤波
# --------------------------------------------------------------------------

def gauss_kernel(sigma: float) -> np.ndarray:
    r = max(1, int(round(sigma * 3)))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-(x ** 2) / (2 * sigma * sigma))
    return (k / k.sum()).astype(np.float32)


def convolve1d(img, kernel, axis: int):
    xp = device.xp_for(img)
    arr = xp.asarray(img, dtype=xp.float32)
    k = device.as_device(np.asarray(kernel, np.float32)) if xp is not np else np.asarray(kernel, np.float32)
    r = len(kernel) // 2
    pad = [(0, 0)] * arr.ndim
    pad[axis] = (r, r)
    p = xp.pad(arr, pad, mode="reflect")
    out = xp.zeros_like(arr)
    for i, kv in enumerate(kernel):
        if kv == 0:
            continue
        sl = [slice(None)] * arr.ndim
        sl[axis] = slice(i, i + arr.shape[axis])
        out = out + float(kv) * p[tuple(sl)]
    return out


def gaussian_blur(img, sigma: float):
    if sigma <= 0:
        return img
    k = gauss_kernel(sigma)
    return convolve1d(convolve1d(img, k, 1), k, 0)


def box_blur(img, radius: int):
    if radius <= 0:
        return img
    k = np.ones(radius * 2 + 1, np.float32) / (radius * 2 + 1)
    return convolve1d(convolve1d(img, k, 1), k, 0)


def box_mean(img, radius: int):
    """积分图实现的半径 r 均值滤波（O(1)/像素），边界按 edge 处理。"""
    xp = device.xp_for(img)
    a = xp.asarray(img, dtype=xp.float32)
    r = int(max(0, radius))
    if r == 0:
        return a
    squeeze = a.ndim == 2
    if squeeze:
        a = a[..., None]
    p = xp.pad(a, ((r, r), (r, r), (0, 0)), mode="edge")
    cs = xp.cumsum(xp.cumsum(p, axis=0), axis=1)
    zero = xp.zeros((1,) + cs.shape[1:], cs.dtype)
    cs = xp.concatenate([zero, cs], axis=0)
    zero2 = xp.zeros((cs.shape[0], 1) + cs.shape[2:], cs.dtype)
    cs = xp.concatenate([zero2, cs], axis=1)
    k = 2 * r + 1
    h, w = a.shape[:2]
    out = (cs[k:k + h, k:k + w] - cs[0:h, k:k + w] - cs[k:k + h, 0:w] + cs[0:h, 0:w]) / (k * k)
    return out[..., 0] if squeeze else out


def guided_filter(guide, src, radius: int = 4, eps: float = 1e-3):
    """导向滤波：让 src(如 alpha) 的边缘贴合 guide(如亮度) 的边缘。"""
    xp = device.xp_for(guide, src)
    I = xp.asarray(guide, dtype=xp.float32)
    p = xp.asarray(src, dtype=xp.float32)
    mI = box_mean(I, radius)
    mp = box_mean(p, radius)
    corrI = box_mean(I * I, radius)
    corrIp = box_mean(I * p, radius)
    varI = corrI - mI * mI
    covIp = corrIp - mI * mp
    a = covIp / (varI + eps)
    b = mp - a * mI
    return box_mean(a, radius) * I + box_mean(b, radius)


def gradient_magnitude(lum, sigma: float = 1.0):
    """梯度强度图，归一化到 99 分位。"""
    xp = device.xp_for(lum)
    a = xp.asarray(lum, dtype=xp.float32)
    if sigma > 0:
        a = gaussian_blur(a, sigma)
    gx = xp.zeros_like(a)
    gy = xp.zeros_like(a)
    gx = xp.concatenate([xp.zeros_like(a[:, :1]), (a[:, 2:] - a[:, :-2]) * 0.5, xp.zeros_like(a[:, :1])], 1) \
        if a.shape[1] > 2 else gx
    gy = xp.concatenate([xp.zeros_like(a[:1, :]), (a[2:, :] - a[:-2, :]) * 0.5, xp.zeros_like(a[:1, :])], 0) \
        if a.shape[0] > 2 else gy
    g = xp.sqrt(gx * gx + gy * gy)
    scale = float(xp.percentile(g, 99)) if g.size else 1.0
    return xp.clip(g / max(scale, 1e-6), 0.0, 1.0).astype(xp.float32)


# --------------------------------------------------------------------------
# 显著性 / 边缘密度（主体选择用）
# --------------------------------------------------------------------------

def center_surround(rgb, sigmas=(2.0, 4.0, 8.0)):
    """多尺度中心-周边对比：与局部均值差得越多越显著。"""
    xp = device.xp_for(rgb)
    a = xp.asarray(rgb, dtype=xp.float32)
    acc = None
    for s in sigmas:
        d = xp.sqrt(((a - gaussian_blur(a, s)) ** 2).sum(-1))
        d = gaussian_blur(d, s)
        acc = d if acc is None else acc + d
    acc = acc / len(sigmas)
    scale = float(xp.percentile(acc, 97))
    return xp.clip(acc / max(scale, 1e-6), 0.0, 1.0).astype(xp.float32)


def spectral_saliency(lum, size: int = 64):
    """频域显著图（谱残差）。输入亮度图，输出与输入同尺寸的 0..1 显著度。"""
    xp = device.xp_for(lum)
    a = xp.asarray(lum, dtype=xp.float32)
    h, w = a.shape[:2]
    small = resize_area(a, min(size, max(2, h)), min(size, max(2, w)))
    if float(xp.std(small)) < 1e-6:
        return xp.zeros_like(a)
    f = xp.fft.fft2(small)
    mag = xp.abs(f)
    phase = f / (mag + 1e-9)
    logmag = xp.log(mag + 1e-6)
    resid = logmag - gaussian_blur(logmag, 2.0)
    sal = xp.abs(xp.fft.ifft2(xp.exp(resid + 1j * xp.angle(f)) * 1.0)) ** 2
    sal = gaussian_blur(sal.real if hasattr(sal, "real") else sal, 2.0)
    scale = float(xp.percentile(sal, 97))
    sal = xp.clip(sal / max(scale, 1e-6), 0.0, 1.0)
    return xp.clip(resize_bilinear(sal.astype(xp.float32), h, w), 0.0, 1.0)


def edge_density(lum, radius: int = 3):
    """边缘密度：强梯度所占比例，用来识别"纹理很花的背景"。"""
    xp = device.xp_for(lum)
    g = xp.asarray(gradient_magnitude(lum, 1.0), dtype=xp.float32)
    return xp.clip(box_mean((g > 0.25).astype(xp.float32), radius), 0.0, 1.0)


# --------------------------------------------------------------------------
# 形态学 / 连通域（小数组，numpy）
# --------------------------------------------------------------------------

_OFF8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _shift(m, dy, dx, fill):
    out = np.full_like(m, fill)
    h, w = m.shape
    ys0, ys1 = max(0, dy), min(h, h + dy)
    xs0, xs1 = max(0, dx), min(w, w + dx)
    out[ys0:ys1, xs0:xs1] = m[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    return out


def dilate(mask, iters: int = 1):
    m = np.asarray(mask).astype(bool)
    for _ in range(max(0, iters)):
        acc = m.copy()
        for dy, dx in _OFF8:
            acc |= _shift(m, dy, dx, False)
        m = acc
    return m


def erode(mask, iters: int = 1):
    return ~dilate(~np.asarray(mask).astype(bool), iters)


def max_filter(a, radius: int):
    out = np.asarray(a, np.float32).copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy or dx:
                out = np.maximum(out, _shift(out, dy, dx, -1e9))
    return out


def components(mask, connectivity: int = 8):
    """连通域标记：行程(run)扫描 + 并查集。返回 (labels, sizes)。

    比逐像素 BFS 快一个数量级；8 连通允许对角相接，4 连通要求共边。
    """
    m = np.asarray(mask).astype(bool)
    h, w = m.shape
    labels = np.zeros((h, w), np.int32)
    if not m.any():
        return labels, []
    pad = np.zeros((h, w + 2), bool)
    pad[:, 1:w + 1] = m
    d = np.diff(pad.astype(np.int8), axis=1)
    sy, sx = np.nonzero(d == 1)
    _ey, ex = np.nonzero(d == -1)
    nr = len(sy)
    if nr == 0:
        return labels, []
    first = np.append(np.searchsorted(sy, np.arange(h), side="left"), nr)
    parent = np.arange(nr, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return int(x)

    gap = 1 if connectivity == 8 else 0
    for y in range(h - 1):
        a0, a1 = int(first[y]), int(first[y + 1])
        b0, b1 = a1, int(first[y + 2])
        if a1 == a0 or b1 == b0:
            continue
        sa, ea = sx[a0:a1], ex[a0:a1]
        sb, eb = sx[b0:b1], ex[b0:b1]
        k0 = np.searchsorted(ea, sb - gap + 1, side="left")
        k1 = np.searchsorted(sa, eb - 1 + gap, side="right")
        cnt = np.maximum(k1 - k0, 0)
        tot = int(cnt.sum())
        if tot == 0:
            continue
        js = np.repeat(np.arange(len(sb)), cnt)
        offs = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        ts = np.repeat(k0, cnt) + offs
        for t, j in zip(ts.tolist(), js.tolist()):
            ra = find(a0 + t)
            rb = find(b0 + j)
            if ra != rb:
                if ra < rb:
                    parent[rb] = ra
                else:
                    parent[ra] = rb
    roots = np.array([find(i) for i in range(nr)], dtype=np.int64)
    _uniq, inv = np.unique(roots, return_inverse=True)
    run_label = (inv.reshape(-1) + 1).astype(np.int32)
    for i in range(nr):
        labels[sy[i], sx[i]:ex[i]] = run_label[i]
    counts = np.bincount(run_label, weights=(ex - sx).astype(np.float64))
    return labels, [int(v) for v in counts[1:]]


def fill_holes(mask):
    """填洞：对背景做连通域，未接触到画框的背景区域即内部空洞。"""
    m = np.asarray(mask).astype(bool)
    if not m.any() or m.all():
        return m
    labels, sizes = components(~m, connectivity=4)   # 背景用 4 连通（与前景 8 连通互补）
    if not sizes:
        return m
    border = set(int(v) for v in np.unique(np.concatenate(
        [labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])) if v > 0)
    holes = np.zeros(len(sizes) + 1, bool)
    for cid in range(1, len(sizes) + 1):
        holes[cid] = cid not in border
    return m | holes[labels]


def hysteresis(mask_strong, mask_weak, growth: int = 1):
    out = np.asarray(mask_strong).astype(bool) & np.asarray(mask_weak).astype(bool)
    if not out.any():
        out = np.asarray(mask_strong).astype(bool)
    prev = out.copy()
    for _ in range(max(0, growth)):
        grown = prev.copy()
        for dy, dx in _OFF8:
            grown |= _shift(prev, dy, dx, False)
        out = grown & np.asarray(mask_weak).astype(bool)
        if out.sum() == prev.sum():
            break
        prev = out
    return out


def remove_small(mask, min_area: int):
    labels, sizes = components(mask)
    if not sizes:
        return np.asarray(mask).astype(bool)
    keep = np.zeros(len(sizes) + 1, bool)
    for i, sz in enumerate(sizes):
        keep[i + 1] = sz >= min_area
    return keep[labels]


def keep_components(mask, rel_area: float = 0.12, min_area: int = 4):
    labels, sizes = components(mask)
    if not sizes:
        return np.asarray(mask).astype(bool)
    order = np.argsort(sizes)[::-1]
    biggest = sizes[order[0]]
    keep = np.zeros(len(sizes) + 1, bool)
    for i in order:
        if sizes[i] >= max(min_area, biggest * rel_area):
            keep[i + 1] = True
    return keep[labels]


def component_stats(mask, labels=None, sizes=None):
    """每个连通域的统计：面积、包围盒、中心、紧致度、贴边程度。"""
    if labels is None:
        labels, sizes = components(mask)
    sizes = sizes or []
    out = []
    h, w = mask.shape
    for cid in range(1, len(sizes) + 1):
        ys, xs = np.where(labels == cid)
        if len(ys) == 0:
            continue
        bh, bw = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        fill = sizes[cid - 1] / float(bh * bw)
        cy, cx = float(ys.mean()) / h, float(xs.mean()) / w
        center_score = 1.0 - min(1.0, float(np.hypot(cy - 0.5, cx - 0.5)) / 0.7071)
        out.append({"id": cid, "area": int(sizes[cid - 1]),
                    "bbox": (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())),
                    "fill": fill, "center": center_score, "centroid": (cy, cx),
                    "touch": float(np.concatenate([labels[0, :] == cid, labels[-1, :] == cid,
                                                   labels[:, 0] == cid, labels[:, -1] == cid]).mean())})
    return out


def downsample_mask(mask, max_side: int = 256):
    """把掩膜降到 max_side 以内（用于结构性清理，避免全分辨率 BFS）。"""
    h, w = mask.shape[:2]
    if max(h, w) <= max_side:
        return np.asarray(mask, np.float32), 1.0
    sc = max_side / max(h, w)
    return resize_area(np.asarray(mask, np.float32), max(8, int(round(h * sc))),
                       max(8, int(round(w * sc)))), sc


def cleanup_mask(mask, min_area: int = 3, fill: bool = True, max_side: int = 256):
    """低分辨率做连通域/填洞清理后再放大（结果几乎一致，但快很多）。"""
    from PIL import Image
    h, w = mask.shape[:2]
    small, _sc = downsample_mask(mask, max_side)
    b = small > 0.35
    if b.any():
        if min_area > 1:
            b = remove_small(b, min_area)
        if fill:
            b = fill_holes(b)
    if b.shape != (h, w):
        im = Image.fromarray((b.astype(np.uint8) * 255), "L").resize((w, h), Image.BILINEAR)
        return np.asarray(im).astype(np.float32) / 255.0 > 0.5
    return b


# --------------------------------------------------------------------------
# k-means（OKLab，支持权重，类型保持）
# --------------------------------------------------------------------------

def kmeans(points, k: int, weights=None, iters: int = 16, seed: int = 12345, sample_cap: int = 20000):
    xp = device.xp_for(points)
    pts = xp.asarray(points, dtype=xp.float32).reshape(-1, 3)
    n = len(pts)
    if n == 0:
        return xp.zeros((0, 3), xp.float32), xp.zeros(0, xp.int32)
    k = int(max(1, min(k, n)))
    w = xp.ones(n, xp.float32) if weights is None else xp.asarray(weights, xp.float32).reshape(-1)
    w = xp.clip(w, 1e-6, None)

    # 播种在主机上做（k-means++），避免设备端 rng 差异
    host = device.as_numpy(pts)
    hw = device.as_numpy(w)
    if n > sample_cap:
        step = int(np.ceil(n / sample_cap))
        host_s, hw_s = host[::step], hw[::step]
    else:
        host_s, hw_s = host, hw
    rng = np.random.default_rng(seed)
    p = hw_s / hw_s.sum()
    centers = [host_s[int(rng.choice(len(host_s), p=p))]]
    d2 = ((host_s - centers[0]) ** 2).sum(-1)
    for _ in range(k - 1):
        prob = d2 * hw_s
        s = prob.sum()
        idx = int(rng.choice(len(host_s), p=prob / s)) if s > 0 else int(rng.integers(0, len(host_s)))
        centers.append(host_s[idx])
        d2 = np.minimum(d2, ((host_s - host_s[idx]) ** 2).sum(-1))
    C = xp.asarray(np.array(centers, np.float32))

    labels = xp.zeros(n, xp.int32)
    for _ in range(iters):
        d_all = lab_dist(pts, C)
        labels = d_all.argmin(-1)
        newC = C.copy()
        for j in range(len(C)):
            sel = labels == j
            s = int(sel.sum())
            if s > 0:
                ww = w * sel
                newC = _set_row(newC, j, (pts * ww[:, None]).sum(0) / xp.maximum(ww.sum(), 1e-6))
            else:
                dmin = host_dmin(d_all, labels)
                far = int(xp.argmax(xp.asarray(dmin * hw if len(dmin) == n else dmin)))
                newC = _set_row(newC, j, pts[far])
        if float(xp.max(xp.abs(newC - C))) < 1e-5:
            C = newC
            break
        C = newC
    return C, lab_dist(pts, C).argmin(-1)


def host_dmin(d_all, labels):
    xp = device.xp_for(d_all)
    d = xp.take_along_axis(d_all, labels[:, None], 1)[:, 0]
    return device.as_numpy(d)


def _set_row(C, j, row):
    xp = device.xp_for(C)
    mask = xp.arange(len(C)) == j
    return xp.where(mask[:, None], row[None, :], C)


def merge_close_colors(centers, weights, min_dist: float = 0.014):
    """合并感知上几乎相同的调色板项（主机侧循环，规模很小）。"""
    C = device.as_numpy(centers)
    W = device.as_numpy(weights)
    order = np.argsort(W)[::-1]
    kept: list[np.ndarray] = []
    kept_w: list[float] = []
    for i in order:
        c = C[i]
        hit = -1
        for j, kc in enumerate(kept):
            if np.linalg.norm(c - kc) <= min_dist:
                hit = j
                break
        if hit < 0:
            kept.append(c)
            kept_w.append(float(W[i]))
        else:
            tot = kept_w[hit] + float(W[i])
            kept[hit] = (kept[hit] * kept_w[hit] + c * float(W[i])) / tot
            kept_w[hit] = tot
    xp = device.xp_for(centers)
    return xp.asarray(np.array(kept, np.float32)) if kept else centers


# --------------------------------------------------------------------------
# 图像 I/O
# --------------------------------------------------------------------------

def load_image(path: str) -> tuple[np.ndarray, np.ndarray]:
    """返回 (rgb float32 HxWx3, alpha float32 HxW)，sRGB 0..1。"""
    with Image.open(path) as im:
        has_alpha = im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info)
        im = im.convert("RGBA" if has_alpha else "RGB")
        arr = np.asarray(im).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.shape[2] == 4:
        return arr[..., :3].copy(), arr[..., 3].copy()
    return arr[..., :3].copy(), np.ones(arr.shape[:2], np.float32)


def save_rgba(path: str, rgb, alpha, scale: int = 1) -> None:
    rgb = device.as_numpy(rgb)
    alpha = device.as_numpy(alpha)
    a = np.clip(alpha, 0, 1)
    if scale > 1:
        rgb = np.repeat(np.repeat(rgb, scale, 0), scale, 1)
        a = np.repeat(np.repeat(a, scale, 0), scale, 1)
    rgba = np.concatenate([np.clip(rgb, 0, 1), a[..., None]], -1)
    Image.fromarray((rgba * 255.0 + 0.5).astype(np.uint8), "RGBA").save(path, "PNG", optimize=True)


def checkerboard(h: int, w: int, cell: int = 4) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    c = ((yy // cell + xx // cell) % 2).astype(np.float32)
    return 0.72 + 0.13 * c
