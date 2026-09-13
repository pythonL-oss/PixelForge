"""像素化核心：主体网格适配 + 细节保留降采样 + SSIM 结构优化 + 调色板 ICM。

与"把图缩小"的本质区别
----------------------
1. 先按主体包围盒做等比重排，让主体正落在输出像素网格上（既不丢主体也不被拉变形）。
2. 每个输出像素的颜色不是简单取平均，而是在"均值 / 重要性加权均值 / 纯净主体色 /
   峰值细节色 / 最亮极值 / 最暗极值"等候选色中挑选；挑选依据是 SSIM 结构相似度在
   2 倍分辨率参考图上的局部窗口匹配，因此细剑刃、眼睛、高光、阴影等低分辨率下的
   关键特征会被显式保留，而不是被平均抹平。
3. 结构优化是可分离双线性的精确坐标下降：改动一格只影响 4×4 子像素、进而只影响
   8×8 个 SSIM 窗口，逐个候选精确求值，不是启发式涂抹。
4. 最后在 OKLab 感知空间做调色板聚类 + Potts 平滑项 ICM，压到固定色数并去掉孤立噪点。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import core as C

_OFF8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


@dataclass
class PixelOptions:
    out_w: int = 32
    out_h: int = 32
    palette: int = 24
    detail: float = 0.75
    structure_passes: int = 3
    supersample: int = 4
    margin: int = 1
    contrast: float = 0.45
    saturation: float = 0.5
    outline: int = 0                      # 0 无 / 1 内部描边 / 2 外部描边
    outline_color: tuple = (0.10, 0.09, 0.13)
    palette_smooth: float = 0.55
    fill: str = "none"                    # none 居中留白 / cover 等比铺满 / stretch 拉伸铺满
    dither: str = "none"                  # none / ordered / floyd
    dither_amount: float = 0.6
    trim: bool = False                    # 去掉四周空白，让主体更饱满
    palette_file: str = ""
    bg: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# 基础几何
# --------------------------------------------------------------------------

def _fit_cells(cw: int, ch: int, avail_w: int, avail_h: int) -> tuple[int, int]:
    if cw <= 0 or ch <= 0:
        return max(1, avail_w), max(1, avail_h)
    s = min(avail_w / cw, avail_h / ch)
    fw = max(1, min(avail_w, int(round(cw * s))))
    fh = max(1, min(avail_h, int(round(ch * s))))
    return fw, fh


def _subject_bbox(alpha: np.ndarray, pad_ratio: float = 0.01):
    h, w = alpha.shape
    m = alpha > 0.45
    if not m.any():
        return 0, 0, w, h
    ys = np.where(m.any(1))[0]
    xs = np.where(m.any(0))[0]
    y0, y1 = int(ys[0]), int(ys[-1]) + 1
    x0, x1 = int(xs[0]), int(xs[-1]) + 1
    py = max(1, int(round((y1 - y0) * pad_ratio)))
    px = max(1, int(round((x1 - x0) * pad_ratio)))
    return max(0, x0 - px), max(0, y0 - py), min(w, x1 + px), min(h, y1 + py)


def _preshrink(rgb, a, max_side: int = 1200):
    """超大幅面先做整数倍盒式降采样（精确且极快），再做精确面积重采样。"""
    h, w = rgb.shape[:2]
    if max(h, w) <= max_side * 2:
        return rgb, a
    while max(h, w) > max_side * 2 and h % 2 == 0 and w % 2 == 0:
        rgb = C.box_down2(rgb)
        a = C.box_down2(a)
        h, w = rgb.shape[:2]
    return rgb, a


def _premult_resize(rgb, a, out_h, out_w):
    rgb, a = _preshrink(rgb, a, max(out_h, out_w) * 2)
    num = C.resize_area(rgb * a[..., None], out_h, out_w)
    den = C.resize_area(a, out_h, out_w)
    plain = C.resize_area(rgb, out_h, out_w)
    out = np.where(den[..., None] < 2e-4, plain, num / np.maximum(den, 1e-5)[..., None])
    return np.clip(out, 0, 1).astype(np.float32), np.clip(den, 0, 1).astype(np.float32)


def _block(x: np.ndarray, s: int, oh: int, ow: int) -> np.ndarray:
    """(oh*s, ow*s[, C]) -> (oh, ow, s*s[, C])。"""
    if x.ndim == 2:
        return x.reshape(oh, s, ow, s).transpose(0, 2, 1, 3).reshape(oh, ow, s * s)
    c = x.shape[2]
    return x.reshape(oh, s, ow, s, c).transpose(0, 2, 1, 3, 4).reshape(oh, ow, s * s, c)


def _shift2(m: np.ndarray, dy: int, dx: int, fill=0):
    out = np.full_like(m, fill)
    h, w = m.shape[:2]
    ys0, ys1 = max(0, dy), min(h, h + dy)
    xs0, xs1 = max(0, dx), min(w, w + dx)
    if ys0 >= ys1 or xs0 >= xs1:
        return out
    out[ys0:ys1, xs0:xs1] = m[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    return out


def _fill_transparent(rgb: np.ndarray, a: np.ndarray, iters: int = 48):
    """把透明区域的颜色用最近的主体色填满（避免参考图里混入背景色）。"""
    filled = rgb.copy()
    known = a > 0.6
    if not known.any():
        return filled, known
    cur = known.copy()
    for _ in range(iters):
        if cur.all():
            break
        acc = np.zeros_like(filled)
        cnt = np.zeros(cur.shape, np.float32)
        for dy, dx in _OFF8:
            shifted_c = _shift2(cur, dy, dx, False)
            shifted_rgb = _shift2(filled, dy, dx, 0.0)
            acc += shifted_rgb * shifted_c[..., None]
            cnt += shifted_c.astype(np.float32)
        new_px = (~cur) & (cnt > 0)
        if not new_px.any():
            break
        filled[new_px] = acc[new_px] / cnt[new_px][:, None]
        cur |= new_px
    return filled, known


# --------------------------------------------------------------------------
# 重要性（细节强度）图
# --------------------------------------------------------------------------

def _importance(rgb: np.ndarray, a: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    sigma = max(0.7, min(h, w) / 400.0)
    lum = C.luminance(rgb)
    edge = C.gradient_magnitude(lum, sigma)
    aw = a
    den = np.maximum(C.box_blur(aw, 2), 1e-3)
    local = C.box_blur(rgb * aw[..., None], 2) / den[..., None]
    dev = np.sqrt(((rgb - local) ** 2).sum(-1))
    scale = max(float(np.percentile(dev, 95)), 1e-3)
    dev = np.clip(dev / scale, 0, 1)
    dev = C.gaussian_blur(dev, max(0.6, sigma))
    imp = np.clip(0.62 * edge + 0.55 * dev, 0, 1)
    return (imp * (a > 0.02)).astype(np.float32)


# --------------------------------------------------------------------------
# 画布构建：把主体放进 N*S 网格
# --------------------------------------------------------------------------

def _build_canvas(rgb: np.ndarray, alpha: np.ndarray, opt: PixelOptions) -> dict:
    S = int(max(2, opt.supersample))
    ow, oh = int(opt.out_w), int(opt.out_h)
    x0, y0, x1, y1 = _subject_bbox(alpha)
    crop_rgb = rgb[y0:y1, x0:x1]
    crop_a = alpha[y0:y1, x0:x1]
    cw, ch = max(1, x1 - x0), max(1, y1 - y0)

    mode = str(getattr(opt, "fill", "none") or "none").lower()
    if mode == "stretch":
        fw, fh = ow, oh                       # 直接拉伸，允许变形
    elif mode == "cover":
        sc = max(ow / max(cw, 1), oh / max(ch, 1))
        fw, fh = max(ow, int(np.ceil(cw * sc))), max(oh, int(np.ceil(ch * sc)))
    else:
        avail_w = max(1, ow - 2 * opt.margin)
        avail_h = max(1, oh - 2 * opt.margin)
        fw, fh = _fit_cells(cw, ch, avail_w, avail_h)
    fit_w, fit_h = fw * S, fh * S

    work_scale = min(1.0, 1024.0 / max(cw, ch))
    ww, wh = max(8, int(round(cw * work_scale))), max(8, int(round(ch * work_scale)))
    work_rgb, work_a = _premult_resize(crop_rgb, crop_a, wh, ww)
    imp = _importance(work_rgb, work_a)

    a_core = work_a ** 4
    w_bias = (0.3 + 0.7 * imp) * work_a
    stack = np.concatenate([
        work_a[..., None],
        w_bias[..., None],
        work_rgb * w_bias[..., None],
        a_core[..., None],
        work_rgb * a_core[..., None],
        work_rgb * work_a[..., None],
        imp[..., None],
    ], -1).astype(np.float32)
    st = C.resize_area(stack, fit_h, fit_w)

    CH, CW = oh * S, ow * S
    canvas = np.zeros((CH, CW, stack.shape[-1]), np.float32)
    ox, oy = ((ow - fw) * S) // 2, ((oh - fh) * S) // 2     # 铺满模式可能为负（居中裁切）
    sx0, sy0 = max(0, -ox), max(0, -oy)
    dx0, dy0 = max(0, ox), max(0, oy)
    pw = min(fit_w - sx0, CW - dx0)
    ph = min(fit_h - sy0, CH - dy0)
    if pw > 0 and ph > 0:
        canvas[dy0:dy0 + ph, dx0:dx0 + pw] = st[sy0:sy0 + ph, sx0:sx0 + pw]

    chan = {
        "a": canvas[..., 0],
        "a_imp": canvas[..., 1],
        "rgb_imp": canvas[..., 2:5],
        "a_core": canvas[..., 5],
        "rgb_core": canvas[..., 6:9],
        "rgb_a": canvas[..., 9:12],
        "imp": canvas[..., 12],
    }
    return {"chan": chan, "S": S, "ow": ow, "oh": oh, "CW": CW, "CH": CH,
            "fit": (fw, fh), "crop": (cw, ch), "offset": (ox, oy)}


def _cell_stats(cv: dict) -> dict:
    """逐格统计：覆盖率、均值色、重要性加权色、纯净主体色、亮/暗极值色、细节强度。"""
    S, ow, oh = cv["S"], cv["ow"], cv["oh"]
    ch = cv["chan"]
    a_b = _block(ch["a"], S, oh, ow)
    A = a_b.mean(-1)
    peak_a = a_b.max(-1)
    a_imp = _block(ch["a_imp"], S, oh, ow)
    sum_imp = a_imp.sum(-1)
    c_imp = _block(ch["rgb_imp"], S, oh, ow).sum(-2) / np.maximum(sum_imp, 1e-5)[..., None]
    a_core = _block(ch["a_core"], S, oh, ow)
    core_cnt = (a_core > 0.01).sum(-1)
    c_core = _block(ch["rgb_core"], S, oh, ow).sum(-2) / np.maximum(a_core.sum(-1), 1e-5)[..., None]
    c_mean = _block(ch["rgb_a"], S, oh, ow).sum(-2) / np.maximum(a_b.sum(-1), 1e-5)[..., None]
    rgb_norm = np.clip(_block(ch["rgb_a"], S, oh, ow) /
                       np.maximum(a_b, 1e-4)[..., None], 0, 1)
    has_core = (core_cnt >= 2) & (a_core.sum(-1) > 0.25)
    use = np.where(has_core[..., None], a_core, a_b)
    rgb_use = rgb_norm
    lum_b = C.luminance(rgb_use)
    imp_b = _block(ch["imp"], S, oh, ow)
    imp_c = imp_b.mean(-1)
    w_imp = (imp_b + 0.05) * use
    pk = np.argmax(w_imp, -1)
    c_peak = np.take_along_axis(rgb_use, pk[..., None, None].repeat(3, -1), axis=2)[:, :, 0, :]

    total = max(1, S * S)
    valid = use > 0.01
    lmax = np.where(valid, lum_b, 0.0).max(-1)
    lmin = np.where(valid, lum_b, 0.0).min(-1)

    ref = np.where(has_core[..., None], c_core,
                   np.where(sum_imp[..., None] > 0.05, c_imp, c_mean))
    nan0 = lambda x: np.clip(np.nan_to_num(x, nan=0.0), 0, 1)  # noqa: E731
    return {"A": A, "peak_a": peak_a, "core": has_core, "c_mean": nan0(c_mean),
            "c_imp": nan0(c_imp), "c_core": nan0(c_core), "core_cov": core_cnt / total,
            "c_peak": nan0(c_peak),
            "ref": nan0(ref), "imp": imp_c, "lum_range": np.maximum(lmax - lmin, 0.0)}


# --------------------------------------------------------------------------
# 透明度：阈值 + 细线保留
# --------------------------------------------------------------------------

def _alpha_mask(A: np.ndarray, peak_a: np.ndarray, thresh: float = 0.5):
    """阈值吸附 + 细线保留。返回 (mask, 被救回的细线格)。

    细线判定：沿某方向两端都在主体内，且垂直方向 1~2 格之外全是背景
    （这样才能和"实心主体的一条直边界"区分开）。"""
    strong = A >= thresh
    if strong.sum() < 4:
        strong = A >= max(0.22, thresh * 0.5)
    weak = (A >= max(0.10, thresh * 0.25)) & (peak_a >= 0.9) & ~strong
    mask = strong.copy()
    added = np.zeros_like(mask)
    dirs = (((0, 1), (1, 0)), ((1, 0), (0, 1)), ((1, 1), (1, -1)), ((1, -1), (1, 1)))
    for _ in range(10):
        add = np.zeros_like(mask)
        for (dy, dx), (py, px) in dirs:
            n1 = _shift2(mask, dy, dx, False)
            n2 = _shift2(mask, -dy, -dx, False)
            p1 = _shift2(mask, py, px, False) | _shift2(mask, 2 * py, 2 * px, False)
            p2 = _shift2(mask, -py, -px, False) | _shift2(mask, -2 * py, -2 * px, False)
            add |= n1 & n2 & ~(p1 | p2)
        add &= weak & ~mask
        if not add.any():
            break
        mask |= add
        added |= add
    return mask, added


# --------------------------------------------------------------------------
# SSIM 结构优化（精确局部坐标下降）
# --------------------------------------------------------------------------

_x5 = np.arange(-2, 3, dtype=np.float32)
_k5 = np.exp(-(_x5 ** 2) / (2 * 1.2 ** 2)); _k5 /= _k5.sum()
_W5 = np.outer(_k5, _k5).astype(np.float32)
_C1 = 0.01 ** 2
_C2 = 0.03 ** 2


def _windowed_stats(u: np.ndarray, d: np.ndarray):
    """u,d: (..., n, n) -> 每个 5x5 高斯窗口的 (mu_u, mu_d, su, sd, cov)，形状 (..., n-4, n-4)。"""
    k = _W5
    sw = np.lib.stride_tricks.sliding_window_view
    uw = sw(u, (5, 5), axis=(-2, -1))
    dw = sw(d, (5, 5), axis=(-2, -1))
    mu_u = np.einsum("ij,...ij->...", k, uw)
    mu_d = np.einsum("ij,...ij->...", k, dw)
    su = np.einsum("ij,...ij->...", k, uw * uw) - mu_u ** 2
    sd = np.einsum("ij,...ij->...", k, dw * dw) - mu_d ** 2
    cov = np.einsum("ij,...ij->...", k, uw * dw) - mu_u * mu_d
    return mu_u, mu_d, np.maximum(su, 0), np.maximum(sd, 0), cov


def _upsample2(grid: np.ndarray) -> np.ndarray:
    """网格 (n,n[,C]) -> (2n,2n[,C])，半像素中心双线性。"""
    n = grid.shape[0]
    extra = ((0, 0),) * (grid.ndim - 2)
    gp = np.pad(grid, ((1, 1), (1, 1)) + extra, mode="edge")
    rows_e = 0.75 * gp[1:n + 1] + 0.25 * gp[0:n]        # 子像素 2c
    rows_o = 0.75 * gp[1:n + 1] + 0.25 * gp[2:n + 2]    # 子像素 2c+1
    inter = np.empty((2 * n, n + 2) + grid.shape[2:], np.float32)
    inter[0::2] = rows_e
    inter[1::2] = rows_o
    cols_e = 0.75 * inter[:, 1:n + 1] + 0.25 * inter[:, 0:n]
    cols_o = 0.75 * inter[:, 1:n + 1] + 0.25 * inter[:, 2:n + 2]
    out = np.empty((2 * n, 2 * n) + grid.shape[2:], np.float32)
    out[:, 0::2] = cols_e
    out[:, 1::2] = cols_o
    return out


def _structure_refine(grid: np.ndarray, ref_l: np.ndarray, cov_win: np.ndarray,
                      cand_rgb: np.ndarray, ref_rgb: np.ndarray,
                      active: np.ndarray, w_struct: float,
                      w_color: float, passes: int) -> np.ndarray:
    """逐格坐标下降最大化 SSIM 结构匹配 + 颜色保真。

    实现要点：一次处理一整行（对列做向量化批量求值），
    因此每格候选色都是"精确"评估，但整体速度比逐格 Python 循环快一到两个数量级。
    """
    n = grid.shape[0]
    if n < 3 or not active.any():
        return grid
    pad = 6
    u_l = np.asarray(_upsample2(np.asarray(C.luminance(grid), np.float32)), np.float32)
    up = np.pad(u_l, pad, mode="edge")
    dp = np.pad(np.asarray(ref_l, np.float32), pad, mode="edge")
    cp = np.pad(np.asarray(cov_win, np.float32), pad, mode="edge")
    gp = np.pad(grid, ((1, 1), (1, 1), (0, 0)), mode="edge")
    lab_ref = C.rgb_to_oklab(np.clip(ref_rgb, 0, 1))
    K = cand_rgb.shape[0]
    cand_lab = C.rgb_to_oklab(np.clip(cand_rgb, 0, 1))
    q = np.array([0.25, 0.75, 0.75, 0.25], np.float32)
    r = np.array([0.75, 0.25, 0.25, 0.75], np.float32)
    near2 = (np.arange(4) < 2)[None, :, None]
    q_a = q[None, None, :, None, None]
    q_b = q[None, None, None, :, None]
    r_a = r[None, None, :, None, None]
    r_b = r[None, None, None, :, None]
    idx = np.arange(n)
    sub_cols = (2 * idx + pad - 1)[:, None] + np.arange(4)[None, :]

    up_win = np.lib.stride_tricks.sliding_window_view(up, (12, 12), axis=(0, 1))
    dp_win = np.lib.stride_tricks.sliding_window_view(dp, (12, 12), axis=(0, 1))
    cp_win = np.lib.stride_tricks.sliding_window_view(cp, (8, 8), axis=(0, 1))

    for _pass in range(max(0, passes)):
        changed = 0
        for i in range(n):
            sel = np.nonzero(active[i])[0]
            if len(sel) == 0:
                continue
            r0 = 2 * i - 5 + pad
            u_row = up_win[r0, ::2][sel]
            d_row = dp_win[r0, ::2][sel]
            cp_row = cp_win[r0 + 2, ::2][sel]
            cv = cand_rgb[:, i, sel, :]                     # (K,m,3)
            nb_col = np.where(near2, gp[1 + i, sel][:, None, :], gp[1 + i, sel + 2][:, None, :])
            nb_row = np.where(near2, gp[i, sel + 1][:, None, :], gp[i + 2, sel + 1][:, None, :])
            diag = np.empty((len(sel), 4, 4, 3), np.float32)
            diag[:, 0:2, 0:2] = gp[i, sel][:, None, None, :]
            diag[:, 0:2, 2:4] = gp[i, sel + 2][:, None, None, :]
            diag[:, 2:4, 0:2] = gp[i + 2, sel][:, None, None, :]
            diag[:, 2:4, 2:4] = gp[i + 2, sel + 2][:, None, None, :]
            reg = (q_a * q_b * cv[:, :, None, None, :]
                   + q_a * r_b * nb_col[None, :, None, :, :]
                   + r_a * q_b * nb_row[None, :, :, None, :]
                   + r_a * r_b * diag[None, :, :, :, :])
            reg_l = np.asarray(C.luminance(reg), np.float32)     # (K,m,4,4)
            u_try = np.repeat(u_row[None], K, axis=0)
            u_try[:, :, 4:8, 4:8] = reg_l
            mu_u, mu_d, su, sd, cov = _windowed_stats(u_try, d_row[None])
            ssim = ((2 * mu_u * mu_d + _C1) * (2 * cov + _C2)) /                    ((mu_u ** 2 + mu_d ** 2 + _C1) * (su + sd + _C2))
            cwp = cp_row[None]
            wsum = np.maximum(cwp.sum((-1, -2)), 1e-3)
            s_mean = (ssim * cwp).sum((-1, -2)) / wsum
            d_col = np.linalg.norm(cand_lab[:, i, sel, :] - lab_ref[i][sel][None, :, :], axis=-1)
            score = w_struct * s_mean - w_color * d_col
            best = np.argmax(score, 0)
            new_c = cv[best, np.arange(len(sel))]
            new_l = reg_l[best, np.arange(len(sel))]
            cur_c = gp[1 + i, 1 + sel]
            changed += int((~np.all(np.isclose(new_c, cur_c, atol=1e-6), axis=-1)).sum())
            gp[1 + i, 1 + sel] = new_c
            sub = sub_cols[sel]
            up[r0 + 4:r0 + 8, sub] = new_l.transpose(1, 0, 2)
        if changed == 0:
            break
    return gp[1:n + 1, 1:n + 1]


# --------------------------------------------------------------------------
# 调色板（OKLab k-means + Potts ICM）
# --------------------------------------------------------------------------

_BAYER4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32)


def _bayer(n: int) -> np.ndarray:
    rep = int(np.ceil(n / 4))
    b = np.tile(_BAYER4, (rep, rep))[:n, :n]
    return (b / 16.0 - 0.5).astype(np.float32)


def _floyd_steinberg(lab, centers, mask, amount: float):
    """OKLab 上的 Floyd–Steinberg 误差扩散（逐格扫描，规模很小）。"""
    n = lab.shape[0]
    K = len(centers)
    work = lab.astype(np.float32).copy()
    idx = np.zeros((n, n), np.int32)
    for y in range(n):
        for x in range(n):
            if not mask[y, x]:
                continue
            p = work[y, x]
            d = centers - p
            k = int(np.argmin((d * d).sum(-1)))
            idx[y, x] = k
            err = (p - centers[k]) * amount
            if x + 1 < n:
                work[y, x + 1] += err * (7 / 16)
            if y + 1 < n:
                if x > 0:
                    work[y + 1, x - 1] += err * (3 / 16)
                work[y + 1, x] += err * (5 / 16)
                if x + 1 < n:
                    work[y + 1, x + 1] += err * (1 / 16)
    return idx


def _load_palette_file(path: str):
    """从 PNG（调色板条/任意小图）或文本（每行 #RRGGBB）读取固定调色板。"""
    if not path:
        return None
    import os
    if not os.path.exists(path):
        return None
    if path.lower().endswith((".txt", ".hex", ".pal")):
        cols = []
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip().lstrip("#")
                if len(line) >= 6 and all(c in "0123456789abcdefABCDEF" for c in line[:6]):
                    cols.append([int(line[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])
        if not cols:
            return None
        return C.rgb_to_oklab(np.asarray(cols, np.float32))
    from PIL import Image
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGBA")).astype(np.float32) / 255.0
    flat = arr.reshape(-1, 4)
    flat = flat[flat[..., 3] > 0.5][:, :3]
    if len(flat) == 0:
        return None
    uniq = np.unique((flat * 255).astype(np.uint8), axis=0).astype(np.float32) / 255.0
    return C.rgb_to_oklab(uniq[:256])


def _palettize(grid: np.ndarray, mask: np.ndarray, weight: np.ndarray, imp: np.ndarray,
               opt: PixelOptions):
    if opt.palette <= 0:
        return grid, 0, None
    lab = C.rgb_to_oklab(grid)
    pts = lab[mask]
    if len(pts) < 4:
        return grid, 0, None
    uniq = np.unique(np.round(pts, 3), axis=0)
    if len(uniq) <= opt.palette:
        return grid, len(uniq), None
    w = np.clip(weight[mask], 1e-3, None)
    fixed = _load_palette_file(getattr(opt, "palette_file", "") or "")
    if fixed is not None and len(fixed) >= 2:
        centers = np.asarray(fixed, np.float32)
        K = len(centers)
        n = grid.shape[0]
        lam_dir = None
        idx = np.argmin(np.linalg.norm(lab[..., None, :] - centers[None, None, :, :], axis=-1), -1)
        if opt.dither == "floyd":
            idx = _floyd_steinberg(lab, centers, mask, float(np.clip(opt.dither_amount, 0, 1)))
        elif opt.dither == "ordered":
            step = float(np.median(np.linalg.norm(
                lab[mask][:, None, :] - centers[None, :, :], axis=-1).min(1))) * 2.0
            off = _bayer(n)[..., None] * step * float(np.clip(opt.dither_amount, 0, 1))
            idx = np.argmin(np.linalg.norm((lab + off)[..., None, :] - centers[None, None, :, :],
                                           axis=-1), -1)
        out = C.oklab_to_rgb(centers[idx])
        return np.where(mask[..., None], out, grid), len(centers), centers
    centers, lab_idx = C.kmeans(pts, int(opt.palette), w, iters=30)
    cnt = np.bincount(lab_idx, minlength=len(centers)).astype(np.float32)
    centers = C.merge_close_colors(centers, cnt, min_dist=0.014)
    K = len(centers)
    if K <= 1:
        return np.repeat(centers[0][None, None, :], grid.shape[0], 0).repeat(grid.shape[1], 1), 1, centers

    n = grid.shape[0]
    d_all = np.linalg.norm(lab[..., None, :] - centers[None, None, :, :], axis=-1)   # (n,n,K)
    imp_n = np.clip(imp, 0, 1)
    cost0 = d_all * (0.6 + 0.4 * imp_n[..., None])
    # Potts 强度自适应：以"最近调色板距离"的中位数为尺度，只用来打破近似平局，
    # 避免把渐变色阶压成色块。
    dmin = d_all.min(-1)
    scale = float(np.median(dmin[mask])) if mask.any() else 0.02
    lam0 = 0.3 * float(opt.palette_smooth) * max(scale, 1e-3)
    lam_dir = []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        nb = np.roll(np.roll(lab, -dy, 0), -dx, 1)
        d2 = ((lab - nb) ** 2).sum(-1)
        s = max(float(np.mean(d2)), 1e-4)
        base = lam0 * np.exp(-d2 / (2.0 * s))
        keep = 1.0 - 0.85 * imp_n
        lam_dir.append(((base * keep) * (1.0 + 2.0 * mask)).astype(np.float32))
    dtype = (getattr(opt, "dither", "none") or "none").lower()
    if dtype in ("floyd", "fs", "error-diffusion"):
        idx = _floyd_steinberg(lab, centers, mask, float(np.clip(opt.dither_amount, 0, 1)))
        idx = _despeckle(idx, mask, rounds=1)
        out = C.oklab_to_rgb(centers[idx])
        return np.where(mask[..., None], out, grid), K, centers
    if dtype in ("ordered", "bayer"):
        dmin_pal = np.linalg.norm(lab[mask][:, None, :] - centers[None, :, :], axis=-1).min(1)
        step = float(np.median(dmin_pal)) * 2.0
        off = _bayer(n)[..., None] * step * float(np.clip(opt.dither_amount, 0, 1))
        idx = np.argmin(np.linalg.norm((lab + off)[..., None, :] - centers[None, None, :, :],
                                       axis=-1), -1)
        idx = _despeckle(idx, mask, rounds=1)
        out = C.oklab_to_rgb(centers[idx])
        return np.where(mask[..., None], out, grid), K, centers
    idx = np.argmin(cost0, -1)
    for _ in range(10):
        changed = 0
        for par in (0, 1):
            yy, xx = np.mgrid[0:n, 0:n]
            sel = ((yy + xx) % 2) == par
            potts = np.zeros((n, n, K), np.float32)
            for (dy, dx), lam in zip(((0, 1), (1, 0), (1, 1), (1, -1)), lam_dir):
                nb_idx = np.roll(np.roll(idx, -dy, 0), -dx, 1)
                if dy > 0:
                    nb_idx[0, :] = -1
                elif dy < 0:
                    nb_idx[-1, :] = -1
                if dx > 0:
                    nb_idx[:, 0] = -1
                elif dx < 0:
                    nb_idx[:, -1] = -1
                valid = nb_idx >= 0
                if valid.any():
                    diff = (nb_idx[..., None] != np.arange(K)[None, None, :])
                    potts += np.where(valid[..., None] & diff, lam[..., None], 0.0)
            tot = cost0 + potts
            new_idx = np.argmin(tot, -1)
            upd = sel & (new_idx != idx)
            changed += int(upd.sum())
            idx = np.where(upd, new_idx, idx)
        if changed == 0:
            break
    idx = _despeckle(idx, mask)
    out = C.oklab_to_rgb(centers[idx])
    return np.where(mask[..., None], out, grid), K, centers


def _despeckle(idx: np.ndarray, mask: np.ndarray, rounds: int = 3) -> np.ndarray:
    """去掉孤立噪点：四邻全同色而自己不同 -> 改成邻色（细线/拐角/斜线不受影响）。"""
    idx = idx.copy()
    for _ in range(rounds):
        nbs = []
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = np.roll(np.roll(idx, -dy, 0), -dx, 1)
            if dy > 0:
                nb[0, :] = -2
            elif dy < 0:
                nb[-1, :] = -2
            if dx > 0:
                nb[:, 0] = -2
            elif dx < 0:
                nb[:, -1] = -2
            nbs.append(nb)
        all_same = nbs[0].copy()
        for nb in nbs[1:]:
            all_same = np.where(nbs[0] == nb, all_same, -2)
        hole = (all_same >= 0) & (all_same != idx) & mask
        if not hole.any():
            break
        idx = np.where(hole, all_same, idx)
    return idx


# --------------------------------------------------------------------------
# 收尾：对比度/饱和度、alpha、描边
# --------------------------------------------------------------------------

def _tone(grid: np.ndarray, mask: np.ndarray, contrast: float, saturation: float) -> np.ndarray:
    """以主体亮度中位数为支点做温和对比度/饱和度增强，保持整体明度不失真。"""
    if (contrast <= 0 and saturation <= 0) or not mask.any():
        return grid
    lab = C.rgb_to_oklab(grid)
    L = lab[..., 0]
    v = L[mask]
    med = float(np.median(v))
    lo, hi = float(np.percentile(v, 3)), float(np.percentile(v, 97))
    if hi - lo < 0.28:
        g = min(1.0 / max(hi - lo, 1e-3), 1.6)
        L = np.clip(med + (L - med) * (0.72 + 0.28 * g), 0, 1)
        med = float(np.median(L[mask]))
    c = 1.0 + float(np.clip(contrast, 0, 1)) * 0.32
    L = np.clip(med + (L - med) * c, 0, 1)
    lab[..., 0] = L
    if saturation > 0:
        ab = lab[..., 1:3]
        mean_ab = ab[mask].mean(0)
        lab[..., 1:3] = mean_ab + (ab - mean_ab) * (1.0 + float(np.clip(saturation, 0, 1)) * 0.30)
    return C.oklab_to_rgb(lab)


def _fill_canvas(grid, mask, iters: int = 512):
    """把主体以外的格子用最近的主体颜色延展填满，并强制不透明（方块材质用）。"""
    if mask.all():
        return grid, np.ones_like(mask)
    filled, _known = _fill_transparent(grid, mask.astype(np.float32), iters=iters)
    out = np.where(mask[..., None], grid, filled)
    return out, np.ones_like(mask)


def _trim_to_content(grid, mask, margin: int = 0):
    """裁掉四周透明边，让主体尽量占满画布（Minecraft 材质常用做法）。"""
    n = grid.shape[0]
    ys = np.where(mask.any(1))[0]
    xs = np.where(mask.any(0))[0]
    if len(ys) == 0 or len(xs) == 0:
        return grid, mask
    y0, y1 = int(ys[0]), int(ys[-1]) + 1
    x0, x1 = int(xs[0]), int(xs[-1]) + 1
    h, w = y1 - y0, x1 - x0
    if h >= n and w >= n:
        return grid, mask
    oy = int(np.clip((n - h) // 2, margin, max(0, n - h)))
    ox = int(np.clip((n - w) // 2, margin, max(0, n - w)))
    g2 = np.zeros_like(grid)
    m2 = np.zeros_like(mask)
    g2[oy:oy + h, ox:ox + w] = grid[y0:y1, x0:x1]
    m2[oy:oy + h, ox:ox + w] = mask[y0:y1, x0:x1]
    return g2, m2


def _outline(grid: np.ndarray, mask: np.ndarray, opt: PixelOptions):
    if opt.outline <= 0 or not mask.any():
        return grid, mask
    col = np.array(opt.outline_color, np.float32)
    ring_in = mask & ~C.erode(mask, 1)
    if opt.outline == 1:
        out = np.where(ring_in[..., None], col, grid)
        return out, mask
    dil = C.dilate(mask, 1)
    ring_out = dil & ~mask
    out = np.where(ring_out[..., None], col, grid)
    return out, dil


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------

def pixelize(rgb: np.ndarray, alpha: np.ndarray, opt: PixelOptions):
    cv = _build_canvas(rgb, alpha, opt)
    st = _cell_stats(cv)
    S, ow, oh = cv["S"], cv["ow"], cv["oh"]

    cov = st["A"]
    A_s = C.gaussian_blur(cov, 0.55)
    mask, rescued = _alpha_mask(A_s, st["peak_a"], 0.5)

    # 候选色：全部是"均值族"（普通均值 / 重要性加权均值 / 纯净主体色），
    # 结构优化只在这三者之间做取舍，因此不会凭空造出高对比噪点。
    cand_list = [st["c_mean"], st["c_imp"], st["ref"]]
    cand = np.stack([np.clip(c, 0, 1) for c in cand_list], 0)            # (K,oh,ow,3)
    cand = np.nan_to_num(cand, nan=0.0, posinf=1.0, neginf=0.0)

    # 初始网格：细节强度决定均值与加权均值的混合
    detail = float(np.clip(opt.detail, 0, 1))
    grid = (1 - detail) * st["c_mean"] + detail * st["ref"]
    grid = np.clip(grid, 0, 1)

    filled, known = _fill_transparent(grid, cov)
    grid = np.where(known[..., None], grid, filled)

    cw2 = C.resize_area(cv["chan"]["a"], 2 * oh, 2 * ow)
    ref_rgb2 = C.resize_area(_fill_transparent(cv["chan"]["rgb_a"] / np.maximum(
        cv["chan"]["a"], 1e-4)[..., None], cv["chan"]["a"])[0], 2 * oh, 2 * ow)
    ref_l2 = C.luminance(ref_rgb2)
    cov_win = C.gaussian_blur(cw2, 1.2)
    ref_rgb_cell = C.resize_area(ref_rgb2, oh, ow)

    if opt.structure_passes > 0 and detail > 0.02:
        boundary = mask & ~C.erode(mask, 1)
        active = mask & ((st["lum_range"] > 0.010) | boundary)
        active &= cv["fit"][0] > 0
        grid = _structure_refine(grid, ref_l2, cov_win, cand, ref_rgb_cell,
                                 active, w_struct=1.25 * detail, w_color=0.5,
                                 passes=int(opt.structure_passes))

    grid = np.where(mask[..., None], grid, st["ref"])
    grid = _tone(grid, mask, opt.contrast, opt.saturation)

    # 被救回的细线格直接用峰值细节色，保证细结构清晰不发灰
    grid = np.where(rescued[..., None], np.clip(st["c_peak"], 0, 1), grid)

    grid_q, used, pal = _palettize(grid, mask, np.clip(cov, 0.15, 1), st["imp"], opt)
    grid = np.where(mask[..., None], grid_q, grid)

    grid, mask_f = _outline(grid, mask, opt)
    fill_mode = str(getattr(opt, "fill", "none") or "none").lower()
    if fill_mode in ("cover", "stretch", "fill", "tile") and mask_f.any():
        grid, mask_f = _fill_canvas(grid, mask_f)
    elif getattr(opt, "trim", False) and mask_f.any():
        grid, mask_f = _trim_to_content(grid, mask_f, opt.margin)
    out_alpha = np.where(mask_f, 1.0, 0.0).astype(np.float32)
    info = {"palette_used": used, "mask": mask, "coverage": cov, "cells": int(mask.sum()),
            "fill": fill_mode,
            "palette": None if pal is None else np.asarray(C.oklab_to_rgb(np.asarray(pal, np.float32)),
                                                          np.float32).round(4).tolist(),
            "fit": cv["fit"], "crop": cv["crop"]}
    return np.clip(grid, 0, 1).astype(np.float32), out_alpha, info
