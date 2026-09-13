"""主体选择：背景/前景双色彩模型 + 显著性先验 + 多轮马尔可夫随机场(ICM) + 导向滤波精修。

流程
----
1. 缩略尺度上建立"背景色彩模型"（四周边带多起点 k-means）与"前景色彩模型"
   （远离背景色且不贴边的像素聚类），得到双模型连续置信度。
2. 能量 = 双模型数据项 + 居中/显著性先验 + 对比度自适应平滑项，由粗到细 ICM 求解。
3. 用当前标注重新估计两个色彩模型（GrabCut 式迭代若干轮），把多色主体、渐变背景
   逐步抠干净。
4. 连通域打分挑选主体（面积/紧致度/居中/贴边程度）。
5. 投影阴影剔除（乘法模型 + "必须位于主体下方且平滑"的门控）。
6. 回到较高分辨率做三值图精修 + 导向滤波，得到贴合图像边缘的 alpha。
"""
from __future__ import annotations

import numpy as np

from . import core as C
from . import device

_NB = ((-1, -1, 0.7071), (-1, 0, 1.0), (-1, 1, 0.7071), (0, -1, 1.0),
       (0, 1, 1.0), (1, -1, 0.7071), (1, 0, 1.0), (1, 1, 0.7071))


def color_rarity(lab_s: np.ndarray, k: int = 12) -> np.ndarray:
    """颜色稀有度：先把颜色聚成 k 类，每类的"与全图其它颜色的加权距离和"，
    再映射回像素 —— 罕见且对比强的颜色（宝石、金属）会被显著提亮。"""
    h, w = lab_s.shape[:2]
    pts = lab_s.reshape(-1, 3)
    idx = _subsample_idx(len(pts), 6000)
    centers = _best_kmeans(device.as_device(pts[idx]), min(k, max(2, len(idx) // 8)),
                           restarts=1, iters=12, seed=5)
    if centers is None:
        return np.zeros((h, w), np.float32)
    Cc = np.asarray(device.as_numpy(centers), np.float32)
    d = np.linalg.norm(Cc[:, None, :] - Cc[None, :, :], axis=-1)      # K x K
    wgt = 1.0 / (1.0 + d)
    global_contrast = (d * wgt).sum(1)
    global_contrast = global_contrast / max(float(global_contrast.max()), 1e-6)
    _, lab_idx = C.nearest_center(lab_s, device.as_device(Cc))
    lab_idx = np.asarray(device.as_numpy(lab_idx))
    out = global_contrast[lab_idx]
    return np.asarray(device.as_numpy(C.gaussian_blur(out.astype(np.float32),
                                                      max(0.8, min(h, w) / 240))), np.float32)


def _subsample_idx(n: int, cap: int) -> np.ndarray:
    if n <= cap:
        return np.arange(n)
    return np.arange(0, n, int(np.ceil(n / cap)))


def texture_energy(lum: np.ndarray, radius: int = 2) -> np.ndarray:
    """局部纹理能量：细节多的地方（首饰、叶片）能量高，平滑背景低。"""
    g = np.asarray(device.as_numpy(C.gradient_magnitude(lum, 1.0)), np.float32)
    e = np.asarray(device.as_numpy(C.box_mean(g, radius)), np.float32)
    scale = float(np.percentile(e, 97)) or 1.0
    return np.clip(e / scale, 0.0, 1.0)


def _border_band(h: int, w: int, thick: int) -> np.ndarray:
    m = np.zeros((h, w), bool)
    t = max(1, min(thick, h // 3, w // 3))
    m[:t, :] = True
    m[-t:, :] = True
    m[:, :t] = True
    m[:, -t:] = True
    return m


def _subsample(pts: np.ndarray, cap: int = 4000):
    n = len(pts)
    if n <= cap:
        return pts
    return pts[::int(np.ceil(n / cap))]


def _best_kmeans(pts, k: int, weights=None, restarts: int = 2, iters: int = 18, seed: int = 7):
    """多起点 k-means，取加权惯性最小者（避免随机初值导致模型漂移）。"""
    if pts is None or len(pts) < 2:
        return None
    host = device.as_numpy(pts)
    hw = None if weights is None else device.as_numpy(weights)
    k = int(max(1, min(k, len(host))))
    best = None
    for r in range(max(1, restarts)):
        Cc, lab = C.kmeans(pts, k, weights, iters=iters, seed=seed + 101 * r)
        Ch = device.as_numpy(Cc).astype(np.float32)
        Lh = device.as_numpy(lab)
        d = np.linalg.norm(host - Ch[Lh], axis=-1)
        inertia = float(((d * d) * (hw if hw is not None else 1.0)).mean())
        if best is None or inertia < best[0]:
            best = (inertia, Ch)
    return best[1]


def _side_uniformity(lab_s: np.ndarray) -> float:
    """四条边带中位色的最大差异（大 = 背景不干净，不该放宽阈值）。"""
    h, w = lab_s.shape[:2]
    t = max(2, int(round(0.05 * min(h, w))))
    parts = [lab_s[:t].reshape(-1, 3), lab_s[-t:].reshape(-1, 3),
             lab_s[:, :t].reshape(-1, 3), lab_s[:, -t:].reshape(-1, 3)]
    meds = [np.median(p_, 0) for p_ in parts]
    dmax = 0.0
    for i in range(4):
        for j in range(i + 1, 4):
            dmax = max(dmax, float(np.linalg.norm(meds[i] - meds[j])))
    return dmax


def _fit_models(lab_s: np.ndarray, valid: np.ndarray, soft_bg, restarts: int = 2):
    """返回 (bg_centers, fg_centers, d_lo, d_hi)。"""
    h, w = valid.shape
    thick = max(2, int(round(0.04 * min(h, w))))
    band = _border_band(h, w, thick) & valid
    pts = lab_s[band]
    if len(pts) < 16:
        pts = lab_s[valid]
    if len(pts) < 4:
        return None, None, None, None
    bg_centers = _best_kmeans(device.as_device(_subsample(pts)), 5, restarts=restarts, seed=11)
    if bg_centers is None:
        return None, None, None, None
    _, d_band = C.nearest_center(lab_s[band], device.as_device(bg_centers))
    t = float(np.percentile(device.as_numpy(d_band), 90))
    d_lo = max(t * 1.1, 0.016)
    d_hi = max(d_lo * 1.7, 0.05)

    # 背景生长 + 用生长结果重估背景模型（渐变/纹理背景更准）
    _, d_all = C.nearest_center(lab_s, device.as_device(bg_centers))
    d_np = device.as_numpy(d_all)
    m = max(2, int(round(0.06 * min(h, w))))
    inner = np.zeros((h, w), bool)
    inner[m:h - m, m:w - m] = True
    if soft_bg is not None:
        fg_sel = valid & (soft_bg < 0.5)
    else:
        thr = max(d_hi * 1.25, float(np.percentile(d_np, 94)))
        fg_sel = valid & (d_np > thr) & inner
    if fg_sel.sum() < 16:
        fg_sel = valid & (d_np > float(np.percentile(d_np, 97))) & inner
    fg_centers = None
    if fg_sel.sum() >= 8:
        fg_centers = _best_kmeans(device.as_device(_subsample(lab_s[fg_sel])), 4, restarts=restarts, seed=29)
    return bg_centers, fg_centers, d_lo, d_hi


def _unary(lab_s: np.ndarray, bg_centers, fg_centers, d_lo: float, d_hi: float,
           center_bias: float, sal_w: float, sal, rad):
    _, d_bg = C.nearest_center(lab_s, device.as_device(bg_centers))
    d_bg = device.as_numpy(d_bg).astype(np.float32)
    s_abs = np.clip((d_bg - d_lo) / max(d_hi - d_lo, 1e-4), 0.0, 1.0)
    if fg_centers is not None and len(fg_centers):
        _, d_fg = C.nearest_center(lab_s, device.as_device(fg_centers))
        d_fg = device.as_numpy(d_fg).astype(np.float32)
        s_rel = d_bg / (d_bg + d_fg + 1e-6)
    else:
        s_rel = s_abs
    p_fg = np.clip(0.62 * s_abs + 0.38 * s_rel, 0.0, 1.0).astype(np.float32)
    cost_bg = p_fg
    cost_fg = (1.0 - p_fg) + center_bias * rad
    if sal is not None and sal_w > 0:
        s = np.clip(sal, 0, 1).astype(np.float32)
        cost_fg = cost_fg - sal_w * s
        cost_bg = cost_bg + sal_w * 0.35 * s
    return cost_bg.astype(np.float32), cost_fg.astype(np.float32), d_bg


def _icm(lab, cost_bg, cost_fg, labels, lam: float, iters: int = 12):
    h, w = labels.shape
    d2s = []
    for dy, dx, _wt in _NB:
        shifted = np.roll(np.roll(lab, -dy, 0), -dx, 1)
        d2s.append(((lab - shifted) ** 2).sum(-1))
    mean_d2 = max(float(np.mean(d2s)), 1e-4)
    nb_w = [(dy, dx, (lam * float(wt) * np.exp(-d2 / (2.0 * mean_d2))).astype(np.float32))
            for (dy, dx, wt), d2 in zip(_NB, d2s)]
    yy, xx = np.mgrid[0:h, 0:w]
    parity = ((yy + xx) % 2).astype(np.int8)
    lab_i = labels.astype(np.int8)
    for _ in range(iters):
        changed = 0
        for par in (0, 1):
            sel = parity == par
            acc_fg = np.zeros((h, w), np.float32)
            acc_bg = np.zeros((h, w), np.float32)
            for dy, dx, w_pq in nb_w:
                nb = np.roll(np.roll(lab_i, -dy, 0), -dx, 1)
                if dy > 0:
                    nb[0, :] = -1
                elif dy < 0:
                    nb[-1, :] = -1
                if dx > 0:
                    nb[:, 0] = -1
                elif dx < 0:
                    nb[:, -1] = -1
                acc_fg += np.where(nb == 0, w_pq, 0.0)
                acc_bg += np.where(nb == 1, w_pq, 0.0)
            new = ((cost_fg + acc_fg) < (cost_bg + acc_bg)).astype(np.int8)
            upd = sel & (new != lab_i)
            changed += int(upd.sum())
            lab_i = np.where(upd, new, lab_i)
        if changed == 0:
            break
    return lab_i


def _upsample_to(labels, shape):
    h, w = shape
    up = np.repeat(np.repeat(labels, 2, 0), 2, 1)
    if up.shape[0] < h or up.shape[1] < w:
        up = np.pad(up, ((0, max(0, h - up.shape[0])), (0, max(0, w - up.shape[1]))), mode="edge")
    return up[:h, :w]


def _mrf_labels(lab, cost_bg, cost_fg, lam: float):
    pyr = [(lab, cost_bg, cost_fg)]
    while min(pyr[-1][0].shape[:2]) > 28:
        l, cb, cf = pyr[-1]
        pyr.append((C.box_down2(l), C.box_down2(cb), C.box_down2(cf)))
    lab_c, cb_c, cf_c = pyr[-1]
    labels = (cf_c <= cb_c).astype(np.int8)
    for l, cb, cf in reversed(pyr):
        if l.shape[:2] != labels.shape[:2]:
            labels = _upsample_to(labels, l.shape[:2])
        labels = _icm(l, cb, cf, labels, lam, iters=12)
    return labels


def _shadow_mask(img, centers):
    """投影阴影检测：阴影 ≈ 背景色乘以一个 <1 的系数（线性光下近似成立）。"""
    lin = np.asarray(device.as_numpy(C.srgb_to_linear(img)), np.float32)
    bg = np.asarray(device.as_numpy(
        C.srgb_to_linear(C.oklab_to_rgb(np.asarray(device.as_numpy(centers), np.float32)))), np.float32)
    best_res = np.full(lin.shape[:2], 1e9, np.float32)
    best_k = np.zeros(lin.shape[:2], np.float32)
    for c in bg:
        denom = float((c * c).sum()) + 1e-6
        k = (lin * c).sum(-1) / denom
        resid = np.linalg.norm(lin - k[..., None] * c, axis=-1)
        upd = resid < best_res
        best_res = np.where(upd, resid, best_res)
        best_k = np.where(upd, k, best_k)
    scale = float(np.mean(np.linalg.norm(lin, axis=-1))) + 1e-6
    return (best_k > 0.14) & (best_k < 0.985) & (best_res / scale < 0.07)


def _strip_shadow(small: np.ndarray, centers, mask: np.ndarray):
    """去掉投影阴影：必须位于主体核心下方、平滑、且与背景色成乘法关系。"""
    if not mask.any():
        return mask, 0
    h, w = mask.shape
    r = max(1, int(round(0.012 * min(h, w))))
    core = C.erode(mask, r)
    if not core.any():
        core = mask
    cand = mask & ~C.dilate(core, 1) & _shadow_mask(small, centers)
    if not cand.any():
        return mask, 0
    ed = np.asarray(device.as_numpy(
        C.edge_density(np.asarray(device.as_numpy(C.luminance(small)), np.float32))), np.float32)
    rows = np.where(core.any(1))[0]
    core_mid = float(rows.mean()) if len(rows) else h / 2.0
    labels, sizes = C.components(cand)
    drop = np.zeros_like(cand)
    for cid, size in enumerate(sizes, start=1):
        if size < max(6, int(0.002 * h * w)):
            continue
        ys, xs = np.where(labels == cid)
        if float(ys.mean()) < core_mid:
            continue
        if float(ed[ys, xs].mean()) > 0.30:
            continue
        drop |= labels == cid
    if not drop.any():
        return mask, 0
    keep = mask & ~drop
    keep = C.fill_holes(C.remove_small(keep, max(8, int(0.0005 * h * w))))
    if keep.sum() < max(8, int(0.45 * mask.sum())):
        return mask, 0
    return keep, int(drop.sum())


def _border_touch(mask: np.ndarray) -> float:
    if not mask.any():
        return 1.0
    edge = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
    return float(edge.mean())


def _pick_subject(mask: np.ndarray, mode: str, center_bias: float, rel_area: float = 0.10):
    labels, sizes = C.components(mask)
    if not sizes:
        return mask
    stats = C.component_stats(mask, labels, sizes)
    biggest = max(sizes)
    if mode == "all":
        keep = np.zeros(len(sizes) + 1, bool)
        for i, s in enumerate(sizes):
            keep[i + 1] = s >= max(6, biggest * 0.02)
        return keep[labels]
    if mode == "largest" or len(stats) == 1:
        sel = [max(stats, key=lambda d: d["area"])]
    else:
        scored = []
        for d in stats:
            area_n = (d["area"] / biggest) ** 0.5
            edge_pen = 1.0 if d["touch"] < 0.02 else 1.0 - 0.6 * min(1.0, d["touch"] * 3)
            score = area_n * (0.55 + 0.45 * d["fill"]) * (0.6 + 0.8 * center_bias * d["center"]) * edge_pen
            scored.append((score, d))
        scored.sort(key=lambda t: -t[0])
        main = scored[0][1]
        sel = [main]
        my0, mx0, my1, mx1 = main["bbox"]
        pad = 0.35 * max(my1 - my0, mx1 - mx0) + 2        # 允许的"离主体多远"
        for _sc, d in scored[1:]:
            y0, x0, y1, x1 = d["bbox"]
            near = not (x1 < mx0 - pad or x0 > mx1 + pad or y1 < my0 - pad or y0 > my1 + pad)
            if (d["area"] >= biggest * max(rel_area, 0.12) and d["touch"] < 0.15
                    and d["center"] > 0.25 and near):
                sel.append(d)
    keep = np.zeros(len(sizes) + 1, bool)
    for d in sel:
        keep[d["id"]] = True
    return keep[labels]


def _upscale_alpha(a: np.ndarray, h: int, w: int) -> np.ndarray:
    from PIL import Image
    im = Image.fromarray((np.clip(a, 0, 1) * 255.0 + 0.5).astype(np.uint8), "L")
    im = im.resize((w, h), Image.BILINEAR)
    return np.asarray(im).astype(np.float32) / 255.0


def _refine_work_res(rgb, coarse, bg_centers, d_lo, d_hi, shadow=None):
    """三值图精修：窄带用色彩置信度 + 导向滤波贴合真实边缘。"""
    h, w = rgb.shape[:2]
    core = coarse > 0.72
    if not core.any():
        return (coarse > 0.5).astype(np.float32)
    r = max(1, int(round(0.012 * min(h, w))))
    near = C.dilate(core, r)
    core_e = C.erode(core, max(1, int(round(r * 0.75))))
    lab_host = np.asarray(device.as_numpy(C.rgb_to_oklab(rgb)), np.float32)
    band_px = _border_band(h, w, max(2, int(round(0.04 * min(h, w)))))
    d_lo_f, d_hi_f = d_lo, d_hi
    sel = band_px.reshape(-1)
    if sel.sum() > 64:
        _, d_b = C.nearest_center(device.as_device(lab_host.reshape(-1, 3)[sel]),
                                  device.as_device(bg_centers))
        d_lo_f = max(d_lo, float(np.percentile(device.as_numpy(d_b), 88)) * 1.05)
        d_hi_f = max(d_lo_f * 1.7, d_hi)
    sh_zone = None
    if shadow is not None:
        rows = np.where(core.any(1))[0]
        ymid = int(rows[len(rows) // 2]) if len(rows) else h // 2
        below = np.zeros((h, w), bool)
        below[max(0, ymid - h // 40):] = True
        sh_zone = shadow & below       # 只有位于主体下方、且符合乘法阴影模型的才算投影阴影
    alpha = np.where(core_e & ~(sh_zone if sh_zone is not None else False), 1.0, 0.0).astype(np.float32)
    ys, xs = np.nonzero(near & ~core_e)
    if len(ys):
        _, d_p = C.nearest_center(device.as_device(lab_host[ys, xs]), device.as_device(bg_centers))
        d_p = device.as_numpy(d_p)
        soft = np.clip((d_p - d_lo_f) / max(d_hi_f - d_lo_f, 1e-4), 0.0, 1.0)
        conf = np.clip(soft * 1.2 - 0.08, 0.0, 1.0)
        if r > 1:
            conf = np.clip(conf + 0.25 * (coarse[ys, xs] - 0.5) * 2, 0.0, 1.0)
        if sh_zone is not None:
            conf = np.where(sh_zone[ys, xs], 0.0, conf)
        alpha[ys, xs] = conf
    lum = np.asarray(device.as_numpy(C.luminance(rgb)), np.float32)
    rad = max(2, int(round(0.008 * min(h, w))))
    alpha = np.asarray(C.guided_filter(lum, alpha, radius=rad, eps=5e-3), np.float32)
    return np.clip(alpha, 0.0, 1.0)


def _solve_mask(lab_s, valid, band, rad, sal, sens: float, center_bias: float,
                sal_w: float, rounds: int = 3, extra: dict | None = None):
    """给定参数解一遍粗粒度主体掩膜，返回 (ratio, mask, bg_centers, d_lo, d_hi)。

    extra 可覆盖阈值/显著性/居中/邻域平滑等，用于多方案搜索。"""
    ex = extra or {}
    sens = float(ex.get("sens", sens))
    center_bias = float(ex.get("center_bias", center_bias))
    sal_w = float(ex.get("sal_w", sal_w))
    lam = float(ex.get("lam", 0.55 + 0.5 * sens))
    subj = str(ex.get("subject", "auto"))
    thr_mul = float(ex.get("thr", 1.0))
    edge_barrier = float(ex.get("edge_barrier", 0.0))

    bg0, fg0, d_lo0, d_hi0 = _fit_models(lab_s, valid, None, restarts=2)
    if bg0 is None:
        return None
    allow_boost = _side_uniformity(lab_s) < 0.12
    best = None
    for k_boost, boost in enumerate((1.0, 0.62, 0.4)):
        if k_boost > 0 and not allow_boost:
            break
        bg_c, fg_c = bg0, fg0
        d_lo_b = max(d_lo0 * boost * thr_mul * (1.0 - 0.35 * (sens - 0.5) * 2), 0.008)
        d_hi_b = max(d_hi0 * boost * thr_mul * (1.0 - 0.20 * (sens - 0.5) * 2), d_lo_b * 1.25)
        labels = None
        for rnd in range(max(1, rounds)):
            cost_bg, cost_fg, d_bg = _unary(lab_s, bg_c, fg_c, d_lo_b, d_hi_b,
                                            center_bias, sal_w, sal, rad)
            hard = band & (d_bg < d_lo_b * 0.9)
            cost_fg = np.where(hard, cost_fg + 6.0, cost_fg)
            labels = _mrf_labels(np.ascontiguousarray(lab_s), cost_bg, cost_fg, lam)
            if rnd + 1 < rounds:
                soft_bg = np.asarray(device.as_numpy(
                    C.gaussian_blur(labels.astype(np.float32), max(0.8, min(lab_s.shape[:2]) / 200))), np.float32)
                bg2, fg2, _dl, _dh = _fit_models(lab_s, valid, soft_bg, restarts=1)
                if bg2 is not None:
                    bg_c = np.concatenate([np.asarray(bg_c, np.float32),
                                           np.asarray(bg2, np.float32)], 0)[:10]
                if fg2 is not None:
                    fg_c = fg2
        mask = labels.astype(bool) if labels is not None else np.zeros(lab_s.shape[:2], bool)
        if edge_barrier > 0:
            mask = _edge_grow(mask, ex.get("tex"), edge_barrier, ex.get("grad"))
        mask = C.remove_small(mask, max(3, int(0.0004 * mask.size)))
        mask = _pick_subject(mask, subj, center_bias)
        mask = C.fill_holes(mask)
        ratio = float(mask.mean())
        if best is None or ratio > best[0]:
            best = (ratio, mask, bg_c, d_lo_b, d_hi_b)
        if ratio >= 0.03:
            if boost == 1.0 or _border_touch(mask) < 0.35:
                break
            if best is not None and best[1] is mask:
                best = (0.0, mask, bg_c, d_lo_b, d_hi_b)
    return best


def _edge_grow(mask, tex, strength: float, grad):
    """按纹理/边缘能量做一次"边界吸附"：把高细节的相邻区域并入主体。"""
    if tex is None or strength <= 0:
        return mask
    m = np.asarray(mask, bool)
    for _ in range(3):
        nb = C.dilate(m, 1)
        add = nb & ~m & (tex > (1.0 - strength) * 0.45)
        if grad is not None:
            add = add | (nb & ~m & (grad > (1.0 - strength) * 0.5))
        if not add.any():
            break
        m = m | add
    return m


def _analyze(lab_s, lum_s, sal_base, band):
    """一次算好所有方案共用的分析图，避免重复计算。"""
    h, w = lab_s.shape[:2]
    grad = np.asarray(device.as_numpy(C.gradient_magnitude(lum_s, 1.0)), np.float32)
    tex = texture_energy(lum_s, 2)
    try:
        rar = color_rarity(lab_s)
    except Exception:  # noqa: BLE001
        rar = np.zeros((h, w), np.float32)
    base = sal_base if sal_base is not None else np.zeros((h, w), np.float32)
    sal_rich = np.clip(0.42 * base + 0.34 * rar + 0.24 * tex, 0, 1).astype(np.float32)
    sal_rich = np.asarray(device.as_numpy(
        C.gaussian_blur(sal_rich, max(0.8, min(h, w) / 260))), np.float32)
    return {"grad": grad, "tex": tex, "rarity": rar, "sal": base, "sal_rich": sal_rich}


def _hypotheses(sens: float, center_bias: float, sal_w: float):
    """候选方案：普通 / 高显著 / 稀有颜色驱动 / 边缘纹理驱动 / 居中主体。"""
    out = [("默认参数", {})]
    out.append(("高显著", {"sal_w": min(0.9, sal_w * 2.2 + 0.25), "thr": 0.9, "sens": sens + 0.1}))
    out.append(("稀有颜色", {"sal_w": min(0.95, sal_w + 0.45), "thr": 1.15, "center_bias": center_bias + 0.1}))
    out.append(("纹理边缘", {"sal_w": min(0.9, sal_w + 0.35), "thr": 1.05, "subject": "largest",
                           "edge_barrier": 0.7}))
    out.append(("居中主体", {"center_bias": min(0.8, center_bias + 0.35), "thr": 0.95, "sal_w": sal_w + 0.15}))
    out.append(("严格背景", {"thr": 1.35, "sens": max(0.0, sens - 0.15)}))
    return out


def _score_mask(mask, ctx, d_bg, d_hi: float, sal_use, center_bias: float):
    """客观质量分：背景可解释性 / 主体与背景对比 / 边界贴边 / 显著性 /
    紧致度 / 面积合理性 / 纹理丰富度。分数越高越像"人眼认定的主体"。"""
    m = np.asarray(mask, bool)
    cov = float(m.mean())
    if not m.any():
        return -1.0, {}
    if cov > 0.985:
        return -1.0, {"cov": cov}
    h, w = m.shape
    bg = ~m
    grad = ctx["grad"]
    tex = ctx["tex"]
    sal = ctx["sal_rich"] if sal_use else ctx["sal"]
    border = m & ~C.erode(m, 1)
    d_lo_ref = max(d_hi, 1e-3)

    s_bg = 1.0 - min(1.0, float(np.percentile(d_bg[bg], 70)) / (d_lo_ref * 1.6)) if bg.any() else 0.0
    s_fg = min(1.0, float(np.median(d_bg[m])) / (d_lo_ref * 2.4))
    s_edge = min(1.0, float(grad[border].mean()) / max(float(np.percentile(grad, 95)), 1e-6))
    s_sal = float(np.clip(float(sal[m].mean()) - float(sal[bg].mean()) * 0.9, -1, 1)) * 0.5 + 0.5 \
        if bg.any() else 0.5
    s_tex = float(np.clip(float(tex[m].mean()) - float(tex[bg].mean()) * 0.9, -1, 1)) * 0.5 + 0.5 \
        if bg.any() else 0.5
    labels, sizes = C.components(m)
    biggest = max(sizes) if sizes else 0
    s_comp = (biggest / max(sum(sizes), 1)) * (biggest / max(1, int(m.sum())))
    d = np.sqrt(2.0) * np.sqrt(2.0) / 2.0
    cy, cx = np.nonzero(m)
    if len(cy):
        ny = (cy.mean() - (h - 1) / 2) / max(h / 2, 1)
        nx = (cx.mean() - (w - 1) / 2) / max(w / 2, 1)
        s_center = 1.0 - min(1.0, float(np.hypot(ny, nx)) / 1.4142)
    else:
        s_center = 0.0
    s_size = 1.0 - min(1.0, abs(np.log2(max(cov, 1e-4) / 0.22)) / 2.3)
    s_touch = 1.0 - min(1.0, _border_touch(m) * 1.6)

    score = (0.19 * s_bg + 0.19 * s_fg + 0.17 * s_edge + 0.13 * s_sal + 0.09 * s_tex
             + 0.09 * s_comp + 0.06 * s_size + 0.04 * s_center + 0.04 * s_touch)
    detail = {"bg": round(s_bg, 3), "fg": round(s_fg, 3), "edge": round(s_edge, 3),
              "sal": round(s_sal, 3), "tex": round(s_tex, 3), "comp": round(s_comp, 3),
              "size": round(s_size, 3), "center": round(s_center, 3), "touch": round(s_touch, 3),
              "cov": round(cov, 3)}
    return float(score), detail


def _search_best_mask(lab_s, valid, band, rad, sal, sens, center_bias, sal_w, rounds,
                      analysis_side: int = 320, early_score: float = 0.78):
    """多方案试算 + 客观打分选优（用算力换准确率）。

    流程：低分辨率上并行试算多套策略 -> 客观指标打分 -> 胜出方案回到工作分辨率重解一次。
    简单图片一旦得分足够高就提前结束，速度几乎不受影响。
    """
    h, w = lab_s.shape[:2]
    scale = min(1.0, float(analysis_side) / max(h, w))
    if scale < 0.999:
        ah, aw = max(24, int(round(h * scale))), max(24, int(round(w * scale)))
        lab_a = np.asarray(device.as_numpy(C.resize_area(lab_s, ah, aw)), np.float32)
        valid_a = np.asarray(device.as_numpy(
            C.resize_bilinear(valid.astype(np.float32), ah, aw)), np.float32) > 0.5
        sal_a = None if sal is None else np.asarray(device.as_numpy(C.resize_area(sal, ah, aw)), np.float32)
        yy, xx = np.mgrid[0:ah, 0:aw].astype(np.float32)
        rad_a = np.sqrt(((yy - (ah - 1) / 2) / max(ah / 2, 1)) ** 2 +
                        ((xx - (aw - 1) / 2) / max(aw / 2, 1)) ** 2) / 1.4142
        band_a = _border_band(ah, aw, max(2, int(round(0.04 * min(ah, aw)))))
    else:
        ah, aw = h, w
        lab_a, valid_a, sal_a, rad_a, band_a = lab_s, valid, sal, rad, band

    lum_a = np.asarray(device.as_numpy(C.luminance(lab_a)), np.float32)
    ctx = _analyze(lab_a, lum_a, sal_a, band_a)
    cands = []
    for name, params in _hypotheses(sens, center_bias, sal_w):
        params = dict(params)
        use_rich = float(params.get("sal_w", sal_w)) > sal_w + 1e-6
        params["grad"] = ctx["grad"]
        params["tex"] = ctx["tex"]
        try:
            b = _solve_mask(lab_a, valid_a, band_a, rad_a, ctx["sal_rich"] if use_rich else sal_a,
                            sens, center_bias, sal_w, rounds, extra=params)
        except Exception:  # noqa: BLE001
            b = None
        if b is None or b[0] < 0.006:
            continue
        _, mask_c, bg_c, _dlo, d_hi_c = b
        _, d_all = C.nearest_center(lab_a, device.as_device(bg_c))
        d_all = np.asarray(device.as_numpy(d_all), np.float32)
        sc, det = _score_mask(mask_c, ctx, d_all, d_hi_c, use_rich, center_bias)
        if sc < 0:
            continue
        sc += 0.02 if name == "默认参数" else 0.0
        cands.append((float(sc), name, b, det, params, use_rich))
        if sc >= early_score:
            break
    if not cands:
        return None, {"smart_choice": "无候选", "smart_candidates": []}
    cands.sort(key=lambda t: -t[0])
    score, name, best, det, params, use_rich = cands[0]

    if (ah, aw) != (h, w):
        # 低分辨率试算会带来碎片，用同一套参数在工作分辨率上重解一次
        lum_f = np.asarray(device.as_numpy(C.luminance(lab_s)), np.float32)
        ctx_f = _analyze(lab_s, lum_f, sal, band)
        pr = dict(params)
        pr["grad"] = ctx_f["grad"]
        pr["tex"] = ctx_f["tex"]
        try:
            again = _solve_mask(lab_s, valid, band, rad,
                                ctx_f["sal_rich"] if use_rich else sal,
                                sens, center_bias, sal_w, rounds, extra=pr)
            if again is not None and again[0] > 0.005:
                best = again
        except Exception:  # noqa: BLE001
            pass
    info = {
        "smart_choice": name,
        "smart_score": round(float(score), 3),
        "smart_candidates": [(n, round(float(sc), 3)) for sc, n, _b, _d, _p, _r in cands],
        "smart_detail": det,
    }
    return best, info


def segment(rgb: np.ndarray, alpha_in: np.ndarray, *, sensitivity: float = 0.5,
            center_bias: float = 0.25, max_side: int = 420, keep_shadow: bool = False,
            saliency: float = 0.5, subject: str = "auto", rounds: int = 3,
            smart: bool = True, analysis_side: int = 320):
    """返回 (alpha 0..1, 信息字典)。alpha 即"主体掩码"，背景为 0。"""
    h, w = rgb.shape[:2]
    info: dict = {"method": "none", "bg_ratio": 0.0}

    if alpha_in is not None and float(np.min(alpha_in)) < 0.999 and \
            float((np.asarray(alpha_in) < 0.5).mean()) > 0.02:
        a = np.asarray(device.as_numpy(C.gaussian_blur(np.asarray(alpha_in, np.float32),
                                                       max(0.6, min(h, w) / 900))), np.float32)
        info["method"] = "alpha"
        info["bg_ratio"] = float((a < 0.5).mean())
        return np.clip(a, 0, 1), info

    scale = min(1.0, max_side / max(h, w))
    sh, sw = max(12, int(round(h * scale))), max(12, int(round(w * scale)))
    small = np.asarray(device.as_numpy(C.resize_area(rgb, sh, sw)), np.float32)
    lab_s = np.asarray(device.as_numpy(C.rgb_to_oklab(small)), np.float32)
    valid = np.ones((sh, sw), bool)
    if alpha_in is not None and alpha_in.shape == (h, w):
        valid = np.asarray(device.as_numpy(C.resize_area(np.asarray(alpha_in, np.float32), sh, sw))) > 0.35

    lum_s = np.asarray(device.as_numpy(C.luminance(small)), np.float32)
    sal = None
    if saliency > 0:
        try:
            cs = np.asarray(device.as_numpy(C.center_surround(small)), np.float32)
            sr = np.asarray(device.as_numpy(C.spectral_saliency(lum_s)), np.float32)
            ed = np.asarray(device.as_numpy(C.edge_density(lum_s, 3)), np.float32)
            sal = np.clip(0.6 * cs + 0.4 * sr - 0.35 * ed, 0, 1).astype(np.float32)
            sal = np.asarray(device.as_numpy(C.gaussian_blur(sal, max(0.8, min(sh, sw) / 240))), np.float32)
        except Exception:  # noqa: BLE001
            sal = None

    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
    rad = np.sqrt(((yy - (sh - 1) / 2) / max(sh / 2, 1)) ** 2 +
                  ((xx - (sw - 1) / 2) / max(sw / 2, 1)) ** 2) / 1.4142
    sens = float(np.clip(sensitivity, 0.0, 1.0))
    sal_w = float(np.clip(saliency, 0, 1)) * 0.35
    band = _border_band(sh, sw, max(2, int(round(0.04 * min(sh, sw)))))
    lam = 0.55 + 0.5 * sens

    if smart:
        best, smart_info = _search_best_mask(lab_s, valid, band, rad, sal, sens,
                                             center_bias, sal_w, rounds, analysis_side)
        info.update(smart_info)
    else:
        best = _solve_mask(lab_s, valid, band, rad, sal, sens, center_bias, sal_w, rounds)
    if best is None or best[0] < 0.008:
        info["method"] = "fallback-full"
        return np.ones((h, w), np.float32), info
    if best is None or best[0] < 0.008:
        info["method"] = "fallback-full"
        return np.ones((h, w), np.float32), info
    _, mask_s, bg_centers, d_lo, d_hi = best
    info["fg_ratio_small"] = float(best[0])
    info["bg_colors"] = np.asarray(device.as_numpy(
        C.oklab_to_rgb(np.asarray(bg_centers, np.float32)))).round(3).tolist()

    shadow_px = 0
    if not keep_shadow:
        mask_s, shadow_px = _strip_shadow(small, bg_centers, mask_s)
    info["shadow_px"] = shadow_px

    wscale = min(1.0, 900.0 / max(h, w))
    if wscale < 0.999:
        wh, ww = max(64, int(round(h * wscale))), max(64, int(round(w * wscale)))
        work_rgb = np.asarray(device.as_numpy(C.resize_area(rgb, wh, ww)), np.float32)
    else:
        wh, ww, work_rgb = h, w, rgb
    coarse_w = np.asarray(device.as_numpy(C.resize_bilinear(mask_s.astype(np.float32), wh, ww)), np.float32)
    if keep_shadow:
        shadow_w = None
    else:
        sh_small, _sc = C.downsample_mask(np.ones((wh, ww), np.float32), 420)
        if _sc < 1.0:
            from PIL import Image
            tiny = np.asarray(device.as_numpy(C.resize_area(work_rgb, sh_small.shape[0],
                                                            sh_small.shape[1])), np.float32)
            s_tiny = _shadow_mask(tiny, bg_centers)
            shadow_w = np.asarray(Image.fromarray((s_tiny.astype(np.uint8) * 255), "L").resize(
                (ww, wh), Image.NEAREST)).astype(np.float32) > 0.5
        else:
            shadow_w = _shadow_mask(work_rgb, bg_centers)
    alpha_w = _refine_work_res(work_rgb, coarse_w, bg_centers, d_lo, d_hi, shadow_w)
    alpha_w = np.asarray(device.as_numpy(C.gaussian_blur(alpha_w, max(0.6, min(wh, ww) / 900))), np.float32)
    alpha_w = np.clip((alpha_w - 0.26) / 0.48, 0.0, 1.0)
    bw = alpha_w > 0.5
    if bw.any():
        bw = C.cleanup_mask(bw, max(2, int(0.00002 * wh * ww)), fill=True, max_side=256)
        alpha_w = np.where(bw, np.maximum(alpha_w, 0.85), np.minimum(alpha_w, 0.3))
    alpha_f = _upscale_alpha(alpha_w, h, w) if (wh, ww) != (h, w) else alpha_w

    info["method"] = "bg+fg 双模型 MRF"
    info["bg_ratio"] = float((alpha_f < 0.5).mean())
    info["saliency"] = bool(sal is not None)
    return np.asarray(alpha_f, np.float32), info
