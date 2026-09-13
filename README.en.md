# PixelForge engine

[中文](README.md) | **English**

This is the algorithm half of [PixelForge](../README.md), pulled out into its own folder.
Dependencies: numpy and Pillow. Install CuPy and the heavy array work moves to the GPU by itself.
No network calls, no model weights — just computation, ready to drop into another project.

It answers two questions:

1. **Where is the subject?** Dual colour models, saliency priors and a multi-scale Markov random
   field, plus an automatic search over six strategies scored by objective measures when the image
   is difficult.
2. **How do I draw it as a decent N×N sprite?** Per-cell candidate colours, SSIM-based structural
   coordinate descent, and a palette quantised in a perceptual space.

---

## What's in here

| File | Contents |
| --- | --- |
| `device.py` | Compute device layer. Every operator is type-preserving: numpy in, numpy out; cupy in, cupy out — one code path for both |
| `core.py` | OKLab, area/bilinear resampling, Gaussian/box/guided filters, four saliency maps, scanline connected components, k-means |
| `segment.py` | Subject selection: dual colour models, multi-scale MRF (ICM), multi-hypothesis search and scoring, shadow removal, trimap refinement |
| `pixelize.py` | Pixelisation: canvas fitting, per-cell candidates, SSIM refinement, palette/dither/outline |
| `demo.py` | Smallest runnable example (CLI) |
| `selftest.py` | Cross-checks the operators against independent reference implementations |

## Install and run

```bash
pip install numpy Pillow        # that's all
pip install cupy-cuda12x        # optional, enables the GPU path

python -m algorithms.selftest   # should print all-pass

python -m algorithms.demo input.png out -s 16,32,64 --colors 16 --preview 8
python -m algorithms.demo bricks.jpg out -s 16 --fill cover      # block texture: fill, opaque
```

From Python:

```python
from algorithms import core, PixelArtOptions, to_pixel_art

rgb, alpha = core.load_image("input.png")          # sRGB 0..1, float32
opt = PixelArtOptions(size=32, palette=16, fill="none", smart=True)
small_rgb, small_a, info = to_pixel_art(rgb, alpha, opt)
core.save_rgba("out.png", small_rgb, small_a)
print(info["palette_used"], info.get("seg_smart_choice"), info.get("seg_smart_score"))
```

The two stages are independent, so you can use just one:

```python
from algorithms import core, segment, pixelize

mask, seg_info = segment.segment(rgb, alpha, smart=True)     # subject only
g, ga, info = pixelize.pixelize(rgb, mask, pixelize.PixelOptions(
    out_w=32, out_h=32, fill="cover", palette=16))
```

---

## Data flow

```
image ─► thumbnail (≤420px) ─┬─► background model: k-means over the border band
                             ├─► foreground model: pixels far from it and off the border
                             ├─► saliency: spectral residual + centre-surround
                             │             + colour rarity + texture energy
                             └─► spatial priors: centre bias, border contact
                                       │
                     data term + priors + contrast-adaptive smoothness → multi-scale MRF (ICM)
                                       │  ▲
                     re-estimate both colour models from the current labelling (a few rounds)
                                       │
                     ● automatic mode: six strategies at ≤320px → eight scores → best one wins
                                       │
                     shadow removal → component scoring → trimap + guided-filter refinement
                                       ▼
                                 alpha (0..1)
                                       │
     subject bounding box fitted into the grid (contain / cover / stretch)
                                       │
     per-cell statistics: coverage / α⁴ clean colour / importance-weighted mean / peak colour
                                       │
     SSIM refinement: exact per-cell coordinate descent (4×4 sub-pixels, 8×8 windows)
                                       │
     thin-feature rescue → palette k-means → Potts ICM → despeckle → dither / outline / tone
                                       ▼
                              N×N sprite (RGBA)
```

---

## Operators

### Device layer

There's no abstract backend interface. Instead every function keeps the array type it was given:
pass numpy, get numpy; pass cupy, get cupy. CuPy + CUDA is probed once at startup and used if
available, otherwise everything silently stays on the CPU with identical results. You can force it
with `PIXELFORGE_DEVICE=auto|cpu|gpu`. The point is that calling code never has to know a GPU exists.

### Colour space

sRGB to linear light, then OKLab:

```
l = 0.4122214708r + 0.5363325363g + 0.0514459929b      (same for m and s)
L = 0.2104542553·l^⅓ + 0.7936177850·m^⅓ − 0.0040720468·s^⅓
a = 1.9779984951·l^⅓ − 2.4285922050·m^⅓ + 0.4505937099·s^⅓
b = 0.0259040371·l^⅓ + 0.7827717662·m^⅓ − 0.8086757660·s^⅓
```

Every distance, cluster and palette decision happens here. Euclidean distance in RGB exaggerates
differences in green and compresses blue and red, which shows up as oddly biased palettes.

### Resampling

`resize_area` is an exact area average: a summed-area table built with cumsum, sampled with linear
interpolation at fractional boundaries, so a downscaled pixel is precisely the mean of what it covers
(the self-test compares it against a brute-force implementation; error is around 1e-7). When
upscaling it degenerates to bilinear, which is usually fine; `resize_bilinear` gives proper
half-pixel-centred interpolation when you need it. `box_down2` does integer-factor box
downsampling for the pyramids.

### Filters

`convolve1d` is a separable 1-D convolution with reflection at the borders. `box_mean` uses a
summed-area table for O(1) cost per pixel, and most local statistics ride on it.

`guided_filter` follows He et al. (2010), guided by luminance:

```
a = cov(I,p) / (var(I) + ε)        b = mean(p) − a·mean(I)
q = mean(a)·I + mean(b)
```

Applied to the soft alpha from segmentation, it pulls the boundary onto the real contour instead of
leaving a fuzzy halo.

### Four saliency maps

| Operator | How | Sensitive to |
| --- | --- | --- |
| `spectral_saliency` | Spectral residual (Hou & Zhang 2007): FFT, subtract a smoothed log-magnitude, keep the original phase, inverse FFT | regions that are unusual for their surroundings |
| `center_surround` | Colour difference from a local mean at σ = 2/4/8 | locally contrasting objects |
| `color_rarity` | Cluster colours into K bins, score each bin by its weighted distance to the others, map back to pixels | rare hues: gemstones, metals, neon |
| `edge_density` / `texture_energy` | Local density / energy of gradient magnitude | detailed objects versus smooth backgrounds |

They're combined with fixed weights in `segment._analyze()` — easy to tune.

### Connected components: scanline runs, not pixel BFS

A per-pixel BFS in Python is hopeless here (a 420×420 mask is ~170k steps). Instead:

1. pad the mask with a zero column on each side, take `np.diff` once to get every run of set
   pixels with its start and end column — fully vectorised;
2. between adjacent rows, use `searchsorted` to find overlapping runs (8-connectivity allows
   diagonal contact, 4-connectivity requires shared edges);
3. union overlapping runs with a disjoint-set structure, path halving and smaller-index-wins.

Cost is O(runs · α), an order of magnitude faster than the pixel loop in practice. `selftest.py`
compares it against a BFS reference for both connectivities and they agree entry for entry. Hole
filling then falls out for free: label the background, and anything not touching the frame is a hole.

### Subject selection

**Two colour models.** The background model comes from the border band (4% thick by default),
multi-start k-means in OKLab (k=5, keep the solution with the lowest weighted inertia); thresholds
`d_lo/d_hi` come from the 90th percentile of border-pixel distance to the nearest centre. The
foreground model clusters pixels that sit far from those colours and away from the border (k=4).

The data term mixes an absolute and a relative measure:

```
s_abs = clip((d_bg − d_lo) / (d_hi − d_lo), 0, 1)          how far from the background colours
s_rel = d_bg / (d_bg + d_fg + ε)                           more like background or foreground?
p_fg  = 0.62·s_abs + 0.38·s_rel

cost_bg = p_fg + 0.35·λ_sal·sal
cost_fg = (1 − p_fg) + center_bias·rad − λ_sal·sal
```

The absolute term alone misclassifies the middle of a gradient background; the relative term alone
collapses once either model is contaminated. Together they're stable.

**Energy and solver.** An 8-neighbourhood MRF:

```
E = Σ_p cost(L_p) + λ·Σ_(p,q) w_t·exp(−‖I_p − I_q‖² / (2·mean‖ΔI‖²))·[L_p ≠ L_q]
```

Smoothness decays exponentially with colour difference, so only visually similar neighbours demand
the same label and the term switches itself off across real edges. It's solved coarse-to-fine: at
each pyramid level, ICM sweeps in checkerboard order (same-parity pixels aren't neighbours, so the
update vectorises), and the coarse labelling initialises the next level. Far more stable than
single-scale ICM, and the boundary hugs edges better.

**Multiple attempts for hard images.** One parameter set will always fail on something like a white
display stand in front of greenery: the border band contains both white and dark green, and the
background model becomes two unrelated things. So automatic mode tries six strategies at ≤320px:

| Strategy | Change | Aimed at |
| --- | --- | --- |
| default | — | the general case |
| high saliency | λ_sal×2.2, threshold×0.9 | whatever visually stands out |
| rare colours | λ_sal+0.45, threshold×1.15 | distinctive hues (gems, metal) |
| texture edges | detail energy + boundary growth | detailed objects |
| centred subject | center_bias+0.35 | objects placed in the middle |
| strict background | threshold×1.35 | clean background, faint subject |

Each result is scored on eight measures (weights 0.04–0.19):

```
s_bg     70th-percentile distance of background pixels to the model, inverted
s_fg     median distance of the subject to the background model
s_edge   mean gradient on the mask boundary / 95th percentile over the image
s_sal    saliency inside minus saliency outside
s_tex    texture energy inside minus outside
s_comp   largest-component share × area share  (penalises fragmentation)
s_size   Gaussian penalty on log2(coverage / 0.22)
s_center distance of the centroid from the image centre
s_touch  fraction of the mask touching the frame
```

The winner is re-solved at full working resolution (low-resolution trials leave fragments behind),
then continues into shadow removal and refinement. If a candidate scores ≥ 0.78 the search stops
early, so easy images barely pay for it.

**Shadows.** In linear light a cast shadow is approximately the background colour times a factor.
For each background centre, a least-squares projection gives k and a residual; `0.14 < k < 0.985`
with a small residual marks a shadow candidate. Two gates follow: it must sit below the subject's core
and the region must be internally smooth — without them the algorithm happily deletes the dark side
of the subject itself.

**Refinement.** The coarse mask is split into definite foreground, a thin unknown band and definite
background. Pixels in the band are decided by colour confidence, and a guided filter (above) pulls
the edge onto the real contour. The result is a soft alpha that the pixelisation stage turns into
coverage statistics.

### Pixelisation

**Fitting.** Aspect-preserving scaling centred in the frame (`fill="none"`), or uniform upscaling
until it covers the canvas with the overflow cropped from the centre (`fill="cover"`, for block
textures), or independent stretching (`stretch`). Sampling happens on a working canvas with
supersampling S=4, so each output cell owns a 4×4 block of sub-pixels.

**Per-cell candidates.** Every cell gets:

- coverage (mean sub-pixel alpha) and peak coverage;
- an α⁴-weighted "clean subject" colour — only near-opaque sub-pixels contribute, so anti-aliased
  edges can't wash the colour out;
- an importance-weighted mean, importance = 0.62·gradient + 0.55·local colour deviation (both normalised);
- the colour of the most detailed sub-pixel.

Thin features get their own pass. After snapping by coverage (≥50% becomes opaque), a line test looks
for cells whose two opposite neighbours are inside the subject while everything 1–2 cells to the
perpendicular side is background; those are pulled back in. That's what keeps a one-pixel blade or
an antenna alive — a plain average erases them.

**SSIM refinement.** This is the most expensive and most valuable part of the engine.

The N×N grid is bilinearly upscaled to 2N (half-pixel centres) giving U, and a 2N reference D comes
from area-downsampling the canvas. The goal is for U to look like D in every local window — in other
words, for the upscaled sprite to resemble the original image.

The problem is non-convex and neighbouring cells interact. But there's a usable property: changing
cell (i,j) only affects its 4×4 sub-pixels (that's the support of the bilinear kernel) and therefore
only 8×8 comparison windows. So the gain from substituting candidate colour c into one cell can be
computed **exactly**, with no approximation:

```
4×4 affected sub-pixels (separable bilinear, q=[0.25,0.75,0.75,0.25], r=[0.75,0.25,0.25,0.75]):
U'[a,b] = q_a·q_b·c + q_a·r_b·nb_col[b] + r_a·q_b·nb_row[a] + r_a·r_b·diag[a,b]

local SSIM contribution (only the affected 8×8 windows, coverage-weighted):
S(c) = Σ_w cov_w·SSIM_w(U', D) / Σ_w cov_w

score(c) = w_struct·S(c) − w_color·‖OKLab(c) − OKLab(c_ref)‖
```

Candidates are restricted to the "mean family" (plain mean, importance-weighted mean, clean subject
colour). The structural term can therefore recover highlights and internal edges without inventing
single-pixel noise — an earlier version allowed bright/dark extremes and produced lightning-shaped
artefacts across smooth spheres.

The implementation doesn't loop per cell (that took over thirty seconds at 128×128). It processes a
whole row at once: gather every cell's patch together, evaluate all candidates together, and only
visit cells that carry structure (in flat regions every candidate is identical, so there's nothing
to decide). That took 64×64 from 1.18 s to 0.46 s and 128×128 from 4.80 s to 1.77 s.

**Palette.** k-means++ seeding with multiple restarts in OKLab, then near-identical colours are
merged (otherwise a gradient develops two indistinguishable bands). A Potts-smoothed ICM removes
isolated noise, with a strength derived from the median nearest-palette distance rather than a fixed
constant — the fixed version flattened spherical shading into a few blobs, which is how that
particular lesson was learned.

**Dither and finishing.** Two dithers: ordered Bayer 4×4 (offset before nearest-colour lookup) and
Floyd–Steinberg error diffusion (diffused in OKLab). Dithering and smoothing are mutually exclusive,
so the Potts pass is skipped when dithering is on. Outlines are either inner (inside the original
silhouette) or outer (expanded by one pixel). Tone adjustment pivots on the subject's median
luminance, which keeps overall brightness intact — the earlier 2%–98% percentile stretch darkened
subjects on white backgrounds, another lesson.

---

## Speed

Single-threaded, 12-core machine:

| Case | Time |
| --- | --- |
| 800×800 image → 16/32/64 | about 2–3 s |
| 3200×3200 image → 16/32/64 | about 5–6 s |
| 8 images × 3 sizes, one process | 19 s |
| same, eight processes | 5.4 s |

Where the wins came from:

| Change | Effect |
| --- | --- |
| SSIM pass row-vectorised, active cells only | 64×64: 1.18 s → 0.46 s; 128×128: 4.80 s → 1.77 s |
| Scanline components + union-find instead of pixel BFS | segmentation 3.6 s → 1.3 s per image |
| Structural cleanup (holes, small components) at 256 px | same |
| Hypothesis search at ≤320 px with early exit | easy images pay almost nothing |

Multiprocessing lives in the application layer (split by file); the engine itself is pure functions,
so parallelising it further is straightforward.

## Options

| Option | Meaning | Default |
| --- | --- | --- |
| `size / out_w,out_h` | output size | 32 |
| `remove_bg / smart` | extract background / automatic subject search | both on |
| `sensitivity` | background removal strength | 0.5 |
| `saliency` | saliency prior weight | 0.5 |
| `subject` | auto / largest / center / all | auto |
| `palette` | palette size, 0 = unlimited | 24 |
| `detail` | detail retention | 0.75 |
| `structure_passes` | SSIM passes, 0 = off | 3 |
| `fill` | none / cover / stretch | none |
| `trim / outline / dither` | crop / outline / dither | off / 0 / none |

## Self-test

```bash
python -m algorithms.selftest

[1] OKLab round-trip max error 1.1e-05
[2] resize_area vs brute force, max error 1.4e-07
[3] components, 8- and 4-connectivity, matches BFS reference: yes
[4] hole filling: pass
[5] end to end 32x32: shape (32, 32, 3) ...
```

It doesn't import anything from the application layer — copy this folder anywhere and it still runs.

## References

1. Kopf & Lischinski, *Content-adaptive image downscaling*, SIGGRAPH 2013 — the structural
   downscaling idea this borrows from
2. Wang et al., *Image Quality Assessment: from Error Visibility to Structural Similarity*, IEEE TIP 2004
3. Rother et al., *GrabCut*, SIGGRAPH 2004 — iteratively re-estimating the colour models
4. He et al., *Guided Image Filtering*, ECCV 2010
5. Hou & Zhang, *Saliency Detection: A Spectral Residual Approach*, CVPR 2007
6. Cheng et al., *Global Contrast based Salient Region Detection*, CVPR 2011
7. Ottosson, *A perceptual color space for image processing* (OKLab), 2020
8. Besag, *On the statistical analysis of dirty pictures*, 1986 — ICM

## Author and licence

Author: **贝dor芬 (pythonl)**

- Bilibili <https://space.bilibili.com/44517287>
- GitHub <https://github.com/pythonL-oss>

MIT licensed, see `LICENSE`. Use it, fork it, ship it commercially — just keep the notice.
