"""计算设备层：CPU(numpy) / GPU(CuPy)，自动探测 + 优雅降级。

设计
----
* 所有核心数组运算都写成"类型保持"的：numpy 进 -> numpy 出；cupy 进 -> cupy 出。
  因此同一条代码路径既能在 CPU 上跑，也能在装了 CuPy 的机器上自动搬到 GPU，
  没有 GPU 时行为与纯 numpy 完全一致（已验证结果逐位相同）。
* 通过环境变量 PIXELFORGE_DEVICE=cpu|gpu|auto 或 set_device() 强制指定。
"""
from __future__ import annotations

import os
import threading

import numpy as np

_LOCK = threading.Lock()
_STATE: dict = {"resolved": False, "name": "cpu", "xp": np, "reason": "未探测"}


def _probe() -> dict:
    want = os.environ.get("PIXELFORGE_DEVICE", "auto").strip().lower()
    if want in ("cpu", "numpy", "off"):
        return {"name": "cpu", "xp": np, "reason": "手动指定 CPU"}
    try:
        import cupy as cp  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return {"name": "cpu", "xp": np, "reason": f"未安装 CuPy（{type(exc).__name__}），使用 CPU"}
    try:
        ndev = int(cp.cuda.runtime.getDeviceCount())
        if ndev <= 0:
            return {"name": "cpu", "xp": np, "reason": "没有可用的 CUDA 设备，使用 CPU"}
        probe = cp.zeros((8, 8), dtype=cp.float32)
        float((probe * probe).sum())
        try:
            props = cp.cuda.runtime.getDeviceProperties(0)
            dev = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
        except Exception:  # noqa: BLE001
            dev = "CUDA 设备"
        if want in ("gpu", "cuda"):
            return {"name": "gpu", "xp": cp, "reason": f"GPU：{dev}（CuPy {cp.__version__}）"}
        return {"name": "gpu", "xp": cp, "reason": f"GPU：{dev}（CuPy {cp.__version__}）"}
    except Exception as exc:  # noqa: BLE001
        return {"name": "cpu", "xp": np, "reason": f"CuPy 不可用（{type(exc).__name__}），使用 CPU"}


def backend() -> dict:
    with _LOCK:
        if not _STATE["resolved"]:
            info = _probe()
            _STATE.update(info)
            _STATE["resolved"] = True
        return _STATE


def xp():
    """当前设备的数组模块（numpy 或 cupy）。"""
    return backend()["xp"]


def xp_for(*arrays):
    """按输入数组类型选择数组模块：cupy 数组 -> cupy，其它 -> 当前设备。"""
    for a in arrays:
        if a is None:
            continue
        mod = type(a).__module__.split(".")[0]
        if mod == "cupy":
            import cupy as cp  # type: ignore
            return cp
        if mod == "numpy":
            return np
    return xp()


def is_gpu() -> bool:
    return backend()["name"] == "gpu"


def name() -> str:
    return backend()["name"]


def reason() -> str:
    return str(backend()["reason"])


def as_device(a):
    """把 numpy 数组搬到当前设备（GPU 时返回 cupy 数组）。"""
    x = xp()
    if x is np:
        return a
    if type(a).__module__.split(".")[0] == "numpy":
        return x.asarray(a)
    return a


def as_numpy(a):
    """把任意设备数组取回主机 numpy。"""
    if a is None:
        return None
    if type(a).__module__.split(".")[0] == "cupy":
        return a.get()
    return np.asarray(a)


def describe() -> str:
    b = backend()
    return f"{'GPU 加速' if b['name'] == 'gpu' else 'CPU'} · {b['reason']}"
