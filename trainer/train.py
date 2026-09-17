#!/usr/bin/env python3
from __future__ import annotations

"""
AerialGS 训练适配层。

基于 Apache-2.0 的 gsplat 官方示例思路做最小适配：
- 使用 pycolmap.Reconstruction 读取 COLMAP 模型（避开旧示例在 Windows 上的二进制解析缺陷）
- 使用预编译 gsplat CUDA 核心
- 使用纯 PyTorch SSIM，无需本机编译 fused-ssim
- 不伪造训练：缺少 gsplat / CUDA 时直接失败
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


OOM_HINT = (
    "CUDA 显存不足（OOM）。请改用更大的缩放倍率（12GB 默认 factor=4；可试 factor=8）、"
    "更少步数或“快速”预设。已尝试写出检查点（若已有可保存状态）。"
    "本工具不支持 CPU / AMD / Intel 训练。"
)


def _die(message: str, code: int = 1) -> None:
    print(message, file=sys.stderr, flush=True)
    raise SystemExit(code)


def require_stack() -> None:
    try:
        import torch  # noqa: F401
    except ImportError:
        _die("未安装 PyTorch CUDA。便携包应包含 torch 2.4.1+cu124。拒绝伪造训练。")
    try:
        import gsplat  # noqa: F401
    except ImportError:
        _die("未安装官方预编译 gsplat 1.5.3+pt24cu124 Windows wheel。拒绝伪造训练。请使用 packaging/build_portable.ps1 组装运行时。")
    import torch

    if not torch.cuda.is_available():
        _die("torch.cuda.is_available() 为 False。需要兼容的 NVIDIA 驱动与 CUDA 运行库。")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AerialGS gsplat trainer")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--data_factor", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=15000)
    parser.add_argument("--strategy", choices=["default", "mcmc"], default="default")
    parser.add_argument("--save_steps", default="7000,15000")
    parser.add_argument("--save_ply", action="store_true")
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--ssim_lambda", type=float, default=0.2)
    parser.add_argument("--max_splats", type=int, default=0)
    parser.add_argument("--grow_grad2d", type=float, default=0.0002)
    parser.add_argument("--coarse_to_fine", action="store_true")
    parser.add_argument("--opacity_reset_every", type=int, default=3000)
    return parser.parse_args(argv)


def _init_from_points(points, colors, device, sh_degree: int, max_splats: int = 0):
    import torch

    from trainer.geometry import knn_sq_dists

    if points is None or len(points) < 16:
        means_np = np.random.randn(2048, 3).astype(np.float64) * 0.3
        rgb_np = np.random.rand(means_np.shape[0], 3)
    else:
        means_np = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        col_np = np.asarray(colors, dtype=np.float64).reshape(-1, 3) if colors is not None and len(colors) == len(means_np) else np.random.rand(means_np.shape[0], 3)
        limit = 200_000
        if max_splats and 0 < max_splats < limit:
            limit = int(max_splats)
        if means_np.shape[0] > limit:
            rng = np.random.default_rng(0)
            idx = rng.choice(means_np.shape[0], limit, replace=False)
            means_np = means_np[idx]
            col_np = col_np[idx]
        rgb_np = np.clip(col_np, 0.0, 1.0)
    dist2 = np.clip(knn_sq_dists(means_np, k=4), 1e-8, None)
    n = means_np.shape[0]
    means = torch.tensor(means_np, device=device, dtype=torch.float32)
    rgb = torch.tensor(rgb_np, device=device, dtype=torch.float32)
    scales = torch.log(torch.sqrt(torch.tensor(dist2, device=device, dtype=torch.float32))).unsqueeze(-1).repeat(1, 3)
    quats = torch.zeros((n, 4), device=device)
    quats[:, 0] = 1.0
    opacities = torch.logit(torch.full((n,), 0.1, device=device))
    sh0 = ((rgb - 0.5) / 0.28209479177387814).unsqueeze(1)
    rest_dim = (sh_degree + 1) ** 2 - 1
    shn = torch.zeros((n, rest_dim, 3), device=device)
    params = {
        "means": torch.nn.Parameter(means),
        "scales": torch.nn.Parameter(scales),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(opacities),
        "sh0": torch.nn.Parameter(sh0),
        "shN": torch.nn.Parameter(shn),
    }
    return torch.nn.ParameterDict(params)


def _save_ckpt(path: Path, splats, step: int, extra: dict) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": step, "splats": {k: v.detach().cpu() for k, v in splats.items()}, "extra": extra}, path)


def _export_ply(path: Path, splats) -> int:
    import torch
    from trainer.ply_io import write_gaussian_ply

    means = splats["means"].detach().cpu().numpy()
    scales = splats["scales"].detach().cpu().numpy()
    quats = splats["quats"].detach().cpu().numpy()
    opacities = splats["opacities"].detach().cpu().numpy()
    sh0 = splats["sh0"].detach().cpu().numpy()
    shn = splats["shN"].detach().cpu().numpy()
    write_gaussian_ply(path, means=means, scales=scales, rotations=quats, opacities=opacities, sh0=sh0, shn=shn)
    return int(means.shape[0])


def _make_strategy(name: str, splats, optimizers, scene_scale: float, grow_grad2d: float, opacity_reset_every: int):
    kind = (name or "default").strip().lower()
    if kind == "mcmc":
        try:
            from gsplat.strategy import MCMCStrategy
        except Exception as exc:
            _die(f"当前 gsplat 无法可靠支持 MCMCStrategy，已阻止假装执行：{exc}")
        try:
            strategy = MCMCStrategy(verbose=True)
        except TypeError:
            strategy = MCMCStrategy()
        strategy.check_sanity(splats, optimizers)
        state = strategy.initialize_state()
        return strategy, state, "mcmc"
    if kind != "default":
        _die(f"未知训练策略 {name}。仅支持 default；MCMC 需 gsplat 1.5.3 官方接口。")
    from gsplat.strategy import DefaultStrategy

    reset_every = int(opacity_reset_every) if opacity_reset_every and opacity_reset_every > 0 else 10**9
    strategy = DefaultStrategy(verbose=True, grow_grad2d=float(grow_grad2d), reset_every=reset_every)
    strategy.check_sanity(splats, optimizers)
    try:
        state = strategy.initialize_state(scene_scale=scene_scale)
    except TypeError as exc:
        _die(
            "gsplat DefaultStrategy.initialize_state 必须传入 scene_scale（需要官方 1.5.3 API）。"
            f" {exc}"
        )
    return strategy, state, "default"


def train(args: argparse.Namespace) -> int:
    require_stack()
    import torch
    from gsplat.rendering import rasterization

    from trainer.dataset import ColmapScene
    from trainer.ssim import ssim

    device = torch.device("cuda")
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = result_dir / "ckpts"
    ply_dir = result_dir / "ply"
    ckpt_dir.mkdir(exist_ok=True)
    ply_dir.mkdir(exist_ok=True)

    scene = ColmapScene(args.data_dir, factor=args.data_factor)
    scene_scale = float(getattr(scene, "scene_scale", 1.0) or 1.0)
    print(
        f"[train] loader={scene.loader} frames={len(scene.frames)} factor={args.data_factor} "
        f"scene_scale={scene_scale:.4f}",
        flush=True,
    )
    splats = _init_from_points(scene.points, scene.colors, device, args.sh_degree, args.max_splats)
    optimizers = {
        name: torch.optim.Adam(
            [param],
            lr=_lr_for(name) * (scene_scale if name == "means" else 1.0),
        )
        for name, param in splats.items()
    }

    strategy, strategy_state, strategy_kind = _make_strategy(
        args.strategy, splats, optimizers, scene_scale, args.grow_grad2d, args.opacity_reset_every
    )

    save_steps = sorted({int(x) for x in str(args.save_steps).split(",") if x.strip()} | {args.max_steps})
    peak_mem = 0.0
    t0 = time.perf_counter()
    last_step = 0
    try:
        for step in range(args.max_steps):
            last_step = step
            frame = scene.frames[step % len(scene.frames)]
            rgb = torch.tensor(scene.load_rgb(frame), device=device).unsqueeze(0)
            viewmats = torch.tensor(frame.viewmat, device=device).unsqueeze(0)
            Ks = torch.tensor(frame.K, device=device).unsqueeze(0)
            colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)
            active_sh = args.sh_degree
            if args.coarse_to_fine and args.sh_degree > 0:
                interval = max(1, args.max_steps // (args.sh_degree + 2))
                active_sh = min(args.sh_degree, step // interval)
            render_colors, render_alphas, info = rasterization(
                means=splats["means"],
                quats=splats["quats"],
                scales=torch.exp(splats["scales"]),
                opacities=torch.sigmoid(splats["opacities"]),
                colors=colors,
                viewmats=viewmats,
                Ks=Ks,
                width=frame.width,
                height=frame.height,
                packed=False,
                absgrad=bool(getattr(strategy, "absgrad", False)),
                sh_degree=active_sh,
            )
            pred = render_colors[:, : frame.height, : frame.width, :3]
            if pred.shape[1:3] != rgb.shape[1:3]:
                pred = torch.nn.functional.interpolate(
                    pred.permute(0, 3, 1, 2),
                    size=rgb.shape[1:3],
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1)
            l1 = torch.abs(pred - rgb).mean()
            ssim_val = ssim(pred, rgb)
            loss = (1 - args.ssim_lambda) * l1 + args.ssim_lambda * (1 - ssim_val)
            strategy.step_pre_backward(
                params=splats,
                optimizers=optimizers,
                state=strategy_state,
                step=step,
                info=info,
            )
            for opt in optimizers.values():
                opt.zero_grad(set_to_none=True)
            loss.backward()
            for opt in optimizers.values():
                opt.step()
            if strategy_kind == "mcmc":
                strategy.step_post_backward(
                    params=splats,
                    optimizers=optimizers,
                    state=strategy_state,
                    step=step,
                    info=info,
                    lr=float(optimizers["means"].param_groups[0]["lr"]),
                )
            else:
                strategy.step_post_backward(
                    params=splats,
                    optimizers=optimizers,
                    state=strategy_state,
                    step=step,
                    info=info,
                    packed=False,
                )
            if args.max_splats and step % 100 == 0:
                n = int(splats["means"].shape[0])
                if n > int(args.max_splats):
                    from gsplat.strategy.ops import remove
                    opa = torch.sigmoid(splats["opacities"]).reshape(-1)
                    keep = torch.topk(opa, int(args.max_splats), largest=True).indices
                    drop = torch.ones(n, dtype=torch.bool, device=opa.device)
                    drop[keep] = False
                    remove(params=splats, optimizers=optimizers, state=strategy_state, mask=drop)
            if torch.cuda.is_available():
                peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / (1024**2))
            if step % 50 == 0 or step + 1 == args.max_steps:
                print(
                    f"step={step} loss={float(loss):.5f} l1={float(l1):.5f} ssim={float(ssim_val):.4f} "
                    f"N={splats['means'].shape[0]} mem={peak_mem:.1f}MB",
                    flush=True,
                )
            if (step + 1) in save_steps:
                _save_ckpt(ckpt_dir / f"ckpt_{step+1}.pt", splats, step + 1, {"loss": float(loss)})
                if args.save_ply:
                    n = _export_ply(ply_dir / f"point_cloud_{step+1}.ply", splats)
                    print(f"[train] wrote ply N={n}", flush=True)
    except torch.cuda.OutOfMemoryError:
        print(OOM_HINT, flush=True)
        try:
            _save_ckpt(ckpt_dir / f"ckpt_oom_{last_step}.pt", splats, last_step, {"oom": True})
        except Exception:
            pass
        _write_stats(result_dir, args, scene, splats, t0, peak_mem, last_step, oom=True)
        return 2
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            print(OOM_HINT, flush=True)
            try:
                _save_ckpt(ckpt_dir / f"ckpt_oom_{last_step}.pt", splats, last_step, {"oom": True})
            except Exception:
                pass
            _write_stats(result_dir, args, scene, splats, t0, peak_mem, last_step, oom=True)
            return 2
        traceback.print_exc()
        return 1

    if args.save_ply:
        n = _export_ply(ply_dir / "point_cloud.ply", splats)
        print(f"[train] final ply N={n}", flush=True)
    _write_stats(result_dir, args, scene, splats, t0, peak_mem, args.max_steps, oom=False)
    return 0


def _lr_for(name: str) -> float:
    return {
        "means": 1.6e-4,
        "scales": 5e-3,
        "quats": 1e-3,
        "opacities": 5e-2,
        "sh0": 2.5e-3,
        "shN": 1.25e-4,
    }.get(name, 1e-3)


def _write_stats(result_dir: Path, args, scene, splats, t0, peak_mem, step, oom: bool) -> None:
    elapsed = time.perf_counter() - t0
    payload = {
        "data_dir": str(args.data_dir),
        "data_factor": args.data_factor,
        "max_steps": args.max_steps,
        "strategy": args.strategy,
        "frames": len(scene.frames),
        "loader": scene.loader,
        "scene_scale": float(getattr(scene, "scene_scale", 1.0) or 1.0),
        "num_gaussians": int(splats["means"].shape[0]) if splats is not None else 0,
        "peak_mem_mb": round(float(peak_mem), 2),
        "elapsed_sec": round(float(elapsed), 2),
        "step": int(step),
        "oom": oom,
    }
    (result_dir / "stats.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
