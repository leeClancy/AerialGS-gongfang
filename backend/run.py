from __future__ import annotations

import argparse
import os
import sys
import webbrowser
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
APP_DIR = BACKEND_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


from backend.app.config import (  # noqa: E402
    APP_HOST,
    APP_PORT,
    choose_listen_port,
    keep_usable_path,
    prefer_ascii_root,
)


def _launch_root() -> Path:
    env = os.environ.get("AERIALGS_ROOT")
    if env:
        candidate = keep_usable_path(Path(env))
        if (candidate / "runtime" / "python" / "python.exe").is_file():
            return candidate
        parent = keep_usable_path(candidate.parent)
        if candidate.name == "app" and (parent / "runtime" / "python" / "python.exe").is_file():
            return parent
        return candidate
    portable = keep_usable_path(APP_DIR.parent)
    if (portable / "runtime" / "python" / "python.exe").is_file():
        return portable
    return keep_usable_path(APP_DIR)


ROOT = prefer_ascii_root(_launch_root())
os.environ["AERIALGS_ROOT"] = str(ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="航拍高斯工坊")
    parser.add_argument("--host", default=os.environ.get("AERIALGS_HOST", APP_HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get("AERIALGS_PORT", APP_PORT)))
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--fake", action="store_true", help="使用假进程（仅测试）")
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("只允许绑定 loopback（127.0.0.1）")

    import uvicorn

    os.environ["AERIALGS_FAKE"] = "1" if args.fake else os.environ.get("AERIALGS_FAKE", "0")
    print(f"运行根目录: {ROOT}")
    print(f"工作缓存: {os.environ.get('AERIALGS_WORK', '自动选择英文路径')}")
    port = choose_listen_port(args.host, args.port)
    if port != args.port:
        print(
            f"端口 {args.port} 已被占用（本机常见是百度拼音 baidupinyin.exe 占用 8765），"
            f"改用 http://{args.host}:{port}/"
        )
    webbrowser_url = f"http://{args.host}:{port}/"
    if not args.no_browser:
        try:
            webbrowser.open(webbrowser_url)
        except Exception:
            pass
    uvicorn.run(
        "backend.app.main:create_app",
        host=args.host,
        port=port,
        factory=True,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
