from __future__ import annotations

import argparse
import base64
import os
import sys
from PIL import Image

from .config import (
    interactive_config,
    load_config,
    resolve_server,
    config_path,
)
from .converter import (
    build_markdown_from_images,
    build_markdown_from_pdf_bytes,
)
from .pdf_utils import summarize_pdf


def _add_run_args(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("input", help="输入文件路径：PDF 或 图片（png/jpg）")
    subparser.add_argument("--dpi", type=int, default=220)
    subparser.add_argument(
        "--debug-pdf",
        action="store_true",
        help="返回叠加可视化框的调试 PDF（base64 打印）",
    )
    subparser.add_argument("--server", help="覆盖服务端 URL（优先级最高）")
    subparser.add_argument(
        "--configure", action="store_true", help="运行前进入交互式配置向导"
    )
    subparser.add_argument(
        "--pages",
        help="页码选择（1-based）：例如 1,2 或 1-3 或 3- 或 -3 或 1,2,5-6",
        default=None,
    )


def _add_summary_args(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("input", help="输入文件路径：仅支持 PDF")
    subparser.add_argument(
        "--pages",
        help="页码选择（1-based）：例如 1,2 或 1-3 或 3- 或 -3 或 1,2,5-6；缺省为全部页",
        default=None,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="mineru-lite",
        description="VLM http-client → Markdown（轻量 CLI）",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_p = subparsers.add_parser("run", help="运行转换")
    _add_run_args(run_p)

    sum_p = subparsers.add_parser("summary", help="仅基于 PDF 文字层的摘要（不调用后端）")
    _add_summary_args(sum_p)

    cfg_p = subparsers.add_parser("config", help="查看/设置/重置 配置")
    cfg_sub = cfg_p.add_subparsers(dest="cfg_cmd")
    cfg_sub.add_parser("show", help="显示当前配置")
    set_p = cfg_sub.add_parser("set", help="设置配置项")
    set_p.add_argument("--server", required=False, help="设置 server_url")
    cfg_sub.add_parser("reset", help="重置（删除配置文件）")
    cfg_sub.add_parser("wizard", help="交互式配置向导（同 --configure）")

    args, _ = parser.parse_known_args(argv)
    if args.command is None:
        # 兼容：无子命令时，等价于 `run`
        args = parser.parse_args(["run", *(argv or [])])

    # 配置子命令
    if args.command == "config":
        cmd = args.cfg_cmd
        if cmd == "show":
            print(f"配置路径：{config_path()}")
            print(load_config() or "{}")
            return
        elif cmd == "set":
            from .config import save_config

            cfg = load_config()
            if getattr(args, "server", None):
                cfg["server_url"] = args.server
                save_config(cfg)
                print("已写入 server_url。")
            else:
                print("未提供 --server。")
            return
        elif cmd == "reset":
            fp = config_path()
            if fp.exists():
                fp.unlink()
            print("已删除配置文件。")
            return
        elif cmd == "wizard" or cmd is None:
            interactive_config(load_config())
            return

    # summary 子命令
    if args.command == "summary":
        path = args.input
        summarize_pdf(path, getattr(args, "pages", None))
        return

    # run 子命令
    if getattr(args, "configure", False):
        interactive_config(load_config())

    server_url = resolve_server(getattr(args, "server", None))
    path = args.input

    if path.lower().endswith(".pdf"):
        with open(path, "rb") as f:
            pdf_bytes = f.read()
        out = build_markdown_from_pdf_bytes(
            pdf_bytes,
            dpi=args.dpi,
            make_debug_pdf=args.debug_pdf,
            server_url=server_url,
            pages_spec=args.pages,
        )
    else:
        img = Image.open(path)
        out = build_markdown_from_images(
            [img], make_debug_pdf=args.debug_pdf, server_url=server_url
        )

    sys.stdout.write(out["markdown"])  # 主输出：Markdown
    if args.debug_pdf and "debug_pdf_bytes" in out:
        b64 = base64.b64encode(out["debug_pdf_bytes"]).decode()
        sys.stderr.write("\n[debug_pdf_bytes_base64]" + b64 + "\n")