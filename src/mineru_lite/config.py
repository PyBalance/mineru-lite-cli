from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict

from platformdirs import PlatformDirs

# tomllib 兼容
try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore
import tomli_w

DEFAULT_SERVER_URL = "http://127.0.0.1:30000"
APP_NAME = "mineru-lite"


def config_dir() -> Path:
    d = PlatformDirs(appname=APP_NAME, appauthor=False)
    p = Path(d.user_config_path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def config_path() -> Path:
    return config_dir() / "config.toml"


def load_config() -> Dict[str, Any]:
    fp = config_path()
    if not fp.exists():
        return {}
    try:
        with fp.open("rb") as f:
            return tomllib.load(f)
    except Exception:
        return {}


def save_config(cfg: Dict[str, Any]) -> None:
    fp = config_path()
    with fp.open("wb") as f:
        tomli_w.dump(cfg, f)


def interactive_config(existing: Dict[str, Any]) -> None:
    print(f"[配置向导] 当前配置文件：{config_path()}")
    cur_server = existing.get("server_url", "") or ""
    print(f"当前 server_url：{cur_server or '(未设置，使用默认)'}")
    new_server = input("请输入新的 server_url（回车跳过保留现状）：").strip()

    if new_server:
        cfg = dict(existing)
        cfg["server_url"] = new_server
        save_config(cfg)
        print("已保存到配置文件。")
    else:
        print("未更改。")


def resolve_server(cli_server: str | None) -> str:
    env_server = os.getenv("MINERU_LITE_SERVER_URL")
    cfg_server = load_config().get("server_url")
    server_url = cli_server or env_server or cfg_server or DEFAULT_SERVER_URL

    # 自动补 http:// 前缀
    if server_url and not re.match(r"^https?://", server_url):
        server_url = f"http://{server_url}"

    return server_url