"""Basic URL safety checks.

Input: URL strings extracted from PDFs.
Output: safety label and warning list.
Side effects: none; this is rule-based and not a security scanner.
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class LinkSafety:
    label: str
    warnings: list[str]


def assess_link(uri: str) -> LinkSafety:
    parsed = urlparse(uri)
    warnings: list[str] = []

    if parsed.scheme == "mailto":
        return LinkSafety(label="邮件链接，需人工确认收件人", warnings=[])

    if parsed.scheme != "https":
        warnings.append("不是 HTTPS 链接")

    host = parsed.hostname or ""
    if not host:
        warnings.append("无法识别域名")
    elif _is_ip_or_local(host):
        warnings.append("链接指向 IP、localhost 或内网地址")
    elif host.startswith("xn--"):
        warnings.append("域名包含 punycode，可能伪装成相似域名")

    if parsed.username or parsed.password:
        warnings.append("URL 中包含用户名或密码")

    if warnings:
        return LinkSafety(label="需要谨慎打开", warnings=warnings)
    return LinkSafety(label="未发现明显风险", warnings=[])


def _is_ip_or_local(host: str) -> bool:
    normalized = host.strip("[]").lower()
    if normalized in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError:
        return False
    return address.is_private or address.is_loopback or address.is_link_local

