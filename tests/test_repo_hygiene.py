"""仓库卫生回归测试（zboy 审计的制度化，2026-10-13）：公开仓库不得包含
机器用户名 / VM 身份名 / 真实 VM IP / 私钥材料 / 私人邮箱。

背景：v1.2 公开后审计发现 6 处个人标识泄露（机器用户名 x4、VM 人类账号名 x2），
已全部清除并相对化/占位符化。真实值保留在本地记忆与 ~/.ssh/config——本测试
以"分段拼接 + 正则"描述模式（防自匹配），只扫文本类白名单扩展名（冻结的
历史发布产物 tarball/PDF 豁免——其泄露为已记录接受的低危历史项，下个论文
版本自然清除）。
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = "tests/test_repo_hygiene.py"

BANNED_LITERALS = [
    "wills" + "macbookpro",              # macOS 机器用户名
    "wills" + "fitzgerald",              # VM 人类账号名（占位符应为 <USER_HUMAN>）
    "BEGIN OPENSSH PRIVATE" + " KEY",
    "BEGIN RSA PRIVATE" + " KEY",
]
BANNED_PATTERNS = [
    (re.compile(r"34\.125\.\d{1,3}\.\d{1,3}"), "VM 历史外网 IP"),
    (re.compile(r"136\.85\.\d{1,3}\.\d{1,3}"), "VM 历史外网 IP"),
    (re.compile(r"[\w.+-]+@(gmail|outlook|qq|163|126|hotmail)\."), "私人邮箱"),
]
TEXT_EXT = {".py", ".md", ".sh", ".tex", ".bib", ".txt", ".toml", ".yml",
            ".yaml", ".cfg", ".ini", ".gitignore", ""}


def _tracked() -> list[str]:
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True,
                         cwd=ROOT, check=True)
    return [f for f in out.stdout.split("\n") if f]


class TestRepoHygiene:
    def test_no_personal_identifiers(self):
        hits = []
        for rel in _tracked():
            if rel == SELF:
                continue
            ext = Path(rel).suffix
            if ext not in TEXT_EXT:
                continue  # 二进制/冻结发布产物豁免（见 docstring）
            p = ROOT / rel
            if not p.is_file() or p.stat().st_size > 300_000:
                continue
            text = p.read_text(encoding="utf-8", errors="ignore")
            for lit in BANNED_LITERALS:
                if lit in text:
                    hits.append(f"{rel}: {lit[:10]}…")
            for pat, desc in BANNED_PATTERNS:
                if pat.search(text):
                    hits.append(f"{rel}: {desc}")
        assert not hits, f"公开仓库发现个人标识泄露：{hits}"

    def test_intended_email_allowed(self):
        """白名单反证：法定联系邮箱（论文）必须仍可出现且不被本测试误杀。"""
        tex = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
        assert "xwang755@asu.edu" in tex
