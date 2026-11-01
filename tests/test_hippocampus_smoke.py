"""ITER15 pilot 三臂管线冒烟（tiny 配置）：分流正确、产物齐全、评估可算。"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PILOT = REPO / "experiments" / "hippocampus" / "pilot_hippocampus.py"
OUT = REPO / "experiments" / "hippocampus"


def _run_variant(variant: str, tag_suffix: str):
    for arm in ("W", "WR", "WS"):
        r = subprocess.run(
            [sys.executable, str(PILOT), "--arm", arm, "--seed", "0",
             "--variant", variant, "--tag-suffix", tag_suffix, "--smoke"],
            capture_output=True, text=True, timeout=900)
        assert r.returncode == 0, f"{arm}/{variant} failed: {r.stderr[-500:]}"
        d = OUT / f"details_{arm.lower()}_s0_smoke{tag_suffix}.jsonl"
        assert d.exists(), f"{arm}/{variant} 缺 details"
        rows = [json.loads(line) for line in d.read_text().splitlines()]
        assert len(rows) == 4, "应有 4 个块"
        # rho 字段=该块实际 slip：a 为 0.1/0.45 交替；b 恒 0.1（镜像块 1/3）
        assert [row["rho"] for row in rows] == ([0.1, 0.45, 0.1, 0.45] if variant == "a"
                                               else [0.1, 0.1, 0.1, 0.1])
        for row in rows:
            assert "pred_mse" in row["eval"]["e999"]
            assert isinstance(row["eval"]["e999"]["decode_r2"], float)
        if arm == "WS":
            assert all(row["boundary"] and row["boundary"].startswith("dream") for row in rows)
        if arm == "WR":
            assert all(row["boundary"] and row["boundary"].startswith("replay") for row in rows)
        if arm == "W":
            assert all(row["boundary"] is None for row in rows)
        # variant b：镜像块标志与 rho 语义（b 固定 ρ=0.1，行内 rho 保留块号语义）
        if variant == "b":
            assert [row["mirror"] for row in rows] == [False, True, False, True]
        # 测试产物不进仓库（正式 run 的文件名无 _smoke 后缀）
        d.unlink(missing_ok=True)


def test_three_arms_smoke_pipeline():
    _run_variant("a", "")


def test_mirror_variant_smoke_pipeline():
    _run_variant("b", "_b")
