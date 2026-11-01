"""ITER15 pilot 三臂管线冒烟（tiny 配置）：分流正确、产物齐全、评估可算。"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PILOT = REPO / "experiments" / "hippocampus" / "pilot_hippocampus.py"
OUT = REPO / "experiments" / "hippocampus"


def test_three_arms_smoke_pipeline():
    for arm in ("W", "WR", "WS"):
        r = subprocess.run(
            [sys.executable, str(PILOT), "--arm", arm, "--seed", "0", "--smoke"],
            capture_output=True, text=True, timeout=900)
        assert r.returncode == 0, f"{arm} failed: {r.stderr[-500:]}"
        d = OUT / f"details_{arm.lower()}_s0_smoke.jsonl"
        assert d.exists(), f"{arm} 缺 details"
        rows = [json.loads(line) for line in d.read_text().splitlines()]
        assert len(rows) == 4, "应有 4 个块"
        assert [row["rho"] for row in rows] == [0.1, 0.45, 0.1, 0.45]
        for row in rows:
            assert "pred_mse" in row["eval"]["e999"]
            assert isinstance(row["eval"]["e999"]["decode_r2"], float)
        if arm == "WS":
            assert all(row["boundary"] and row["boundary"].startswith("dream") for row in rows)
        if arm == "WR":
            assert all(row["boundary"] and row["boundary"].startswith("replay") for row in rows)
        if arm == "W":
            assert all(row["boundary"] is None for row in rows)
    # 测试产物不进仓库（正式 run 的文件名无 _smoke 后缀）
    for arm in ("W", "WR", "WS"):
        (OUT / f"details_{arm.lower()}_s0_smoke.jsonl").unlink(missing_ok=True)
