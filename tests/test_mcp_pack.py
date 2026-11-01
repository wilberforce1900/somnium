"""MCP 打包含件测试。

stdio 交互层（Popen+管道）在本机 Python 3.9 × pytest 捕获组合下有深层死锁
（独立脚本三次验证通过、pytest 下必挂——服务器的 stdin read 永远收不到已
flush 的写入）。工程裁决：**逻辑层全部直接对 dispatch() 单测**（与 stdio 循环
共享同一分发函数，逻辑等价），stdio 层只保留一个带看门狗的握手冒烟——该路径
在生产（各 harness 拉起）场景已由 integrations/README 的验证步骤覆盖。
"""
import importlib.util
import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SERVER = REPO / "mcp" / "server.py"
CKPT = REPO / "mcp" / "somnium-canonical-dh64.pt"


@pytest.fixture(scope="module")
def srv():
    """以模块方式加载 mcp/server.py（import 一次，torch 只加载一次）。"""
    spec = importlib.util.spec_from_file_location("somnium_mcp_server", SERVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # 导入即定义 dispatch/load_model；模型惰性加载
    yield mod
    mod._MODEL = None


def rpc(srv, mid, method, params=None):
    out = srv.dispatch({"jsonrpc": "2.0", "id": mid, "method": method,
                        "params": params or {}})
    assert out is not None and out["id"] == mid
    return out


def call_tool(srv, name, args=None, mid=100):
    r = rpc(srv, mid, "tools/call", {"name": name, "arguments": args or {}})
    assert "result" in r, f"{name} errored: {r.get('error')}"
    assert r["result"].get("isError") is not True, r["result"]["content"][0]["text"]
    return json.loads(r["result"]["content"][0]["text"])


def test_canonical_checkpoint_exists():
    assert CKPT.exists(), "规范 checkpoint 缺失（应随仓库在档）"


def test_initialize_handshake(srv):
    r = rpc(srv, 1, "initialize", {"protocolVersion": "2025-06-18"})
    assert r["result"]["protocolVersion"] == "2025-06-18"
    assert r["result"]["capabilities"] == {"tools": {}}
    assert r["result"]["serverInfo"]["name"] == "somnium-mcp"
    assert srv.dispatch({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert srv.dispatch({"jsonrpc": "2.0", "id": 2, "method": "ping"})["result"] == {}


def test_tools_list_complete(srv):
    r = rpc(srv, 3, "tools/list")
    names = {t["name"] for t in r["result"]["tools"]}
    assert names == {"somnium_info", "somnium_observe", "somnium_imagine",
                     "somnium_episode", "somnium_dream"}


def test_unknown_tool_and_method(srv):
    r = rpc(srv, 4, "tools/call", {"name": "no_such_tool", "arguments": {}})
    assert r["error"]["code"] == -32602
    r = rpc(srv, 5, "no/such/method")
    assert r["error"]["code"] == -32601
    # 通知（无 id）不响应
    assert srv.dispatch({"jsonrpc": "2.0", "method": "notifications/cancelled"}) is None


def test_info_reports_model_card(srv):
    info = call_tool(srv, "somnium_info")
    assert info["d_obs"] == 8 and info["d_h"] == 64 and info["n_actions"] == 4
    assert isinstance(info["params"], int) and info["params"] > 0
    assert info["trained_cycles"] >= 1
    assert "recipe" in info and "provenance" in info


def test_episode_is_real_grid8(srv):
    ep = call_tool(srv, "somnium_episode", {"steps": 12, "seed": 42})
    assert len(ep["obs"]) >= 4 and all(len(o) == 8 for o in ep["obs"])
    assert all(0 <= a <= 3 for a in ep["actions"])
    ep2 = call_tool(srv, "somnium_episode", {"steps": 12, "seed": 42})
    assert ep["obs"] == ep2["obs"]  # 同种子同情节
    hard = call_tool(srv, "somnium_episode", {"steps": 12, "seed": 42, "slip": 0.45})
    assert hard["slip"] == 0.45


def test_observe_reports_metrics_vs_copy(srv):
    ep = call_tool(srv, "somnium_episode", {"steps": 12, "seed": 42})
    ob = call_tool(srv, "somnium_observe", {"seq": ep["obs"], "actions": ep["actions"]})
    assert ob["steps"] == len(ep["obs"]) - 1
    assert ob["somnium_mse"] >= 0 and ob["copy_frame_mse"] >= 0
    assert ob["r2_vs_copy"] is None or isinstance(ob["r2_vs_copy"], float)
    assert ob["final_latent"]["h_norm"] > 0
    assert ob["final_latent"]["uncertainty_sigma"] > 0


def test_imagine_deterministic_per_seed(srv):
    ep = call_tool(srv, "somnium_episode", {"steps": 8, "seed": 7})
    a = call_tool(srv, "somnium_imagine", {"obs": ep["obs"][0], "steps": 4, "seed": 7})
    b = call_tool(srv, "somnium_imagine", {"obs": ep["obs"][0], "steps": 4, "seed": 7})
    c = call_tool(srv, "somnium_imagine", {"obs": ep["obs"][0], "steps": 4, "seed": 9})
    assert a["imagined_obs"] == b["imagined_obs"]
    assert a["imagined_obs"] != c["imagined_obs"]
    assert len(a["imagined_obs"]) == 4 and all(len(o) == 8 for o in a["imagined_obs"])
    assert len(a["uncertainty_sigma"]) == 4


def test_dream_readonly_does_not_persist(srv):
    ep = call_tool(srv, "somnium_episode", {"steps": 12, "seed": 42})
    before = call_tool(srv, "somnium_observe", {"seq": ep["obs"], "actions": ep["actions"]})
    dr = call_tool(srv, "somnium_dream", {"episodes": [ep["obs"]], "batches": 3,
                                          "persist": False})
    assert dr["model_persisted"] is False
    assert isinstance(dr["delta"], float)
    after = call_tool(srv, "somnium_observe", {"seq": ep["obs"], "actions": ep["actions"]})
    assert after["somnium_mse"] == before["somnium_mse"]  # 只读试梦：权重已回滚


def test_dream_persist_updates_weights(srv):
    ep = call_tool(srv, "somnium_episode", {"steps": 16, "seed": 123})
    before = call_tool(srv, "somnium_observe", {"seq": ep["obs"], "actions": ep["actions"]})
    dr = call_tool(srv, "somnium_dream", {"episodes": [ep["obs"]] * 2, "batches": 10,
                                          "persist": True})
    assert dr["model_persisted"] is True
    after = call_tool(srv, "somnium_observe", {"seq": ep["obs"], "actions": ep["actions"]})
    # 梦在所提交情节上应降低预测损失（真学习，非空转）
    assert after["somnium_mse"] < before["somnium_mse"]
    srv._MODEL = None  # 清状态，避免泄漏到后续测试


def test_bad_inputs_return_tool_errors(srv):
    r = rpc(srv, 900, "tools/call", {"name": "somnium_observe",
                                     "arguments": {"seq": [[1, 2, 3]]}})
    assert r["result"].get("isError") is True
    r = rpc(srv, 901, "tools/call", {"name": "somnium_imagine",
                                     "arguments": {"obs": [0.1] * 8, "steps": 99}})
    assert r["result"].get("isError") is True
    r = rpc(srv, 902, "tools/call", {"name": "somnium_dream",
                                     "arguments": {"episodes": []}})
    assert r["result"].get("isError") is True


def test_stdio_handshake_smoke():
    """stdio 层冒烟：子进程 + 读线程看门狗（此路径由各 harness 生产使用）。"""
    proc = subprocess.Popen([sys.executable, str(SERVER)], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            text=True, bufsize=1)
    lines = []

    def reader():
        lines.append(proc.stdout.readline())

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                 "params": {"protocolVersion": "2025-06-18"}}) + "\n")
    proc.stdin.flush()
    t.join(timeout=180)
    proc.terminate()
    assert lines and lines[0], "stdio 握手超时/无响应"
    r = json.loads(lines[0])
    assert r["result"]["serverInfo"]["name"] == "somnium-mcp"


def test_train_canonical_smoke(tmp_path):
    """tiny 配置端到端：训练 → 自描述 checkpoint → 可被服务器加载。"""
    ck = tmp_path / "smoke.pt"
    r = subprocess.run(
        [sys.executable, str(REPO / "experiments/mcp_pack/train_canonical.py"),
         "--cycles", "2", "--dream-pre", "3", "--d-h", "16", "--seed", "0",
         "--out", str(ck)],
        capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-500:]
    assert ck.exists()
    spec = importlib.util.spec_from_file_location("somnium_mcp_server_smoke", SERVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.CKPT = ck  # 指向 smoke 产物
    mod._MODEL = None
    info = call_tool(mod, "somnium_info", mid=700)
    assert info["d_h"] == 16
    ep = call_tool(mod, "somnium_episode", {"steps": 8, "seed": 1}, mid=701)
    ob = call_tool(mod, "somnium_observe", {"seq": ep["obs"], "actions": ep["actions"]},
                   mid=702)
    assert ob["somnium_mse"] >= 0
    mod._MODEL = None
