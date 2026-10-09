# GCP 上云手册（阶段 1 · 实机版 2026-10-09 修订）

> 环境由外部智能体搭建完毕（交接报告已并入），本手册记录**实机事实**与点火流程。
> 红线：凭证不进 git；纯合成环境无个人数据；**跑完 Stop（项目终了才 Delete）**。

## 0. 实机事实（取代原"建机"章节）

| 项 | 值 |
|---|---|
| 项目 / VM | `<PROJECT-ID>` / `ai-somnium-trainer` |
| Zone | **us-west4-c**（us-west1 三区均不支持 g2+L4，勿回退）|
| 规格 | g2-standard-4（4vCPU/16GB）+ 1×L4 + Ubuntu 22.04.5 minimal + 50GB 盘 |
| 驱动 | Google cuda_installer.pyz 安装，nvidia-smi 验证通过 |
| Python 环境 | `~/venvs/torch`（torch 2.14.1+cu126，CUDA 可用，L4 实测矩阵乘法通过）|
| VM 用户 | `<USER>` |
| 额度 | $300 至 2026-12-15；已实测覆盖 GPU 训练；注意 us-west4 费率或略高于 us-central1 |
| 额外依赖 | 只需 `pip install pytest`（torch 已装，勿重装）|

## 0.5 本机 → VM 访问桥（关键，因本机 googleapis API 域被墙、gcloud 不可用）

- 本机已生成专用密钥 `~/.ssh/somnium_gcp`（ed25519）。
- 用户在浏览器 Console SSH 里执行一次：
  `echo '<公钥>' >> ~/.ssh/authorized_keys`
- 之后本机直连：`ssh -i ~/.ssh/somnium_gcp -o ConnectTimeout=8 <USER>@<外网IP>`
- VM 生命周期（启停/删）留用户控制台操作；科学侧全走上述 SSH。
- 注意：VM Stop 后临时外网 IP 释放，重启后 IP 会变——每次点火前重取 IP。

## 1. 点火序列（SSH 通后逐条执行）

```sh
# 本机：上行（排除 venv/git/断点档）
sh scripts/cloud_up.sh <USER>@<IP>
# VM：装 pytest + 环境验证
source ~/venvs/torch/bin/activate
cd ~/mission_yin_yang && pip install pytest && python -m pytest tests -q
# CUDA 首秀（设备接线首次 CUDA 实跑 + L4 耗时锚点）
python experiments/e0_dream_first/run_e0.py --device cuda --tag cloudsmoke
```

## 2. R1 正式矩阵（阶段 1 主实验，口径见 PHASE1.md）

```sh
for dh in 64 256 1024; do
  for sched in wake_only dream_first alternate; do
    for seed in 0 1; do
      python3 experiments/e0_dream_first/run_e0.py --schedule $sched --seed $seed \
        --full --d-h $dh --device cuda --save-every 5 --tag r1-dh$dh
    done
  done
done
```
3 日程 × 2 seed × 3 规模 = 18 run（含 alternate，与 PHASE1 §1 口径一致）。
L4 上单 run 预计 <1 分钟量级（d_h=1024 档约 2–3 分钟）。

## 3. 回传与合并

```sh
sh scripts/cloud_down.sh <USER>@<VM外网IP>   # 结果回本机
# registry_cloud.md 为云端新增行，核对后合并进本机 registry.md
```

## 4. 收尾：Stop 而非 Delete（策略修订 2026-10-09）

驱动+torch 环境装在持久盘上（重装成本 ~15 分钟），删机即丢。
采用：**跑完用户在控制台 Stop VM**（磁盘 ~$5/月，额度内可忽略）；
项目终了或换机时才 Delete（删前备份 results/ 与 checkpoints）。

## 故障速查（实机版）

- SSH 连不上：①VM 是否在运行（Stop 后 IP 会变，重启重取）；
  ②本机测试 `ssh -i ~/.ssh/somnium_gcp -o ConnectTimeout=8 ...`；
  ③公钥是否已追加到 VM 的 authorized_keys。
- cuda unavailable：`nvidia-smi` 查驱动（重装：`sudo python3 ~/cuda_installer.pyz install_driver`）。
- Zone 报错：us-west1 全区不支持 g2+L4（已实证），留 us-west4-c。
- Spot 抢占：当前 Standard 无此风险；若将来转 Spot，续跑加
  `--resume ckpt_e0_<...>.pt`（断点体系已备）。
- us-west4 费率或高于 us-central1——每轮实验后核一次 Billing 页实际数字。
