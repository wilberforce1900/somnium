# GCP 上云手册（阶段 1 · 标准档 $120 · 单卡 L4）

> 前提：用户已确认使用 **$300 免费试用额度**。红线：凭证不进 git；
> 纯合成环境无个人数据；**每次跑完即删机**；预算告警 50%/80%。

## 0. 用户侧一次性操作（约 10 分钟，控制台完成）

1. **升级付费账号**（GPU 不在试用范围）：Billing → 概览 →「升级」。
   剩余额度保留，升级本身不扣费——只有额度耗尽后才走绑定的卡。
2. **申请 GPU 配额**：IAM → 配额 → 搜索 `NVIDIA L4 GPUs`（区域选 us-central1）
   → 编辑上限为 1 → 提交理由（research/training，通常数分钟-1 天批）。
3. **预算告警**：Billing → 预算 → 新建 $120 阈值，设 50%（$60）与 80%（$96）告警。

## 1. 建机（gcloud 已登录 `gcloud auth login` 且 `gcloud config set project <PID>`）

```sh
gcloud compute instances create myy-l4 \
  --zone=us-central1-a --machine-type=g2-standard-4 \
  --accelerator=count=1,type=nvidia-l4 \
  --image-family=common-cu124 --image-project=deeplearning-platform-release \
  --boot-disk-size=50GB
# 深度学习镜像自带 NVIDIA 驱动 + CUDA + 常用 ML 栈，免装驱动。
# 备选：PyTorch 预装镜像（免 pip）——先查当前可用名：
#   gcloud compute images list --project=deeplearning-platform-release | grep pytorch
#   然后 --image-family=pytorch-2-x-cu124（以查到的为准）替换 common-cu124。
```

首次跑通用 STANDARD 按需；跑通后改 `--provisioning-model=SPOT
--instance-termination-action=STOP`（省 ~60-70%，配 --save-every 续跑）。

## 2. 上行 + 环境自检

```sh
gcloud compute ssh myy-l4 --zone=us-central1-a
# VM 内：
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 若无 torch：pip3 install -r ~/mission_yin_yang/requirements-cuda.txt
cd ~/mission_yin_yang && python3 -m pytest tests -q          # 80 测试应全绿
python3 experiments/e0_dream_first/run_e0.py --device cuda --tag cloudsmoke
```

本机侧（新开终端）：
```sh
sh scripts/cloud_up.sh <用户名>@<VM外网IP>
```

## 3. R1 正式矩阵（阶段 1 主实验，口径见 PHASE1.md）

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

## 4. 回传与合并

```sh
sh scripts/cloud_down.sh <用户名>@<VM外网IP>   # 结果 jsonl/json 回本机
# registry 行在云端追加——手工核对后合并进本机 registry.md
```

## 5. 删机（务必）

```sh
gcloud compute instances delete myy-l4 --zone=us-central1-a
```

## 故障速查

- `cuda unavailable`：`nvidia-smi` 看驱动；换镜像 family `common-cu121`。
- 配额拒绝：换区域（us-west1/us-east1）重申，或先用 T4（g2→n1 高机器）。
- Spot 被抢占：重跑同命令加 `--resume ckpt_e0_<...>.pt`（已存档断点续跑）。
