# arXiv 提交包（2026-10-10）

## 上传文件
`paper/arxiv-submission.tar.gz`（38KB：main.tex + refs.bib + figs/×2）
arXiv 会用它的 TeX Live 自行编译；所有宏包均为标准包，无兼容风险。

## 表单字段（逐项复制）

**Title:**
```
Somnium: A Recurrent World Model that Learns to Dream Before It Wakes
```

**Authors:**
```
Xiaoming (Will) Wang
```
（arXiv 作者栏填 "Xiaoming Wang" 可，展示名以你档案为准；名字带括号别名没问题）

**Abstract（纯文本，勿含 LaTeX 命令）:**
```
We train a small non-Transformer world model, Somnium, under a developmental
curriculum: chaos -> dream -> wake. A single recurrent substrate carries two
dynamics -- fast streaming expansion (yang) and energy-relaxation contraction
(yin) -- softly mixed by a gate; prediction happens in latent space (JEPA-style
next-latent), and training begins with a dream phase of self-generated rollouts
before any real environment interaction. Dream-first pretraining reliably cuts
the real-interaction steps needed to reach a fixed accuracy by ~60%: replicated
across five independent runs at 16K parameters (ratios 0.39-0.42 against a
pre-registered threshold of 0.6). Three further lines of evidence -- a scale
ladder, a dose-scaling control, and 871-round autonomous growth runs -- converge
on the same conclusion: the advantage is capacity-bound. It disappears at
210K-3.2M parameters, scaling the dream budget with width does not restore it,
and a dream-grown model that widens from 64 to 486 dimensions stays healthy
while the step to 1024 destabilizes training. Alongside the positive result we
report a registry of pre-registered negative results: a 64-entry VQ situation
codebook hurts training, a learned gate yields no accuracy gain over fixed
dynamics (though its polarity is a load-bearing developmental marker we
characterize as a ratchet), and feeding dream statistics back into sampling
does not improve calibration. All hypotheses, thresholds, and outcomes --
including two overnight infrastructure failures -- are logged in a public run
registry.
```

**Comments:**
```
7 pages, 2 figures. Code, full run registry, and per-run records:
https://github.com/wilberforce1900/somnium
```

**Primary category:** `cs.LG`
**Cross-list:** `cs.AI`（建议）；可选 `cs.NE`
**ACM class / MSC:** 留空

**License 建议:** 默认 "arXiv.org perpetual, non-exclusive license" 即可
（最通行；代码侧另用 MIT，二者独立）

## 流程与时间

1. arxiv.org 注册：用户名/密码 + **邮箱用 xwang755@asu.edu**（机构邮箱加 `will.wang@asu.edu`
   类别的收信验证；部分类目对 .edu 自动通过首次提交审核）
2. Start New Submission → 上传 tar.gz → arXiv 自动编译（可先点 Preview PDF 核对）
3. 填元数据（上表）→ 选类目 → 提交
4. **截止时间**：工作日 14:00 ET 前提交，次日凌晨（ET）announce；超时顺延一天
5. moderation 审核 1-2 个工作日（正常学术内容无碍）

## 若遇到 Endorsement 要求（备用路径）

cs.LG 对全新账号可能要求 endorsement（有 .edu 邮箱通常减免）。如被要求：
1. arXiv 会给一个 endorsement 链接 → 发给任何近期在 cs.LG/cs.AI 发过文的作者
2. 或先挂 GitHub preprint 占位（git 时间戳已定优先权），再寻 endorsement
3. endorsement 一旦获得，后续提交无需再要

## 发布后清单（我做）

- [ ] 拿到 arXiv 编号（arXiv:2026.xxxxx）→ 回填 GitHub README 顶栏 + 提交
- [ ] 仓库转回 public（论文引用了仓库链接，需可访问）——**等你一声令**
- [ ] 夜二收割后刷新 fig_overnight（纳入下半场数据）→ arXiv v2 可选
