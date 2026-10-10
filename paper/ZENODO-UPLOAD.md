# Zenodo 上传包（2026-10-10）

> 目标：为论文取得正式可引用 DOI（形如 `10.5281/zenodo.XXXXXXX`）。
> 无任何背书门槛；注册 2 分钟（可直接用 GitHub 登录）+ 上传 5 分钟。

## 上传文件（两个）

1. `paper/somnium-paper.pdf`（主文件，126KB）
2. `paper/somnium-paper-source.tar.gz`（LaTeX 源码，38KB）

## 网页流程（zenodo.org）

1. **登录**：右上 Sign up / Log in → **用 GitHub 账号一键登录**（最省事）；
2. 右上角头像旁 **"New upload"**；
3. **Files**：把上面两个文件拖进去；
4. **Basic information** 按下表逐项填；
5. **License**：选 `Creative Commons Attribution 4.0 (CC BY 4.0)`；
6. **Access**：`Open Access`（默认）；
7. **Publish** → DOI 立即铸成（记录页右上角显示）。

## 表单字段（逐项复制）

**Title:**
```
Somnium: A Recurrent World Model that Learns to Dream Before It Wakes
```

**Creators（作者）:**
- Name: `Wang, Xiaoming (Will)`
- Affiliation: `Independent Researcher`

**Description（Markdown 可用）:**
```
Preprint v1.0. Paper PDF attached; LaTeX source in the source archive.
Code, full run registry, and per-run records:
https://github.com/wilberforce1900/somnium

We train a small non-Transformer world model, Somnium, under a developmental
curriculum: chaos -> dream -> wake. A single recurrent substrate carries two
dynamics -- fast streaming expansion (yang) and energy-relaxation contraction
(yin) -- softly mixed by a gate; prediction happens in latent space (JEPA-style
next-latent), and training begins with a dream phase of self-generated rollouts
before any real environment interaction. Dream-first pretraining reliably cuts
the real-interaction steps needed to reach a fixed accuracy by ~60%: replicated
across five independent runs at 16K parameters (against a pre-registered
threshold). Three further lines of evidence -- a scale ladder, a dose-scaling
control, and 871-round autonomous growth runs -- converge on the same
conclusion: the advantage is capacity-bound. Alongside the positive result we
report a registry of pre-registered negative results (a VQ situation codebook
that hurts training; a learned gate with no accuracy gain whose polarity is a
one-way developmental ratchet; dream-statistics feedback with zero calibration
gain). All hypotheses, thresholds, and outcomes are logged in a public run
registry.

Writing note: the scientific work (experiments, code, 90+ run records) was led
by the author; text drafting was AI-assisted; the author takes full
responsibility for all content.
```

**Keywords（分号分隔）:**
```
world models; pretraining; self-supervised learning; continual learning; pre-registration; negative results
```

**Language:** `English`
**Resource type:** `Publication` → `Preprint`
**Publication date:** `2026-10-10`
**Version:** `1.0`

**Related works（可选加分项）:**
- Relation: `is supplemented by`
- Identifier: `https://github.com/wilberforce1900/somnium`
- Resource type: `Software`

## 可选润色：DOI 印上论文首页（推荐，多花 5 分钟）

1. 在表单 **Digital Object Identifier** 处点 **"Reserve DOI"**（发布前预留）；
2. 把预留到的 DOI（`10.5281/zenodo.XXXXXXX`）发我；
3. 我 5 分钟内重新编译一版把 DOI 印在首页脚注的 PDF；
4. 你在 Zenodo 替换文件后 **Publish**——DOI 直接出现在论文里，成品感拉满。

不做也行：先发布，拿到 DOI 后我回填 README/GitHub Release，论文 PDF 留到下次修订再印。

## 发布后（我做）

- [ ] DOI 回填：README 顶部 + GitHub Release v0.1-paper 说明 + 项目记忆；
- [ ] 更新仓库内 README 的引用格式（"Cite as: Wang, X. (2026). Somnium... Zenodo. DOI"）；
- [ ] （可选）夜二收割时一并推送。
