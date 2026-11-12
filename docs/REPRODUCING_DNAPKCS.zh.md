# vcdd 虚拟细胞药物设计 — DNA-PKcs 抑制剂筛选管线

基于 [vcdd](https://github.com/ljvmiranda921/vcdd)（Virtual-Cell Drug Design）搭建的 DNA-PKcs
抑制剂全链筛选管线，stage1–4 已在本机 GPU 上真实完整跑通（2026-08）。

**目标**：从 20 万生成式小分子库中筛选「加入化合物后抑制 DNA 损伤修复（DDR 抑制）」的候选。

```
20万小分子 → Vina-GPU 对接初筛 → Boltz-2 结构精筛 → XPert 转录组打分 → 生成式优化改造
```

> vcdd 框架的上游原始说明见 `README-vcdd-upstream.md`。

---

## 1. 环境与依赖

```bash
conda activate vcdd          # /home/geng/miniconda3/envs/vcdd

# 执行任何命令前：
export PYTHONPATH=/home/geng/vcdd-0.3.0:tmp_pydeps          # tmp_pydeps: pip --target 依赖
export VCDD_XPERT_DIR=/home/geng/vcdd-0.3.0/tmp_xpert       # XPert 模型权重
export VCDD_UNIMOL_WEIGHT_DIR=/home/geng/vcdd-0.3.0/tmp_unimol_weights
```

| 组件 | 版本/位置 | 作用 |
|---|---|---|
| vcdd | 0.2.0（本仓库 `vcdd/`） | 四段管线框架（YAML 配置 + ledger 断点续跑） |
| Vina-GPU 2.1 | 源码编译 quickvina2_gpu | stage1 批量对接（先跑 `vina_fix.sh`） |
| Boltz-2.2.1 | conda env + `~/.boltz/` 权重 | stage2 结构预测 |
| cuequivariance 三件套 | `cuequivariance_torch` / `ops_cu12` / `ops_torch_cu12` | **Boltz GPU 推理必需** |
| XPert / Uni-Mol | `tmp_xpert/`、`tmp_unimol_weights/` | stage3 转录组预测 |
| RDKit / python-docx | conda env + `tmp_pydeps/` | 分析与 Word 报告 |

---

## 2. 目录结构

```
vcdd-0.3.0/
├── README.md                  # 本文件（项目复现指南）
├── README-vcdd-upstream.md    # vcdd 框架上游说明
├── run_dnapkcs.yaml           # 管线主配置（stage1–4 全部参数）
├── DNA-PKcs/                  # 受体准备（7OTW 口袋裁剪，423 aa）
├── dnapkcs_run/               # 管线运行产物（见 §5）
├── msa_out/receptor_chain_A_vcdd.a3m   # 受体离线 MSA（1221 条同源）
├── msa_receptor_vcdd.fasta    # vcdd 实际受体序列（MSA 必须与之逐字符一致）
├── scripts/                   # 分析/报告脚本（§6）
├── vina_fix.sh                # Vina-GPU 构建与修复（SMALL_BOX 内核补丁等）
├── scripts/msa_fetch.py       # 离线 MSA 查询（ColabFold REST 直调 + NUL 清洗）
├── scripts/tmp_pocket_crop.py # 长受体 → 口袋片段裁剪工具
├── vcdd/  tests/  research/   # vcdd 框架本体与开发资产（勿删）
├── tmp_xpert/  tmp_unimol_weights/  tmp_pydeps/   # 模型权重与依赖
└── tmp_data/                  # 旧训练数据缓存（论文写作时自行甄别处置）
```

---

## 3. 管线复现（四段全命令）

> 全部命令在仓库根目录执行。耗时参考：stage1 ≈ 61 h、stage2 ≈ 9 h（双卡）、
> stage3 ≈ 15 min、stage4 ≈ 2.3 h。

```bash
cd /home/geng/vcdd-0.3.0
export PYTHONPATH=/home/geng/vcdd-0.3.0:tmp_pydeps

# 0) 库/受体准备（一次性）
python -m vcdd.cli prep-library   --config run_dnapkcs.yaml --out dnapkcs_run
python -m vcdd.cli prep-receptor  --config run_dnapkcs.yaml --out dnapkcs_run

# 1) Vina-GPU 初筛（20 万分子）
nohup python -m vcdd.cli stage1 --config run_dnapkcs.yaml --out dnapkcs_run \
  > dnapkcs_run/logs/stage1.nohup.log 2>&1 &

# 2) Boltz-2 精筛（top500，双卡）
nohup python -m vcdd.cli stage2 --config run_dnapkcs.yaml --out dnapkcs_run \
  > dnapkcs_run/logs/stage2.nohup.log 2>&1 &

# 3) XPert 转录组打分（top200，A375 / 10 µM / 24 h）
export VCDD_XPERT_DIR=/home/geng/vcdd-0.3.0/tmp_xpert
export VCDD_UNIMOL_WEIGHT_DIR=/home/geng/vcdd-0.3.0/tmp_unimol_weights
nohup python -m vcdd.cli stage3 --config run_dnapkcs.yaml --out dnapkcs_run \
  > dnapkcs_run/logs/stage3.nohup.log 2>&1 &

# 4) 生成式优化（objective Top10 亲本；GA + R-group；Vina 重对接）
nohup python -m vcdd.cli stage4 --config run_dnapkcs.yaml --out dnapkcs_run \
  > dnapkcs_run/logs/stage4.nohup.log 2>&1 &
```

**配置要点（`run_dnapkcs.yaml`）**：

| 段 | 关键项 | 值 | 说明 |
|---|---|---|---|
| objective | genes_up | CDKN1A, GADD45A | DDR 激活共识（上调基因） |
| objective | target_gene | PRKDC | DNA-PKcs 敲除一致性打分 |
| stage1 | engine | quickvina2_gpu | Vina-GPU（gpu_batch_size 200） |
| stage2 | local_boltz | devices [0,1]；`protein_msa_files: {A: msa_out/…a3m}`；`use_msa_server: false` | 离线 MSA |
| stage4 | top_parents | 10 | 亲本数（自 20 调低） |
| stage4 | generators | constrained_ga, rgroup | 双生成引擎，analogs_per_parent 50 |

---

## 4. 关键修复备注（复现必读的踩坑记录）

1. **Vina-GPU 2.1**（`vina_fix.sh`）
   - meeko 受体预处理会留下 (0,0,0) 的假氢 UNL → 对接前剔除；
   - 二进制需 `-DBUILD_KERNEL_FROM_SOURCE` 编译（运行时依赖源码根目录）；
   - 内核缺 `-DSMALL_BOX` 宏 → 改 `lib/main_procedure_cl.cpp` 后重编；
   - 部署用 `make out` 的无宏版本，任意 cwd 可运行。

2. **离线 MSA**（`msa_fetch.py`）
   - config 用 `protein_msa_files: { A: <path>.a3m }` + `use_msa_server: false`，同蛋白只查一次；
   - **a3m 首行序列必须与 vcdd input.yaml 的 protein.sequence 逐字符一致**，否则 Boltz 报
     `MSA does not match input sequence, creating dummy` 后崩溃；
   - ColabFold `result/download/<id>` 返回 gzip+tar → 解包取 `*.a3m`；
   - a3m 常含 NUL → 清洗（只保留 `>` 行与标准氨基酸/`-` 行）。

3. **Boltz GPU 推理**
   - 缺失 cuequivariance 三件套 → `ModuleNotFoundError`，推理永不启动（CPU 空转、GPU 0%）；
     装齐后 26–37 s/分子；
   - 短链可用 self-MSA（自身序列即 a3m，零网络）。

4. **stage2 位点 QC 坐标帧**
   - Boltz 自由折叠构象与晶体 PDB 不同坐标系，`ligand_in_box` 必须先把受体叠合
     （Superimposer）再比较；未叠合的 `pose_qc.parquet` 数值不可直接引用。

---

## 5. 产物索引（dnapkcs_run/）

| 路径 | 内容 |
|---|---|
| `stage1/ranking.parquet` | 167,342 分子 Vina 分数（-11.6 ~ +0.8 kcal/mol） |
| `stage2/consensus_ranking.parquet` | 500 分子 Boltz-2 结果（ipTM/ptm/consensus_rank） |
| `stage2/boltz/local/*/out/…/predictions/` | 每分子 CIF 结构 + 置信度 JSON |
| `stage3/objective_scores.parquet` | 200 分子 XPert 打分（objective_rank） |
| `stage3/signatures/xpert.parquet` | **200 × 978 基因 log2FC 全矩阵**（可算任意基因签名） |
| `stage3/top50_ddr_inhibitors.csv` | 前 50 DDR 抑制候选（修复签名排序） |
| `stage3/DNA-PKcs_前50_DDR抑制候选_含结构图.docx` | Top50 报告（含 2D 结构图） |
| `stage4/analogs.parquet` | 457 类似物（SMILES / 改造描述 / Δ分数 / guardrail） |
| `stage4/pareto.parquet` | Pareto 前沿（53 个） |
| `stage4/DNA-PKcs_stage4_生成式优化报告.docx` | Stage4 详细报告（含 Top15 结构图 + SAR） |

**Top50 DDR 候选评分口径**（见 `scripts/top50_ddr.py`）：
`repair_inhibition` = 16 个修复核心基因（BRCA1, RAD51C, CHEK1/2, MSH6, LIG1, PCNA,
RFC2/5, RPA1/2/3, DDB2, TP53BP1, TP53, TOP2A）平均下调幅度；
`ddr_inhibition = 0.6 × 修复抑制 + 0.4 × DDR激活(gene_score)`。

---

## 6. 分析/报告脚本（scripts/）

| 脚本 | 用途 |
|---|---|
| `top50_ddr.py` | 由 xpert.parquet 计算修复抑制签名 → Top50 CSV + Word |
| `top50_docx_imgs.py` | Top50 报告嵌入 2D 结构图 |
| `gen_stage4_report.py` | Stage4 详细报告（统计 + Top15 结构图 + SAR 规律） |

```bash
export PYTHONPATH=/home/geng/vcdd-0.3.0:/home/geng/vcdd-0.3.0/tmp_pydeps
python -B scripts/top50_ddr.py
python -B scripts/top50_docx_imgs.py
python -B scripts/gen_stage4_report.py
```

---

## 7. 复现注意

- 所有命令从仓库根目录执行（`tmp_pydeps` 为相对路径）；
- stage2 MSA 路径写入生成的 input.yaml；换机需重跑 prep 阶段；
- GPU 数量在 `stage2.local_boltz.devices` 指定；
- 沙箱受限时用 `pip install <pkg> --target tmp_pydeps` 并加入 PYTHONPATH。