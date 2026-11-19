# 教师 K/V 输入修复头

冻结 Draft 和 Target，将上一验证轮合法边界的教师 K/V 投影成 128 维控制向量，
输入 rank-32 输出侧修复头，再投影得到下一轮 Draft logits。
本仓库保留历史 ACK、ACK_H、ACK_KV 三组匹配容量的实现。
Draft 层内注意力不改；在 Draft 内插入 K/V 槽的是另一个 `spec-draft-single-kv-slot` 仓库。

```text
教师验证前向中已有的末两层 K/V
  → 最近四个已处理位置及有效掩码
  → K/V 投影（共 98,304 参数）→ 128 维 control
  → 输出侧修复头(hidden, control, position) → frozen LM head → Draft 候选
```

## 原实验与实现

| 条件 | 教师输入 | 共同计算 |
|---|---|---|
| ACK | 仅普通回执和本地状态 | 本地状态进入相同的 K/V 双流投影 |
| ACK_H | 再加合法边界的 4096 维教师 hidden | 相同投影和修复容量 |
| ACK_KV | 再加教师第 34/35 层、最近 4 个已处理位置的真实 post-RoPE K/V | 相同投影和修复容量 |

K/V shape 是两份 `[2,4,8,128]` BF16，输入张量约 32 KiB；H 是 8 KiB。
这是信息输入对照，**没有把 K/V 压成 8 字节，也没有测出这些张量的真实双机传输耗时**。
纠正/bonus token 尚未被 Target 消费，不能进入取样 K/V。在线使用本请求本轮实际
Target 前向中的 K/V；保存的 RAW K/V 仅用于离线训练，不作在线反馈替代。

`runtime/ack_head.py`、`teacher_kv_task59_analyst.py`、`same_forward_kv.py`、
`cache_adapter.py` 和原共享损失保留原字节。`legacy/` 保存原训练/评估/路径源。
`evidence/provenance.json` 及各 `*_adapter.patch` 说明迁移改动。

迁移版去掉原六组实验完成的队列前置，直接读取已有合法公共窗口；不生成虚假的前序完成回执。
按 `assets.template.json` 配置路径，支持将旧 `/data` 前缀重映射到新挂载目录。
训练仍使用原数据顺序、私有 Gumbel RNG、AdamW、断点、每窗归一和按请求等权的校准选轮。

历史默认：seed42、5 epochs、batch32、每轮 4,897 更新、合计 24,485 更新。
损失为每窗有效位置等权 CE 再平均窗口，校准亦按有效位置等权。
`--loss front` 另设 8→1 训练权重，校准口径保持等权；它是新配置，不能冒充已有历史结果。
两种配置必须用不同输出目录。本仓库没有 collect 阶段，不补采新的 Target 训练分布。

历史 503 题原件：RAW 2.118632；ACK 2.231466；ACK_H 2.229876；ACK_KV 2.230423。
指标为总接受 Draft token / 总验证轮，详见 `evidence/historical_503_audit.json`。
这批开发结果没有证明 K/V 比普通回执更好，也没有证明端到端加速。
本次迁移入口尚未重新完成 GPU 训练/503 题评测；CPU 核验范围见 `evidence/cpu_smoke.json`。

## 本机检查

Python 3.11+。GPU 主机按 CUDA 版本安装匹配的 PyTorch 2.9.1，再安装其余依赖。
Transformers 固定为 4.57.1，与原 K/V 观察器的 Qwen3 源码哈希匹配。
4.57.3 的该文件哈希不同，原检查会拒绝；本仓库保留这个检查，没有关闭它。
请为此仓库使用独立环境，避免改动其余实验的依赖。

```text
python -m pip install -r requirements.txt
python experiment.py check-repo
python experiment.py cpu-smoke
```

CPU 冒烟使用原真实头、AdamW、safetensors 和微型 Qwen3，检查容量匹配、零残差、
标签隔离、有效位置掩码、信号梯度、路径重映射、断点恢复及真实 post-RoPE K/V。
它不等同于 0.6B/8B 生产权重或 GPU 数值/耗时验证。

## 换机输入

源码仓库不包含模型权重、RAW5000 分片、训练 tensor cache、教师 K/V 特征、训练断点。
`inputs/` 包含原 503 题、原模型清单及历史注册/冻结绑定，数据与权重另行迁移。

```text
models/Qwen3-0.6B/             原模型清单全部文件
models/Qwen3-8B/               评估时使用
train/index.jsonl              原 SHA 6b8797ea…c3ff6
train/cache/<request_id>.safetensors
teacher-kv/common_train.jsonl  原 SHA 5c0ad85e…16f24
teacher-kv/common_calibration.jsonl  原 SHA 9f462271…da5
RAW 与 teacher features.safetensors  按公共窗口原字段保留或 path_map 重映射
```

首次 configure 将代码和输入绑定写入输出目录；已存在但不一致的配置拒绝覆盖。
`preflight` 核源文件、题单、模型及训练/教师窗口索引哈希；不是整个 tensor 数据集的逐字节审计。
各窗口还由原读取器核标签和共同边界。独立确认 300 题不在执行范围。

## 独立训练与评估

在 `assets.json` 填实际路径。下例只表示手动执行方法，不自动派 GPU。
所有同机实验应配置同一 `gpu_lock`。原 GPU 入口要求 Linux/fcntl，取得锁并检查无外部
compute 进程后才加载模型，不抢占已有任务。

```text
python experiment.py configure --assets assets.json --out /srv/spec-runs/teacher-kv/equal --loss equal
python experiment.py preflight --out /srv/spec-runs/teacher-kv/equal
python experiment.py train --out /srv/spec-runs/teacher-kv/equal
python experiment.py eval --out /srv/spec-runs/teacher-kv/equal
python experiment.py compare --out /srv/spec-runs/teacher-kv/equal
```

默认三组均执行。只跑 K/V 臂时 configure 加 `--arms ACK_KV`，仍有 RAW 接受量对照。
评估先做 EOS/截断小样本与原生 RAW 对账，再继续原 503 题；Target 每步实际验证，
封存答案只作事后断言。缺少旧六题独立 oracle 文件时重做相应原生 Target 检查。
已封存逐请求结果会核验后续跑；不重复生成已完成请求。

本入口的训练断点保存优化器、位置、更新次数和独立 RNG；同配置可续跑。
若只复用历史训练好的头，不应伪造本轮训练完成：创建一个各臂 `path/sha256/epoch` 的
JSON，执行以下命令直接绑定真实权重，之后 eval 即可。

```text
python experiment.py import-checkpoints --out /srv/spec-runs/teacher-kv/equal --checkpoints checkpoints.json
```

历史原队列的训练断点注册哈希不同；该命令用于评估旧权重，不把它伪装成本入口的续训状态。
RAW 接受量可复用原件，但归档 RAW 的读取耗时不是真实推理测速。
新服务器仍需先验证模型/精度/最终输出，再做单独的端到端性能测量。

历史导入核对 inputs/historical_selected_arms.json 中的臂名、选中轮次和权重 SHA-256，
仅适用于历史等权训练。训练和历史导入各用独立的输出目录。比较时需要保留实际选中权重、
逐请求 oracle、回执及其引用的边界张量。训练恢复会核对臂、冻结配置和更新位置。
旧运行使用原源码绑定与断点格式，请用相应原版本继续。
