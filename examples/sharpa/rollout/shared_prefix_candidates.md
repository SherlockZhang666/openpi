# N 个候选共用一次 VLM 前缀：依据、分析与验证（2026-09-14）

**一句话**：一次请求的 N 个候选来自**同一个观测**，图像 + prompt 这段 VLM 前缀对它们完全相同。
原实现把观测复制 N 份当 batch，前缀被重算 N 遍；现在前缀只算一次、把 KV cache 复制给 N 个候选，
只有 action expert 的去噪循环按 N 路跑。结果与原实现等价，RTX 5090 Laptop 上 N=4 从
p50 571 / p95 646 ms 降到 **p50 202 / p95 247 ms**（经 websocket），放进了 rollout 客户端 267 ms 的预算。

改动只在服务端 `src/openpi/policies/candidate_policy.py`，模型文件（`models/pi0.py`）和机器人客户端都没动。
候选协议本身（请求 / 响应的键）不变，见同目录 `README.md`。

---

## 1. 问题：真机那台卡上 N=4 放不下

handoff-round3 §3 要求每个 chunk 边界采 N=4 个候选、随机执行一个。客户端的推理预算是
`--infer-lead 8` 步 @ 30 Hz = **267 ms**（`--infer-lead` 的含义见客户端 `loop.py`：下一次推理提前
这么多步发出，新 chunk 也从这个下标开始执行）。

交接文档里的延迟是 A100-80GB 上测的，真机用的是笔记本上的 RTX 5090 Laptop（24 GB）。用 `ping_policy.py`
经 websocket 实测（原实现）：

| N | p50 | p95 | 267 ms 预算 | A100（交接文档，原实现） |
|---|---|---|---|---|
| 1 | 183 ms | 251 ms | ✅（余 15 ms） | 92 ms |
| 2 | 308 ms | 364 ms | ❌ 要 `--infer-lead 12` | 128 ms |
| 4 | 571 ms | 646 ms | ❌ 要 `--infer-lead 21` | 193 / 235 ms |

GPU 上没有别的进程（`nvidia-smi` 只有 server 一个），不是资源争用：这张卡单次推理本来就比 A100 慢约 2 倍，
而延迟又随 N 近似线性涨。

`--infer-lead 21 --chunk-steps 9` 虽然"放得下"，但每个 chunk 只执行第 21–29 个动作——离观测 0.7–1 s、
模型预测最不准的一段，观测滞后 0.7 s，近乎开环；录下的数据不再代表这个 policy 的正常行为，所以不可取。
学校的 GPU 服务器在防火墙后，把 server 放远端也不现实。

## 2. 分析：慢在哪里

`models/pi0.py` 的 `Pi0.sample_actions` 分两段：

1. **前缀**：`embed_prefix(observation)` 把各路图像（SigLIP）和 tokenized prompt 嵌入，
   过一遍 PaliGemma（2B）得到 `kv_cache`。**只做一次**。
2. **去噪**：`while_loop` 做 `num_steps`（默认 10）步 Euler（或 `sde_eta` 的随机步），每步把
   带噪动作 + 时间嵌入成 suffix，由 action expert（约 300M）带着 `kv_cache` 做注意力，得到速度场 `v_t`。

原来的 `_sample_candidates` 把**同一个观测 `np.repeat` 成 N 份**再调 `sample_actions`，于是第 1 段——
整个流程里最重的一段——被按 batch=N 并行重算。N 个候选唯一不同的是初始噪声 `x_1`，它只进入第 2 段。

### 为什么共用前缀是**精确**的，不是近似

- 前缀的 KV 只由 observation 决定：`sample_actions` 算 `kv_cache` 时根本没有 suffix 参与，
  而去噪循环里 `PaliGemma.llm` 返回的新 cache 被丢弃（`(prefix_out, suffix_out), _ = ...`），
  cache 在循环里从不更新。所以它与噪声、时间步、候选下标全都无关。
- 注意力掩码上，suffix 的 `ar_mask` 以 `True` 开头，前缀 token 看不到 suffix（`make_attn_mask`），
  这是 π0 的"前缀-后缀"结构本身保证的。
- 因此 N 份观测算出来的 N 份 KV cache 逐元素相同；算一份再复制，数学上就是同一个东西。

## 3. 实现

`candidate_policy.py`：

- `_shared_prefix_sample_actions(model, rng, observation, noise, *, num_steps, sde_eta)`：
  `observation` 是 batch 1，`noise` 是 batch N。前缀在 batch 1 上算出 `kv_cache` 后：
  - `kv_cache` 每个张量沿 **axis=1** 复制 N 份——`gemma.Module` 的层是 `nn.scan` 出来的，
    cache 形状是 `(层数, batch, token, kv_heads, head_dim)`；
  - `prefix_mask` 沿 batch 复制 N 份；
  - `observation` 整体沿 batch 复制 N 份（π0 的 `embed_suffix` 要读 `observation.state` 做 state token；
    π0.5 不读，复制的开销可以忽略）；
  - 去噪循环逐行照抄 `Pi0.sample_actions`，包括 `sde_eta` 分支和它的 rng 用法
    （`jax.random.fold_in(rng, 1)` 起步、每步 split），所以同 rng、同噪声下与原实现一一对应。
- `_shared_prefix_fn(policy)`：用和 `nnx_utils.module_jit` 相同的"冻结 module 状态"方式 `jax.jit`，
  每个 policy 建一次；和原实现一样，每个新 N 编译一次。
- `_sample_candidates(..., share_prefix=True)`：只在 **JAX 且模型是 `Pi0`（π0 / π0.5）** 时走共享前缀；
  PyTorch 模型或其它模型类型自动回到原来的 batch=N 路径。噪声的生成（`policy._rng` 的 split 方式、温度缩放）
  两条路径完全相同。
- `CandidatePolicy(share_prefix=True)` 是默认；`share_prefix=False` 退回原实现。
- 响应多一个键 `candidate_sampler`：`"shared_prefix"` 或 `"batched"`，说明这次走的是哪条路。

**为什么照抄循环而不是改 `pi0.py`**：这个 repo 是上游 fork，要反复 rebase；候选逻辑一直放在自己的模块里、
不碰上游文件（`candidate_policy.py` 模块说明的最后一段）。代价是循环体有两份——这由下面的等价性测试钉住：
`pi0.py` 的采样循环一旦改了而这里没跟，测试会失败。

## 4. 验证

### 4.1 等价性（CPU，`debug_pi05` 随机权重，float32）

`src/openpi/policies/candidate_shared_prefix_test.py`：

| 测试 | 内容 |
|---|---|
| `test_shared_prefix_gives_the_batched_candidates[eta=0.0 / 0.5]` | 同 seed、同噪声，共享前缀与 batch=N 的候选 `atol=1e-5` 一致，噪声逐位相同 |
| `test_candidates_still_differ_with_a_shared_prefix` | 共享前缀没有让候选塌成同一个 |
| `test_the_wrapper_shares_the_prefix_by_default` | 默认走共享前缀，`share_prefix=False` 走原路径 |

原有的 `candidate_policy_test.py`（包括"每个候选都能用它的噪声单独 `infer` 复现"）在新默认下照样通过：
两个文件共 **20 passed**。

```bash
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q \
    src/openpi/policies/candidate_policy_test.py src/openpi/policies/candidate_shared_prefix_test.py
```

### 4.2 真实权重上的数值差异（RTX 5090 Laptop，`egg_70ep_b64/15120`）

同一个观测、同一个 rng（因而同样的 4 份噪声），以"用候选 i 的噪声单独 `policy.infer` 一次（batch 1）"为参照：

| 比较 | 最大绝对差 |
|---|---|
| **原实现**（batch=4）vs batch-1 参照 | 9.96e-03 |
| **共享前缀** vs batch-1 参照 | 5.17e-03 |
| 共享前缀 vs 原实现 | 8.41e-03 |
| 参照：候选之间本身的差距 | 6.18e-02 |

另一次独立测量：共享前缀 vs 原实现最大差 6.48e-03（手指列 6.48e-03 rad，腕部列 2.86e-04），候选间差距 6.77e-02。

解读：CPU float32 上两者一致到 1e-5，GPU 上差到 1e-2 量级，而**原实现自己**与 batch-1 参照也差这么多（甚至更多）。
这是 GPU 混合精度下不同 batch 形状选到不同 kernel 引起的数值误差，不是算法差异；共享前缀并不比原实现离"单独推理"更远。

### 4.3 延迟

直接调用（不经 websocket，含输入/输出变换），每组先编译一次：

| 配置 | 次数 | p50 | p95 | p99 | max | 首次调用（编译） |
|---|---|---|---|---|---|---|
| N=1 普通 `infer` | 80 | 180 | 198 | 210 | 231 ms | 18.7 s |
| N=4 原实现 | 20 | 546 | 623 | — | 623 ms | 23.7 s |
| **N=4 共享前缀** | 80 | **203** | **214** | **221** | **223 ms** | 5.3 s |
| N=8 共享前缀 | 80 | 226 | 237 | 266 | 278 ms（1/80 > 267） | 12.8 s |

（第一轮只测 20 次时，N=4 共享前缀的 p95 = max = 283 ms，是刚编译完的偶发慢调用；80 次里没有一次超过 267 ms。）

**端到端**，server 用 `scripts/sharpa_serve.sh ... --num-candidates 4 --noise-temperature 1.0` 起，客户端
`ping_policy.py --n 40 --num-candidates 4`：

```
steady   p50           202 ms   over 39 calls
         p95           247 ms
         max           271 ms
OK -- p95 inference fits in 267 ms with 20 ms to spare.
```

N=4 现在只比 N=1 慢约 20 ms：去噪部分虽然按 N 路跑，但 action expert 小、suffix 只有 30 个 token，
在 GPU 上并行几乎不增加耗时；原来多出来的三百多毫秒几乎全是重复的前缀。

## 5. 考虑过、没有采用的方案

| 方案 | 为什么没用 |
|---|---|
| N=4 + `--infer-lead 21 --chunk-steps 9` | 执行 chunk 最不准的尾段、观测滞后 0.7 s，行为不代表 policy 本身 |
| 改用 N=2 + `--infer-lead 12` | 每边界候选少一半；观测滞后变大，与之前的基线 rollout 行为不一致 |
| server 放学校 A100，SSH 转发 | 防火墙；每次请求要上传约 0.7 MB 图像，延迟取决于网络，不可控 |
| 去噪步数 10 → 5 | 改变了 policy 的采样分布，录下的不是这个 policy 的行为 |
| 被选中的候选先算（N=1），其余事后用存下的观测和噪声补算 | 也可行（`sde_eta=0` 时候选由观测+噪声唯一确定），但要改服务端协议（请求带噪声）和客户端（存原始观测、事后补算），改动大得多。共享前缀已经放得下，留作后备 |

## 6. 注意事项

- **交接文档里 A100 的延迟数字是原实现测的**（N=4 193 ms、N=8 310 ms）。服务器端同步这个改动后会更快，需要重测。
- 每个新 N 仍然要 JIT 编译一次（本机 N=4 首次调用约 20 s），机器人使能前用实际的 N 预热——客户端 `cli.py` 已经这么做。
- 余量只有 ~20 ms。真跑时偶尔超预算只会打一行 `inference was not ready at the chunk boundary`；
  若频繁出现，用 `--infer-lead 9`（300 ms），见 `RUNBOOK.md` §9b ②。
- 以后如果 `models/pi0.py` 的 `sample_actions` 改了（例如采样器、掩码、adaRMS 条件），要同步改
  `_shared_prefix_sample_actions`；`candidate_shared_prefix_test.py` 会在两者不一致时失败。
