# The rollout client moved

真机执行客户端现在住在机器人仓库里：

    ~/openarm/openarm_track/rollout/

包括 `cli.py`（原来是 `main.py`，后来在那边被拆开）、`ping_policy.py`、`probe_hand_units.py`、离线测试、
`RUNBOOK.md`（逐步带判据的操作手册）和 `README.md`（架构说明）；其中
`golden_trace.npz` 与 `GOLDEN.md` 固定了迁移前的行为基线，`golden_test.py`
就是拿它来做校验的。

**为什么搬走。** 这份客户端 import 的是 rig 自己的 `collect/` 模块 —— `RosSources`、
`StateReceiver`、`camera.resolve_device` —— 也就是当初录训练数据的那几个读取器。
它跑在 rig 的解释器上，不在 openpi 的。而 openpi 是一个上游 fork，把机器人代码留在这里
只会在每次 rebase 时制造冲突。

**openpi 这边还需要提供的东西**（客户端仍然依赖）：

- `scripts/serve_policy.py` —— 策略服务端，跑在有 GPU 的机器上：

      uv run scripts/serve_policy.py policy:checkpoint \
          --policy.config=sharpa_egg --policy.dir=<ckpt>/<step>

- `packages/openpi-client/` —— websocket + msgpack 客户端。客户端把它上 `sys.path`
  而不是 pip 装（它声明 `numpy<2.0`，rig venv 跑 numpy 2.5）。位置由 `OPENPI_ROOT`
  指定，默认 `/home/yiming/Public/openpi`。
- `src/openpi/training/sharpa_configs.py`、`src/openpi/policies/sharpa_policy.py`
  —— 训练侧的配置与数据变换。
- `examples/sharpa/convert_sharpa_data_to_lerobot.py` —— 数据转换。它的
  `_palm_centric_wrist` 定义了 28 维动作里腕部那 6 维的含义，客户端的
  `actions_test.py` 逐字复制了它来做往返验证。**改这里就得同步改那边**，
  那个测试就是为了在没同步时炸出来。

历史在 openpi 的 git 里仍然完整：`git log --follow examples/sharpa/rollout/main.py`。

## 采 N 个候选 + 噪声温度（2026-09-13）

服务端（`src/openpi/policies/candidate_policy.py`）可以一次返回 N 个候选 chunk，给 tactile
steering 用。**默认 N=1、T=1.0，行为与之前逐位相同**，现有客户端不改也照常工作。

启动时给默认值（`sharpa_serve.sh` 会把多余参数原样传下去）：

    scripts/sharpa_serve.sh <ckpt>/<step> sharpa_egg --num-candidates 8 --noise-temperature 1.0

也可以在每个请求的 obs 字典里单独覆盖（这两个键在输入变换之前就被拿掉，模型看不到）：

| 请求键 | 类型 | 含义 |
|---|---|---|
| `num_candidates` | int ≥ 1 | 这次要几个候选 |
| `noise_temperature` | float ≥ 0 | 初始噪声 `x_1 = T·ε`，T=1 就是模型自己的采样器 |

N>1 或 T≠1 时响应多出这些键：

| 响应键 | 形状 | 含义 |
|---|---|---|
| `actions` | (30, 28) | **候选 0**，老客户端执行的就是它 |
| `actions_candidates` | (N, 30, 28) | 全部候选，经过与 `actions` 相同的输出变换 |
| `candidate_noise` | (N, 30, 32) | 每个候选的初始噪声（可复现：拿它单独调一次 `infer(noise=...)` 就得到同一个候选） |
| `num_candidates`, `noise_temperature` | 标量 | 本次实际用的值 |
| `candidate_sampler` | str | `"shared_prefix"`（默认，前缀只算一次）或 `"batched"`（原实现），见 [`shared_prefix_candidates.md`](shared_prefix_candidates.md) |

**客户端要改的**（在机器人仓库里，不在这里）：

1. 每个 chunk 边界在请求里带上 `num_candidates`（以及需要时的 `noise_temperature`）。
2. 从 `actions_candidates` 里选一个执行 —— 现在是**随机选**（random-of-N），以后换成 scorer。
3. **全部候选、`candidate_noise`、选中的下标、N、T 都录进 `flight.npz`**，否则事后分不清执行的是哪一个。

⚠️ 两条代价：

- **延迟随 N 增长**。2026-09-14 起 N 个候选**共用一次 VLM 前缀**（KV cache 复制给 N 路去噪，结果与原来等价），
  N 的代价主要只剩去噪部分：RTX 5090 Laptop 上经 websocket 实测 N=4 **p50 202 / p95 247 ms**（原实现 571 / 646 ms），
  N=8 直接调用 p95 237 ms。依据、等价性论证与全部测量见 [`shared_prefix_candidates.md`](shared_prefix_candidates.md)。
  客户端的预算是 267 ms（`--infer-lead 8 / --fps 30`），余量不大，**先用 `ping_policy.py` 带上 N 实测**。
  （旧数字供参考：原实现在 A100-80GB 上 N=1 92 ms、N=4 193 ms、N=8 310 ms、N=16 558 ms，
  `tactile_steering/dp2/out/dp2b_candidates.md`；同步这个改动后需重测。）
- **每个新 N 都要重新 JIT 编译**（几十秒）。机器人使能之前，用实际要跑的 N 先预热一次。

T≠1 时候选来自另一个分布：best-of-N 不受影响；加权重采样要做 `π/π_T` 修正（HANDOFF §7）。

## 随机（SDE）采样：`sde_eta`（分支 `sharpa-rollout-sde`，2026-09-13）

默认的采样器是确定性的：从初始噪声出发，10 步 Euler 积分 flow ODE，积分过程中不再有随机性。
`sde_eta` ∈ [0, 1] 换成 DDIM-η 式的随机采样器（`src/openpi/models/pi0.py::flow_ddim_eta_step`）：
每一步由速度还原出 x̂₀ 与 ε̂，再按 q(x_{t'} | x_t, x̂₀) 走下一步并注入新噪声。

    scripts/sharpa_serve.sh <ckpt>/<step> sharpa_egg --num-candidates 4 --sde-eta 0.5

也可以在请求里带 `sde_eta`。**默认 0 = 原来的采样器，逐位不变**；`sde_eta` 只对 JAX 模型有效（PyTorch 模型会报错）。
**不需要重训**：用的是同一个速度场，只换了推理时的积分方式。

⚠️ **它不是「让候选更分散」的旋钮。** 模型准确时，η 取任何值采到的都是**同一个分布**；η 改变的是怎么采，
不是采哪个分布。而且步数少时它往往让样本**更集中**：一维高斯上用解析速度场实测（10 步），
样本标准差 η=0 为 0.247、η=1 为 0.219（真值 0.3）；步数加到几百步后各 η 都收敛到 0.3。
要比策略本身更分散，只能换分布（例如 `noise_temperature`）。η 值不值得用，先离线量候选散布
（`tactile_steering/dp2/dp2b_candidates.py`）再决定。

⚠️ `sde_eta > 0` 时每一步的噪声不在 `candidate_noise` 里，**候选不能再只凭 `candidate_noise` 复现**。
