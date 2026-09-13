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

**客户端要改的**（在机器人仓库里，不在这里）：

1. 每个 chunk 边界在请求里带上 `num_candidates`（以及需要时的 `noise_temperature`）。
2. 从 `actions_candidates` 里选一个执行 —— 现在是**随机选**（random-of-N），以后换成 scorer。
3. **全部候选、`candidate_noise`、选中的下标、N、T 都录进 `flight.npz`**，否则事后分不清执行的是哪一个。

⚠️ 两条代价：

- **延迟随 N 线性涨**：batch=N 会把 VLM 前缀并行重算 N 遍。`egg_70ep_b64/15120` 在 A100-80GB 上实测
  （10 步去噪，`tactile_steering/dp2/out/dp2b_candidates.md`）：N=1 92 ms、N=4 193 ms（最大 235）、
  **N=8 310 ms（最大 324）**、N=16 558 ms。客户端的预算是 267 ms（`--infer-lead 8 / --fps 30`）⇒
  **N=4 放得下，N=8 放不下**（要 `--infer-lead` ≥ 10）。真机的卡不是 A100，**先用 `ping_policy.py` 带上 N 实测**。
- **每个新 N 都要重新 JIT 编译**（几十秒）。机器人使能之前，用实际要跑的 N 先预热一次。

T≠1 时候选来自另一个分布：best-of-N 不受影响；加权重采样要做 `π/π_T` 修正（HANDOFF §7）。
