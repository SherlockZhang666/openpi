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
