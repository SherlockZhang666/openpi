# 真机 rollout 操作手册

在 OpenArm + Sharpa Wave 左手真机上跑训练好的 pi0.5。

**这份是"站在机器人前面照着敲"的操作单**；为什么这么设计、动作怎么解码、坐标约定是什么，
在同目录的 [`README.md`](README.md)。

> 本文用中文，和 `~/openarm/openarm_track/vr/README.md`、`vr/collect/README.md` 一致 ——
> 跑这套流程时要来回对照的就是那两份。目录里的代码和 `README.md` 保持英文，跟 openpi 上游一致。

**每一步都给了判据。判据不过就停下，别往下走** —— 后面的错误会以完全无关的形式冒出来。
标 ⚠️ 的步骤真机会动。

---

## 0 · 终端分配

一次 rollout 要开 6 个终端。先想清楚哪个是哪个，比中途找回来容易：

| # | 干什么 | 谁的解释器 |
|---|---|---|
| 1 | policy server | `openpi/.venv`（`sharpa_serve.sh` 自己处理） |
| 2 | `ping_policy.py` / 临时查东西 | rig venv |
| 3 | 手：`probe_hand_units.py` | rig venv |
| 4 | 臂 bring-up，**起了就不要碰** | ROS |
| 5 | teleop 节点 | rig venv + ROS |
| 6 | `cli.py`，rollout 客户端 | rig venv + ROS |

### 每开一个新终端，先跑这段

```bash
export V=/home/yiming/openarm/openarm_track/vr          # rig checkout
export OPENPI=/home/yiming/Public/openpi
```

**终端 4、5、6 还要多跑这三行**（要 ROS）：

```bash
source /opt/ros/jazzy/setup.bash
source /home/yiming/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=0
```

终端 1、2、3 不需要 ROS。变量是 per-shell 的，**新开一个终端就要重设一次** —— 下面每步的命令
都假设当前终端已经跑过对应的这段。

---

## 1 · 起 policy server（不动机器人，可以提前很久做）

```bash
# 终端 1
cd $OPENPI
scripts/sharpa_serve.sh ./checkpoints/sharpa-pi05/egg_70ep_b64/15120
```

**判据**：日志里两行都要有 ——

```
INFO:root:Loaded norm stats from .../egg_70ep_b64/15120/assets/local_repo
INFO:websockets.server:server listening on 0.0.0.0:8000
```

第一行是重点。norm stats 必须来自 **checkpoint 自己的 `assets/`**；只拷了 `params/` 的
checkpoint 会**加载成功但跑无归一化的策略** —— 不报错，臂往别处去。`sharpa_serve.sh` 会在
文件缺失时直接拒绝启动。

```bash
# 终端 2
cd ~/openarm/openarm_track/rollout
$V/.venv/bin/python ping_policy.py --n 40
```

**判据**：`OK -- p95 inference fits in 267 ms`，且 chunk shape 是 `(30, 28)`。

首次调用 36 秒是 XLA 编译，正常，**每次重起 server 都要重来一遍**。所以别等机器人已经使能了
才第一次连 —— 在第 3 步之前就把 server 预热好。

### 1b · 换成 diffusion policy（baseline_dp）时，第 1 步改成这样

server 用 rig 的 lerobot venv（这台机器上唯一能在 RTX 5090 上跑 torch 的），不是 openpi 的：

```bash
# 终端 1
cd /home/yiming/Public/baseline_dp
./serve_dp.sh trained_model/checkpoint_epoch_300 --sampler ddim --steps 20 --samples 4   # 真跑用这个，~75 ms
# DDPM 50 步（离线 157 ms，真跑时量到 215-300 ms，每个 chunk 边界都 stall）：
./serve_dp.sh trained_model/checkpoint_epoch_300
# 训练时的完整 100 步（~315 ms 一个 chunk，任何 lead 都装不下）：
./serve_dp.sh trained_model/checkpoint_epoch_300 --steps 100
```

**判据**：`serving on ws://0.0.0.0:8000 ({'model': 'dp', 'horizon': 16, 'obs_horizon': 2, ...`
一行里要有 `'sampler'` 和 `'warmup_ms'`。`norm_stats.json` 缺失时脚本直接拒绝启动（训练脚本把
归一化统计量写死在 `train.py` 里，checkpoint 自己不带）—— 第一次跑先
`<venv>/bin/python dp_policy.py --write-norm-stats trained_model/checkpoint_epoch_300`。

```bash
# 终端 2（不需要 ROS）
cd /home/yiming/Public/baseline_dp
V_DP=/home/yiming/openarm/openarm_track/vr/.venv-lerobot/bin/python
$V_DP ping_dp.py --n 20                                   # 延迟 + 量程
$V_DP ping_dp.py --replay ~/openarm/data/engaged/20260903_002436_pick_up_the_egg/ep_0008
```

**判据**：第一条 `OK -- p95 inference fits in 200 ms`（`--infer-lead 6` 的预算，dp 的默认值）；
第二条 `OK -- the served policy reproduces the commanded wrist trajectory`，腕部误差 policy 列
明显小于 hold-current 列（正常是 2.5-3 mm 对 4.5-5 mm）。replay 走的是客户端一模一样的
payload，关节列、帧序、resize、反归一化任何一处接错都会在这里变成几十毫米。

第 8、9 步的命令把 `--prompt "pick up the egg"` 换成 `--backend dp`，**去掉** `--hand-norm-stats`
（统计量从握手里来，带了会被拒），真跑加 `--wrist-step-lin 60 --wrist-step-ang 20 --wrist-slew-lin 6 --wrist-slew-ang 2`（见下）。dry run 判据第 1 条 head 变成 `640x400`（dp 的默认
`--image-short-side 400`），第 3 条预算变成 200 ms。

⚠️ **场景要和训练一致**（`reference_start_scene.png`：鸡蛋放在指尖前方约一个手掌的距离、桌子中部，
碗在鸡蛋右上）。dp 输出的是绝对目标位姿，而且它是**从头部相机画面**里读腕部位置的（关节状态几乎
不影响）：鸡蛋/碗摆得和训练不一样，第一个 target[0] 就会偏几厘米，离链的当前位姿超过
20 mm / 5.7° 就在第 0 步被单步护栏拦下 —— 那是护栏在工作，不是 bug。

真跑之前先用只读工具看一眼（要 ROS、server 在跑、rig 起着；不发任何指令）：

```bash
$V/.venv/bin/python check_dp_start.py
```

它打出策略的 target[0] 到 FK(q_meas) 的距离，并把当前画面存到 `/tmp/dp_start_live.png`
对照 reference。个位数毫米最好；超过 20 mm 它会说 NOT READY。2026-09-11 实测：场景对齐后
第一步仍有 18 mm / 6.3°（策略自身的感知误差），转角刚好超过 5.7° 的护栏，所以真跑要带
`--wrist-step-lin 30 --wrist-step-ang 10`。这两个 flag 会把单步腕部护栏放宽到 30 mm / 10°，
对整个 run 的每一步都生效，跑飞时被拦得晚一点 —— 只放这么多，够第一步用就行。

**真跑还要带 `--wrist-slew-lin 6 --wrist-slew-ang 2`**（2026-09-11 run 004 的结论）。dp 输出的是绝对
目标，相邻两个 chunk 对"腕该在哪"的判断差几毫米（训练数据上离线量到 3.6 mm 中位），在真机上会
放大成振荡：chunk 边界跳一下 → 伺服滞后冲到 3 cm → 下一个 chunk 从画面里看到滞后的臂就往回拉 →
再下一个往前推，边界跳到 5-21 mm，抓蛋时 35 mm 被护栏拦停。slew 把链向目标的逼近限到每步 6 mm /
2°（训练数据单步最大 4.6 mm / 1.3°，分布内的动作碰不到它），边界的跳变变成几步训练速度的走位；
单步护栏照旧看**原始**距离，跑飞照样停。pi0.5 的增量指令不受影响。

server 也换成 DDIM-20 并加 `--samples 4`：真跑时 DDPM-50 量到 215-300 ms（离线 157），每个 chunk
边界都 stall；DDIM-20 真机上 78 ms。`--samples 4` 把同一观测的 4 次采样在一个 batch 里平均
（73 ms，不多花时间），离线把相邻 chunk 的分歧从中位 3.3 / 最大 8.7 mm 压到 2.3 / 4.4 mm。

run 005 / 006（带 slew）都走到了鸡蛋并合拢手指，然后在抓取瞬间被护栏拦下：那一刻腕部被手指
半遮住，相邻 chunk 的原始分歧到了 30 mm 和 40 mm / 12°。有了 slew，实际运动始终不超过每步
6 mm / 2°，单步护栏只是在检查"目标一致性"而不是限速，所以真跑放到
`--wrist-step-lin 60 --wrist-step-ang 20`；累积包络（离锚点 45 cm / 1.8 rad）不变。

### 1c · π0.5 多候选（N=4, T=1.0）时，第 1 步改成这样

第 9b 步（第三轮多候选）用这组。温度批次改成 T=1.5 / 2.0 时只改 `--noise-temperature`。

```bash
# 终端 1
cd $OPENPI
scripts/sharpa_serve.sh ./checkpoints/sharpa-pi05/egg_70ep_b64/15120 sharpa_egg \
    --num-candidates 4 --noise-temperature 1.0
```

```bash
# 终端 2
cd ~/openarm/openarm_track/rollout
$V/.venv/bin/python ping_policy.py --n 40 --num-candidates 4
```

**判据**：同第 1 步 —— `Loaded norm stats from .../15120/assets/...`、`server listening on 0.0.0.0:8000`；
ping 出 `OK -- p95 inference fits in 267 ms`，没有 `CHUNK PROBLEMS`。第一次调用是 N=4 的 JIT 编译（几十秒），不计入。
本机实测 N=4 p50 202 / p95 247 ms；ping 出五六百毫秒说明 openpi 不是带共享前缀采样的版本（见 9b ②）。

---
起CAN：
```
sudo ip link set can0 down 2>/dev/null
sudo ip link set can0 type can bitrate 1000000 dbitrate 5000000 fd on
sudo ip link set can0 up

sudo ip link set can1 down 2>/dev/null
sudo ip link set can1 type can bitrate 1000000 dbitrate 5000000 fd on
sudo ip link set can1 up
```

## 2 · 手上电

**先手后臂，关机反过来。** 手挂在左臂末端；让一只还在被驱动的手跟着载体臂做回零斜坡是自找麻烦。

手 `192.168.10.10`，主机网卡 `enp129s0` = `192.168.10.240/24`（NM 配置 `SharpaWave`）。

```bash
# 终端 3
cd ~/openarm/openarm_track/rollout
$V/.venv/bin/python probe_hand_units.py --read-only
```

**判据**：打出 22 个弧度值。**这一步只读，一根手指都不动** —— 它只是绑了手自己的 500 Hz
广播口（`:50000`），不从任何人手里拿走东西。

收不到 → 手没上电，或网口不对。

**这次 rollout 不需要 Manus 手套、不需要重定向器、不需要 Quest。** 那三条链是给遥操作的。

---

## 3 · 定手部单位 —— 只做一次，之后记住结果

⚠️ **会动一根手指**，并把手从 Pilot 的 `GLOVE` 源上抢过来（脚本退出时还回去）。

```bash
$V/.venv/bin/python probe_hand_units.py --i-am-watching
```

**判据**：`VERDICT: --hand-units rad`（或 `deg`）。**把这个值记下来**，之后每次 rollout 都要带。

为什么要测：数据集是弧度（`vr/collect/SCHEMA.md`，以及 `sharpa_command.decide_units` 2026-08-31
的实测抓包），而厂商自己的 `sharpa_wave_example.py` 用一张**角度**表驱动
`set_joint_position`。两者不可能同时描述同一个调用，差的是 **57.3 倍**，所以不猜，测。

探针命令的是 `0.30`：弧度解读下是 17°（在该关节 ~20° 量程内），角度解读下是 0.005 rad
（等于没动）。**两种解读下都伤不到硬件**，而读回值（一定是弧度）跟不跟得上就是答案。

> **Pilot 要不要开着？** 这条没有离机确认过。**先不开 Pilot 试** —— SDK 是直连
> `192.168.10.10` 的。如果 `get_all_device_sn()` 返回空列表，再开
> `/opt/sharpa-pilot/sharpa-pilot --no-sandbox` 重试。脚本 10 秒内给答案，不用猜。

---

## 4 · 臂 bring-up

> ⚠️ **两条臂都会使能电机并跑 `return_to_zero()`。先清空双臂工作空间。**

```bash
# 终端 4   （已跑过第 0 节的两段 prelude）
ros2 launch /home/yiming/openarm/openarm_track/revised/revised_bringup.launch.py \
    use_fake_hardware:=false right_can_interface:=can0 left_can_interface:=can1
```

`can0` = 右臂，`can1` = 左臂（挂着 1.44 kg 的 Sharpa 手）。

**判据**：日志 `Reached zero position (worst residual ...)`，残差很小（实测 0.017 / 0.023 rad）；
`ros2 control list_controllers` 里两个 JTC **active**、两个 forward **inactive**。

🚫 **这个进程从此不要 Ctrl-C。** 它会走 `on_deactivate → disable_all()` 断力矩，手臂直接砸下去。
有序退出见第 10 节。

---

## 5 · 走到 home

> ⚠️ 真机会动。

```bash
$V/scripts/both_to_home.sh
```

数值从 `arm_model.Q_HOME` 现查，不硬编码。

**不能在全零下垂位啮合**：joint4 的下限恰好是 0，臂正贴着限位且接近奇异，笛卡尔跟踪会严重退化。

---

## 6 · preflight（纯只读，什么都不启动）

```bash
$V/scripts/collect_up.sh preflight
```

这是**采集**用的检查表，对 rollout 有一项会红，**可以忽略**：

- `no Quest on adb` —— rollout 不用 Quest。

⚠️ **`nothing owns :50011 -- Sharpa Pilot is not running` 从第三轮起不能再忽略。** policy 不吃触觉，
但每次录制的 rollout 都要把触觉（F6 + 形变图）录进 `episode.hdf5`，Pilot 和 tap 必须在跑，见 §9b。
只有明确带 `--no-tactile`（台架测试）才可以不开。**`--backend dp` 默认不录触觉、不需要 tap 和 Pilot**
（要录就加 `--with-tactile`）；T-Rex 一直需要触觉。

**rollout 真正要看绿的四项**：

| 项 | 为什么它重要 |
|---|---|
| bring-up 在跑 | — |
| `can0/can1 live` | 看**包计数**，不看 `/joint_states` —— PCAN 掉线时话题照样 750 Hz 发布，只是数值是陈的 |
| **`motors ENABLED`（ERR nibble == 1）** | 最阴的一项。断电重上电后电机照样应答状态查询、effort 读出来 ≈0，**其它检查全绿而手臂是软的**。驱动的 `parse_motor_state_data` 只取 `data[1..7]`，永远不会告诉你这件事 |
| 两路相机的**彩色节点**存在且**没被占** | 节点号是 `camera.py:resolve_device()` 现场解析的（Gemini 336L 有 8 个节点，只有一个是彩色）。OrbbecViewer / realsense-viewer / 采集器开着都会占住 |

---

## 7 · 起 teleop 节点

🚫 **不要用 `collect_up.sh up`。** 它会拉起 tap / fanout / bridge / 采集器：采集器抢相机，
fanout 抢 UDP 口 `:9873`。rollout 只要那一个节点。

```bash
# 终端 5   （已跑过第 0 节的两段 prelude）
$V/scripts/preflight_udp.sh 9873          # 清掉占着这个口的僵尸进程

$V/.venv/bin/python -u $V/openarm_vr_teleop.py \
    --side left --udp-port 9873 --scale 1.0 --vmax 0.5 \
    --hand --telemetry --switch --grav-ff-mode effort --home-on-start
```

> ⚠️ **`--home-on-start` 会让臂在启动时自己动一次**（约 8 cm，`--home-vmax` 0.15 rad/s，一两秒）。
> 站在能按急停的地方再回车。

**为什么必须带 `--home-on-start`。** JTC 只写 position，`tau_ff` 没人写就是 0，保持力矩全靠
位置误差，所以发裸 `Q_HOME` 的臂会沉下去 —— 2026-09-08 实测 0.18 rad，**指尖比训练起始位姿低
7.8 cm**，正好从"悬在桌面上方"变成"低于桌面"，相机也就拍不到鸡蛋了。

> 同日起 `both_to_home.sh` 自己发 `Q_HOME + g(Q_HOME)/KP`，力矩上与 effort 模式等价，所以
> 这一沉本身应该已经没了。`--home-on-start` 仍然要带：它是那一步的**独立复核**（走同一条
> HOMING 路径、同样的门禁），而且偏置走的是位置路、依赖 KP 是真增益，没有 effort 那么硬。
> 起节点后照样按下面的判据看日志和实测位姿。

节点启动时锚点取的是 `q_meas`（`openarm_vr_teleop.py` 里 `_latch_hold(self.q_meas,
"controller switch complete")`），effort 模式下 `_ff()` 返回零，所以**节点会老老实实保持在
沉下去的位姿，不会自己升回来**。采集时操作员是按 VR 手柄的 home 键让节点走一次 HOMING 的；
rollout 没有手柄，`--home-on-start` 就是那个键的替代。

它走的是**同一条 HOMING 代码路径**（同样的 `--home-vmax`、软限位、关节限位钳制、超力矩跳闸、
engage-gap 门禁），目标同样是 `arm_model.Q_HOME`。注意目标是**指令**不是实测：38 条训练
episode 的实测起始位姿是 `[+0.028, -0.659, +0.246, +1.943, +0.243, -0.249, +0.114]`，与
`Q_HOME` 差最多 0.14 rad —— 那个差就是负载下的稳态跟踪误差。发同样的指令、挂同样的负载，
实测就会落回同一个地方，所以**不要**手工去凑那个实测值。

**和采集时只差 `--scale`**（采集是 `0.5`）。数据集里的 `action/wrist_pose_b` 记的是缩放
**之后**的目标，所以策略输出的位移已经在机器人空间了；节点还留在 0.5 会把每个平移砍一半。

真要留在 0.5，就给客户端传 `--node-scale 0.5`，翻译层会预先除掉。同样的动作，多一个能搞错的地方。

`preflight_udp.sh` 不能省：UDP 没有端口共享，一个残留的 `udp_fanout` 会**静默吃掉**整条腕部
数据流，节点起得干干净净、切了控制器、收到零帧。

**判据**（三条）：

```bash
ros2 control list_controllers | grep left_forward_position_controller     # 期望 active
```

节点日志里要看到这两行，顺序不能反：

```
HOMING to [+0.0277, -0.6647, +0.2176, +1.9850, +0.3141, -0.1337, -0.0242] at 0.15 rad/s
HOLD @ [...] (reached home)
```

看到 `--home-on-start gave up` 就是 engage-gap 门禁拦下了（臂离锚点超过 `--engage-gap`）。
它**只试一次**，不会反复重试，节点停在 HOLD。这时不要硬上：回 JTC 重跑第 5 步再起节点。

最后**用眼睛确认手已经在桌面上方**，鸡蛋在腕部相机视野里。同目录下有两张训练首帧的参考图，
是从训练集 `20260903_000158/ep_0000` 的第 0 帧（home、啮合前）抽的：

- `reference_start_scene.png` —— 场景相机：手掌心向下悬在桌面上方，鸡蛋在中间偏左，黑碗在右侧
- `reference_start_wrist.png` —— 腕部相机：鸡蛋在画面右侧，桌面占下方三分之二

**摆场景就对着这两张图摆**，比凭记忆准。腕部相机里看不到鸡蛋 = 还没到能 rollout 的状态。

> 此后臂归 forward controller 管，`both_to_home.sh` **不报错也不动**（轨迹发给了一个没在管臂的
> 控制器）。收臂只能用 `after_teleop_home.sh`。

---

## 8 · Dry run —— 每次都做

```bash
# 终端 6   （已跑过第 0 节的两段 prelude）
cd ~/openarm/openarm_track/rollout
$V/.venv/bin/python cli.py --prompt "pick up the egg"
```

**不带 `--enable-*` 就什么都不发。** 全速跑完整条链路 —— 相机、状态、推理、动作解码、腕部积分链、
安全包络 —— 只是不碰硬件。

`--prompt` 必须**逐字**是 LeRobot 数据集里的 task 字符串（配置用 `prompt_from_task=True`）：
`pick up the egg` / `pull tissue` / `extract card` / `push tablet`。

**判据，五条都要看**：

1. 两个相机的设备号和输出尺寸打出来了（head → `410x256`，wrist → `456x256`）
2. `all observation sources live`，四个 age 都远小于 0.25 s
3. `infer NNN ms` 在 267 ms 以内（`--infer-lead 8 / --fps 30` 的预算）
4. 每秒那行的腕部行程 —— **dry run 里就跑飞的，真跑一定跑飞**
5. 启动时打的 `envelope in force: wrist step ... mm / ... deg` 那一行，对照训练集单步上限
   （4.6 mm / 1.31°）过一眼。带 `--hand-norm-stats` 时这行现在**也**由 checkpoint 的
   `norm_stats.json` 决定腕部单步包络，不再只管手指 —— 目前两个已发布 checkpoint 打出来的都是
   18.26 mm / 5.24°，约为训练上限的 4 倍，属于预期内。一旦某个新 checkpoint 的统计量收得比这个
   紧很多，这行会先变，值得在真跑前多看一眼，而不是等真跑时莫名其妙被单步护栏拦下来。

dry run 不需要啮合：锚点用 `FK(q_meas)` 现算，那正是节点啮合时会取的同一个位姿。

---

## 9 · 真跑

```bash
$V/.venv/bin/python cli.py \
    --prompt "pick up the egg" \
    --hand-units rad \
    --hand-norm-stats $OPENPI/checkpoints/sharpa-pi05/egg_70ep_b64/15120/assets/local_repo/norm_stats.json \
    --enable-hand --enable-arm
```

- `--hand-units` 用第 3 步测出来的值。
- `--hand-norm-stats` **每次都带**。它把手指包络从固件的 ±π/2 收到策略实际训练过的逐关节范围
  —— 比如 `index_MCP_AA` 从 ±1.571 收到 `[-0.28, +0.20]`，八倍。不带的话客户端会退回固件钳位
  并打 warning。
- **录制现在默认开着**（只要带了 `--enable-arm` 或 `--enable-hand`），**不用再写 `--record`**。
  每次跑会自动建一个 `<backend>_<任务>_<日期-时间>/` 目录。**pi0.5 的全部存在
  `~/openarm/rollouts/pi0.5/` 下**，和 dp / T-Rex（仍在 `~/openarm/rollouts/` 下）分开，例如
  `~/openarm/rollouts/pi0.5/pi0.5_egg_T1.0_ode_20260914-153012/`。`--record-dir <目录>` 可以改放别处。里面留这几样东西：

  | 文件 | 是什么 |
  |---|---|
  | `pi0.5_egg_T1.0_ode_20260914-153012.npz`（与目录同名） | 逐步的 state / 指令 / 实发指令，啮合锚点与 `--node-scale`（腕部轨迹靠它们还原到基坐标）。小、写得快 |
  | `head.jpgs` `wrist.jpgs` | 策略**实际看到**的画面，每个控制步一帧，跑的过程中就在往下写（约 14 MB/分钟）|
  | `plots.png` | EE 位姿（指令 vs FK 实测 vs 两者之差）、7 个臂关节、22 个手关节、手部指令残差热图 |

  **视频是事后生成的，rollout 期间不编码**：

  ```bash
  $V/.venv/bin/python <rollout>/make_video.py <运行目录>            # head.mp4 + wrist.mp4
  $V/.venv/bin/python <rollout>/make_video.py <运行目录> --fps 5    # 放慢看细节
  $V/.venv/bin/python <rollout>/make_video.py <运行目录> --cam wrist
  ```

  第 N 帧就是第 N 步 —— 可以停在某一帧再去查那一步的数字。`mpv` / `vlc` / 浏览器都能放。

  `--no-record` 全关，`--no-frames` 只留数组（就没有视频可做了），`--no-plot` 跳过出图。
  dry run 默认**不**录（通常是连着调第五次），要录就显式给 `--record <目录>`；
  给 `--record <路径>/<名字>.npz` 则建 `<名字>/<名字>.npz`，用自己起的名字代替自动名。
  图也可以事后重画：`<rig>/.venv/bin/python plot_rollout.py <运行目录>`
  —— 必须用 rig 的解释器，它是这台机器上唯一同时有 pinocchio（做 FK）和 matplotlib 的。

运行中：**`p`** 暂停（臂保持在最后一个目标，啮合不断；恢复时会丢掉暂停前的推理重新算一个当前的），
**`q`** 退出。

🚫 **要停就按 `p` 或 `q`，不要 Ctrl-C 终端 4。**

---

## 9b · 第三轮：π0.5 多候选 + 触觉录制 × 30（handoff-round3 §3）

和第 9 步是同一条链路，多出来的只有下面这些。**每批 30 次：同一个 checkpoint（15120）、同一个 N、同一个 T。**

批次顺序：**先 T=1.0 × 30**，再 T=1.5 × 30、T=2.0 × 30（只改 `--noise-temperature`，其余不动）。
采样器一律用默认的确定性 ODE（`sde_eta=0`，openpi `sharpa-rollout-sde` 分支的默认值；客户端不发 `sde_eta`）。
鸡蛋只放在训练数据覆盖的小范围内 —— π0.5 训练集里鸡蛋位置分布本来就窄，先不往外扩。

**① server 带候选默认值起**（终端 1，替换第 1 步的命令）：

```bash
scripts/sharpa_serve.sh ./checkpoints/sharpa-pi05/egg_70ep_b64/15120 sharpa_egg \
    --num-candidates 4 --noise-temperature 1.0
```

**② 用实际的 N 测延迟**（终端 2）。第一次调用是这个 N 的 JIT 编译（几十秒），不计入：

```bash
$V/.venv/bin/python ping_policy.py --n 40 --num-candidates 4
```

判据：`OK -- p95 inference fits in 267 ms`，没有 `CHUNK PROBLEMS`。**N 定下来之后 30 次都不改。**

本机（RTX 5090 Laptop）2026-09-14 实测，走 websocket：N=4 **p50 202 / p95 247 ms**，`--infer-lead 8` 放得下。
这靠的是服务端的**共享前缀**采样（`candidate_policy.py`：图像+prompt 的 VLM 前缀只算一次，KV cache 复制给 N 个候选，
只有去噪按 N 路跑）。openpi 没有这个改动时 N=4 是 p50 571 / p95 646 ms，放不下 —— ping 出来五六百毫秒就先检查
openpi 是不是最新的。

真跑时偶尔一次推理超预算只会打一行 `inference was not ready at the chunk boundary`，那个边界晚一点接上；
如果这行**频繁**出现，改用 `--infer-lead 9`（300 ms 预算，`--chunk-steps` 仍是 15），并在 notes 里记下。

**③ Pilot + 触觉 tap**（新开终端 7，无 ROS）。Pilot 开着、**触觉/形变视图打开**（设备只在有人请求时才算形变图）：

```bash
sudo $V/.venv/bin/python $V/collect/sharpa_tap.py --serve /tmp/sharpa_tap.sock --stats
```

判据：`collect_up.sh preflight` 里 `Sharpa Pilot owns :50011` 为绿，`/tmp/sharpa_tap.sock` 存在。

**④ 真跑**（终端 6）：

```bash
$V/.venv/bin/python cli.py \
    --prompt "pick up the egg" \
    --hand-units rad \
    --hand-norm-stats $OPENPI/checkpoints/sharpa-pi05/egg_70ep_b64/15120/assets/local_repo/norm_stats.json \
    --enable-hand --enable-arm \
    --num-candidates 4 --noise-temperature 1.0
```

上面这条要先 `cd ~/openarm/openarm_track/rollout`，不然会报 `can't open file '.../cli.py'`
（收完手常常停在 `vr/` 下）。**不想管当前目录，就用绝对路径**，在哪个目录都能跑：

```bash
$V/.venv/bin/python ~/openarm/openarm_track/rollout/cli.py \
    --prompt "pick up the egg" \
    --hand-units rad \
    --hand-norm-stats $OPENPI/checkpoints/sharpa-pi05/egg_70ep_b64/15120/assets/local_repo/norm_stats.json \
    --enable-hand --enable-arm \
    --num-candidates 4 --noise-temperature 1.0
```

换批次时只把 `--noise-temperature` 改成 1.5 / 2.0（SDE 采样器再加 `--sde-eta 0.5` 之类，默认 0 = ODE）。

**运行目录按采样设置命名**：`~/openarm/rollouts/pi0.5/pi0.5_egg_T<温度>_<ode|sde<eta>>_<日期-时间>/`，例如
`pi0.5_egg_T1.0_ode_20260914-212616`、`pi0.5_egg_T1.5_ode_...`、`pi0.5_egg_T2.0_sde0.5_...`，npz 与目录同名，
各批次在盘上一眼分得开（`ls ~/openarm/rollouts/pi0.5 | grep T1.5_ode`）。名字不是凭参数猜的：客户端核对 server
回报的实际 T / sde_eta，和请求不一致就停下（例如 server 用 `--sde-eta` 起的，客户端却没带），不会让一批数据挂错名字。
`--noise-temperature` / `--sde-eta` 必须和 `--num-candidates` 一起给。2026-09-14 之前录的 9 条已按文件里记录的
T=1.0、server 默认 sde_eta=0 改名为 `..._T1.0_ode_...`（`index.jsonl` 和 hdf5 里的 `rollout.run_dir` 同步改了）。客户端每次请求都带 T，会覆盖 server 的默认值。
`--selection-seed` 不用给：每条 episode 自动取一个新 seed，打在日志里并写进文件。

**⑤ 一次运行的流程**（客户端自己按这个顺序走，操作员只管看日志和按键）：

| 阶段 | 日志 | 操作员 |
|---|---|---|
| 触觉检查 | 5 根手指都在推数据，否则直接退出并说原因 | — |
| 预热 | `warming the server up with N=4` → `warm-up done` | 等（换 N 后第一次几十秒） |
| **首段自由空间** | `FREE SPACE: hand open, touching NOTHING, for 2.0 s` | 手张开，什么都别碰 |
| rollout | 同第 9 步；`p` 暂停、`q` 结束 | **失败也跑完整条，不要中途放弃重来** |
| **结束** | 按 **`q`**（不要 Ctrl-C） | 觉得动作做完了就按一下 `q` |
| **尾段自由空间** | `recording 2.0 s of tail` | **什么都不用做**，没人碰机器人。手上还有受力（蛋没放下）会打 `tail: fingers still loaded` 并记进文件 |
| **当场标注** | 菜单 `1 success … 6 other` | 选一个；`other` 要写一句话 |

**每次结束后的复原顺序**（server、tap、bring-up 一直开着不动）：

1. 终端 6 标注完，客户端退出。臂由 teleop 节点保持，手保持最后的指令、Control Source 留在 SDK。
2. `$V/scripts/after_teleop_home.sh`：停节点 → 切 JTC → 臂回 home → 手回 home → **手交还 IDLE**。
   如果它说 `没有需要收的臂`（节点已经自己退出、臂已回到 JTC），用 **`after_teleop_home.sh left`** 显式指定。
   如果终态打印 **`臂已到位，但收手失败`**：臂已经在 home，只是手没收（例如偶发的
   `Failure to connect to server for current_coeff`，SDK 这一条指令没送到手上，手指没动）。
   **不用重跑整个脚本**，单独重跑收手：`$V/.venv/bin/python $V/hand_home.py --send`（绝对路径，在哪个目录都能跑），
   结尾看到 `control source -> IDLE` 即可。连续失败再查手的网线 / Pilot。
3. 摆下一个鸡蛋位置。
4. 终端 5 **重起 teleop 节点**（第 7 步，`preflight_udp.sh 9873` + 带 `--home-on-start` 的那条）—— 上一步把它停了。
5. 终端 6 跑下一次。

Pilot 底部状态栏的 Control Source 保持 **IDLE / 空闲**（左边栏的 Sliders / Glove 只是界面，不是 Control Source）。
2026-09-14 起客户端和收手脚本都先切 SDK 再设参数，即使停在 APP 也能接管；不要选 GLOVE。

安全包络触发、`stale observation` 停止也会走到尾段和标注菜单 —— **照常标注，不要删**；
停止原因和步号、每次 `p` 暂停/恢复的步号都自动记在事件里。标注时 Ctrl-C 会存成 `unlabeled`，不会丢数据。

free 段里如果有手指 |F| 波动超过 0.5 N，会打 `something touched the hand` 的 warning —— 这条的零点不可信，
notes 里记一句。

**⑥ 每次运行留下的东西**（`~/openarm/rollouts/pi0.5/pi0.5_egg_T1.0_ode_<日期-时间>/`）：

| 文件 | 内容 |
|---|---|
| `episode.hdf5` | **rt-v1 格式**（和采集的 episode 同一个写入器、同一套 schema），传感器线程**固定 30 Hz**，不受控制频率影响：`obs/` `action/` `teleop/` `hand/`（位置/速度/力矩）`action/hand_joint_pos`（客户端发出的手指令）`tactile/f6` `tactile/deform` `tactile/deform_valid`；每个 chunk 边界的 `cand/obs_step` `cand/actions (K,N,30,28)` `cand/noise (K,N,30,32)` `cand/chosen` `cand/exec_steps`（[起, 止) 控制步）`cand/infer_ms`；attrs：`outcome` `events` `tactile_zero_offset` `rollout`（N、T、seed、checkpoint、各种 cadence）`git_openpi` `git_rig` `server_metadata` |
| `pi0.5_egg_T1.0_ode_<日期-时间>.npz` `head.jpgs` `wrist.jpgs` `plots.png` | 同第 9 步；npz 里也有同名的 `cand/*` 和 `events_json` |
| `head_first.jpg` `wrist_first.jpg` | 启动时（啮合之前）的场景首帧 |
| `../index.jsonl` | 每条一行：路径、帧数、`success` |

rt-v1 文件和采集的差别：**没有 MKV**，行时钟是传感器线程；`time/frame_index` 是该行时刻最新的**控制步**
（= `head.jpgs` 第 k 帧 = `flight.npz` 第 k 行，开始前为 -1）。`attrs["row_clock"]` 写着这件事。

**⑦ 先录 1–2 条传回去确认格式**，再跑剩下的。自查：

```bash
$V/.venv/bin/python -c "
import h5py, json, glob
p = sorted(glob.glob('/home/yiming/openarm/rollouts/pi0.5/*/episode.hdf5'))[-1]
f = h5py.File(p, 'r'); a = f.attrs
print(p, a['num_frames'], 'rows', a['outcome'])
print('tactile valid', f['tactile/valid'][()].all(1).mean(), 'deform_live', json.loads(a['tactile_final'])['deform_live'])
print('cand', f['cand/actions'].shape, 'chosen', f['cand/chosen'][()], 'seed', f['cand'].attrs['selection_seed'])
print('events', [e['kind'] for e in json.loads(a['events'])])
"
```

判据：`tactile valid` 接近 1.0、`deform_live True`、`cand` 是 `(K, 4, 30, 28)`、`chosen` 不全是 0、events 以
`free_head_begin` 开头、以 `free_tail_end` 结尾。

---

## 10 · 收（顺序和开机相反）

```bash
# 1. 客户端：按 q
# 2. 收臂 —— ⚠️ 真机会动，脚本自带确认和 CAN 活性门禁
$V/scripts/after_teleop_home.sh
# 3. 如果第 3 步开了 Pilot，GUI 里 Control Source 切回 NONE
# 4. 有序停机：切 JTC → 慢速回零 → 才停 bring-up
$V/scripts/shutdown_all.sh
```

`shutdown_all.sh` 有**硬闸门**：任一条臂的回零没报 SUCCESSFUL 就停在那里、不停 bring-up，让两条臂
继续保持力矩。**断力矩只能在全零下垂位做** —— 左臂挂着 1.44 kg 的手，从抬起姿态断电后果更严重。

只是想歇一会儿再跑一轮：链路都还在，直接重跑第 9 步。

---

## 安全网 —— 知道它们在哪，以及各自看的是什么

客户端有**三个安全包络**，全都在 `robot_openarm.apply` 里，dry run 也全跑一遍，只是不碰硬件：

| 包络 | 触发 | 日志 | 看的是什么 |
|---|---|---|---|
| 腕部·单步 | 单个动作腕部位移 > 20 mm 或转角 > 5.7°（带 `--hand-norm-stats` 时改由 checkpoint 的统计量决定，见 §8 判据 5） | `SAFETY STOP: single action moves the wrist ...` | 训练数据里单步上限是 4.6 mm / 1.3°。这一步既没被积分也没被发出去，rollout 停止，节点保持在**最后一个通过检查的目标** |
| 腕部·累积 | 累计离开啮合点 > 45 cm 或 1.8 rad | `SAFETY STOP: ... left the envelope` | 逐步都在分布内的慢漂移只有这层看得见。rollout 停止，节点保持在最后一个通过检查的目标 |
| 手部 | 手指目标超出逐关节范围，或单步变化超过限速 | 不停 rollout —— 静默钳位/限速到范围内；启动时打的 `envelope in force: ... hand ... rad` 就是这层当前的边界 | 固件钳位是 ±π/2，`--hand-norm-stats` 把它收到 checkpoint 训练过的逐关节范围。这层不是"触发即停"，是每一步都先过一遍 |

前两层互补，都需要：单步护栏看不见 4 mm/step 的漂移（每个动作都正常），而累积包络要一百步、三秒才
发火。手部这层跟腕部两层各自独立生效，谁都不能替代谁。

除了这三个包络，还有一个独立的停止触发——不是安全包络，是对**观测**本身的门禁：任一观测源超过
0.25 s 不更新（`stale observation (max_age=0.25s): {...}`），整条循环直接停，不拿冻结的画面继续
喂策略。它和上面三层不是一回事：上面三层管的是"要发出去的命令是否在分布内"，这层管的是"喂给策略
的输入是否新鲜"。

再往下是 teleop 节点自己的 8 层（软限位包络、逐关节限速、偏差跳闸、RAMP_BACK、重力前馈），
rollout 客户端一行没碰。

---

## 出问题时

| 现象 | 多半是 |
|---|---|
| `no frame within 10s from /dev/videoN` | 相机被别的进程占着。关掉 OrbbecViewer / realsense-viewer / 采集器，`collect_up.sh preflight` 会指出 pid |
| `ModuleNotFoundError: No module named 'sharpa'` | SDK 不在 `/opt/sharpa-wave-sdk/python`。`rig.py` 默认加这个路径，装在别处就设 `SHARPA_SDK_ROOT`。**不需要 `LD_LIBRARY_PATH`**，扩展模块自带 `RUNPATH` |
| `the policy client needs: websockets>=12.0` | rig venv 继承了系统的 10.4，没有 `.sync`。按报错里那条命令装。**别顺手升 msgpack** —— `python-can 4.3.1` 钉着 `msgpack~=1.0.0`，而系统的 1.0.3 完全够用 |
| `the node did not engage within 5s; it is in HOLD` | 节点拒绝啮合：臂离它的 hold 锚点超过 `--engage-gap`（0.15 rad）。看终端 5 的日志。刚 home 完就起节点通常不会有这问题 |
| `no telemetry on /openarm_teleop/left/state` | 节点没带 `--telemetry`，或者这个进程没在 spin rclpy |
| `inference was not ready at the chunk boundary` | 推理超预算。`ping_policy.py` 会算出该用多大的 `--infer-lead`。机器上有别的东西在吃 GPU 也会这样 |
| `stale observation` 且列的是 `joint_states` | bring-up 掉了，或者 CAN 冻了。**先看 `can0/can1` 的包计数**，`/joint_states` 会继续发陈数据 |
| 手一动不动，臂正常 | Pilot 的 Control Source 还在 `GLOVE`；或者 `--hand-units` 反了（弧度值被当角度读 ≈ 命令 0） |
| 手指瞬间冲到极限 | `--hand-units` 反了（角度值被当弧度读，差 57.3 倍）。立刻 `q`，重跑第 3 步 |
| 臂朝错误方向走，但幅度合理 | 四元数 wxyz/xyzw 搞反，或 `--node-scale` 和节点的 `--scale` 对不上。前者有测试钉着（`actions_test.py`），后者查终端 5 的启动参数 |
| 每个平移都只有预期的一半 | 节点还是 `--scale 0.5` 起的 |
| `after_teleop_home.sh` 报 `controller_manager 不在` / `这个终端列不出控制器` | **多半不是 bring-up 挂了，是这个终端连不上 ROS。** 先确认：`pgrep -af ros2_control_node` 有进程、`can1` 的 `rx_packets` 每秒涨几千 → bring-up 活着、臂有力矩（2026-09-14 就是这样）。然后在这个终端：`echo $ROS_DOMAIN_ID` 必须是 0（设成别的值脚本不会覆盖）；`ros2 daemon stop && ros2 daemon start`；还不行就新开终端 `conda deactivate` + source ROS 两行 + `export ROS_DOMAIN_ID=0`。**不要为这条去重启 bring-up 或跑 `shutdown_all.sh`** |
| rollout 标注完、退出时 `Segmentation fault` / `Aborted (core dumped)` / `terminate called without an active exception` | 发生在数据**存完之后**的收尾（手 SDK、GStreamer、rclpy 的原生线程退出顺序），臂和手早已释放。先确认数据：运行目录里有 `episode.hdf5`（不是 `.part`）和同名 `.npz`，用 §9b ⑦ 自查。只缺 `plots.png` 就补画：`$V/.venv/bin/python ~/openarm/openarm_track/rollout/plot_rollout.py <运行目录>`。2026-09-14 起崩溃时会打印各 Python 线程的位置（faulthandler），出图也挪到了子进程 —— 再崩把终端最后一屏贴出来 |
| 运行目录里只有 `episode.hdf5.part`、没有 npz | 那是 2026-09-14 修掉的 `finish_rt` NameError（那之前的一条）。`.part` 可以救成 `episode.hdf5`（截到最后一次落盘的行，缺 `cand/*`） |
| `no tactile tap at /tmp/sharpa_tap.sock`，但 `pgrep -af sharpa_tap` 有进程 | 长时间运行的 tap 的 socket 文件被 `/tmp` 清理删了，进程活着但谁也连不上。`sudo pkill -INT -f "sharpa_tap.py --serve"`，再按 §9b ③ 重起 |
| 收手失败：`收手失败（退出码 1）` / `failed to set current coeff: Failure to connect to server` / `failed to set speed coeff: Operation not allowed` | **臂不受影响**（收手在臂走到位之后才做，手停在原地、没收到位置指令）。单独重跑收手即可，不用重跑 `after_teleop_home.sh`：`$V/.venv/bin/python $V/hand_home.py --send`（先不带 `--send` 只规划、不动手）。结尾看到 `control source -> IDLE` 就收好了。`Failure to connect to server`：SDK 这一条指令没送到手上，偶发，重跑一般就过；连续失败先看手有没有电、网线 `cat /sys/class/net/enp129s0/operstate` 是不是 `up`。`Operation not allowed`：别的控制源占着手（Pilot 的 APP），2026-09-14 起脚本先切 SDK 再设参数，应该不再出现；还出现就在 Pilot 底部状态栏切到 IDLE 再重跑 |
| Pilot 点了 reset 后 Control Source 变成 `APP` | Pilot 自己的行为：Sliders 页的 reset 第一步就是切 MASTER（=APP）。rollout 前后都不要点 reset；客户端和收手脚本 2026-09-14 起先切 SDK 再设参数，APP 下也能接管 |

---

## 附：怎么选 checkpoint

Hub 上有两个：`egg_70ep_b64/8640`（40 epoch）和 `egg_70ep_b64/15120`（70 epoch）。

> ⚠️ 模型卡把前者写成 `08640`，**Hub 上是不补零的 `8640`**。`hf download --include` 匹配不到
> 任何文件时**退出码 0、一个字节不下**。数文件，别看退出码。

**没有验证集。** 38 条 episode 全部进了训练，一条没留。所以模型卡上的
`loss 0.0382 → 0.0180`、`finger MAE 0.0042 rad`、`wrist pos err 2.50 mm` 全是在**它自己训过的
数据**上算的 —— 衡量的是背得多熟，不是换个场景还行不行。而且 38 条 demo 训 70 epoch 是故意训进
过拟合区的（先把 pipeline 端到端打通，质量到机器人上再判）。

平常靠 val loss 挑 checkpoint 的路这里不存在。`8640` 和 `15120` 谁更好，**没有任何离线数字能
告诉你**，两种可能都真实：`15120` 拟合更紧 → 复现示范更准；或者它把那 38 次里鸡蛋的具体位置背
下来了 → 蛋挪 5 cm 就散架，反而 `8640` 稳。

所以只能真机 A/B：

1. **先别比。** 先确认至少有一个能动：场景摆成和采集时尽量一致（鸡蛋位置、光照、桌面），
   用 `15120` 跑 5 次。全崩就不是选 checkpoint 的问题，回去查观测对齐。
2. **再比。** 每个各 10 次，**交替跑**（A B A B…），不要先跑完 A 再跑 B —— 光照会变、手会热、
   摆位手法会漂，连着跑会把这些全算到后跑的那个头上。
3. 每次记：成功/失败、失败模式（没抓到 / 抓到掉了 / 手往错方向去 / 撞到东西）、鸡蛋起始位置。
   全程 `--record`。
4. **统计上诚实**：各 10 次只能看出大差距（8/10 vs 3/10 算数）。5/10 vs 6/10 是噪声。
5. 想知道是不是过拟合，故意把蛋放到示范里没出现过的位置。**分布内更好、分布外更差**是过拟合的
   典型特征。

比成功率更早给信号的东西：把 `--record` 的 npz 和训练 episode 的腕部轨迹、手指曲线摆一起看。
策略是在做示范里那个动作，还是在做一个看着像但时序不对的动作，这里先看得出来。
