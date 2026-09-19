# 采集数据 · 从零到一次跑通

单左臂 + Sharpa 手 + Manus 手套 + Quest,按顺序往下敲,每步都有判据。
**判据不过就停在那一步排查,不要往下走** —— 否则问题会一直拖到最后 `up --task` 才炸。

开五个终端。标 `常驻` 的不要关、不要 Ctrl-C。

| 终端 | 跑什么 |
|---|---|
| T1 | SharpaManusClient `常驻` |
| T2 | 重定向器 `常驻` |
| T3 | Sharpa Pilot (GUI) `常驻` |
| T4 | 臂 bring-up `常驻` |
| T5 | 采集(独占键盘) |

---

## 0 · 网络

任意终端:

```bash
ip -br addr show enp129s0        # 期望 192.168.10.240/24
ping -c3 192.168.10.10           # 期望 0 丢包
```

网卡是常驻的 NetworkManager profile `SharpaWave`,正常不用动。

---

## 1 · CAN

> **PCAN 必须直插机身 USB 口,不要走 hub。** 这不是建议。实测插在 Genesys Logic hub
> (`05e3:0610`)上时每分钟掉线一次,设备号从 `Dev 020` 一路涨到 `027` 然后彻底消失;
> 改直插机身后 14.5 分钟零掉线。掉线的后果见文末"坑 3",很难查且有危险。

```bash
sudo ip link set can0 down 2>/dev/null
sudo ip link set can0 type can bitrate 1000000 dbitrate 5000000 fd on
sudo ip link set can0 up

sudo ip link set can1 down 2>/dev/null
sudo ip link set can1 type can bitrate 1000000 dbitrate 5000000 fd on
sudo ip link set can1 up
```

`can0` = 右臂,`can1` = 左臂(挂着 1.44 kg 的手)。

如果报 **`Cannot find device "can0"`**,就是盒子根本不在 USB 上,直接跳到坑 3。

**判据 A — 配置对不对:**

```bash
ip -details link show can1 | sed -n '1,4p'
```

四样都要对:

```
state UP ... LOWER_UP                       链路起来了
can <FD> state ERROR-ACTIVE                 正常工作态(不是错误!)
berr-counter tx 0 rx 0                      零错误
bitrate 1000000 sample-point 0.750          配置生效
```

> `ERROR-ACTIVE` 这个名字唬人,但它是 CAN 健康节点的**正常状态**。
> 真正的坏状态是 `BUS-OFF`、`STOPPED`、`ERROR-PASSIVE`。

**判据 B — 稳不稳,必须等着验:**

先记一条基线:

```bash
date +%H:%M:%S; ip -o link show | grep -E 'can[01]' | awk -F: '{print $1, $2}'; lsusb | grep 0c72
```

**等至少 5 分钟**(趁这段时间去起手套链路),再跑一遍同样的命令。
接口 index 和 `Device NNN` **必须一个数字都没变**。

编号在涨 = 盒子还在反复掉线,bitrate 会被悄悄清掉、接口退回 DOWN。
**这种状态下绝对不要起 bring-up**(坑 3)。先把 USB 插稳再继续。

---

## 2 · Sharpa 手(先手后臂)

手挂在左臂末端,所以手的链路要在臂之前起,关机时反过来。

### 2.1 Manus 手套

打开 **Manus Core**,确认手套已连接、已校准。手套戴好,**Quest 左手柄绑在手套上**。

### 2.2 SharpaManusClient — T1 `常驻`

```bash
cd /home/yiming/sim/openarm-sharpa-sim/sharpa-manus-sdk/client
script -qfc ./SharpaManusClient.out /dev/null
```

`script -qfc` 是必须的,这程序要一个 PTY,直接 nohup 不出数据。

**判据**:打印 `glove: Left is published` 再进下一步。

### 2.3 重定向器 — T2 `常驻`

```bash
source /home/yiming/miniforge3/etc/profile.d/conda.sh
conda activate sharpa
export PYTHONPATH=/opt/sharpa-wave-sdk/python:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/sharpa-wave-sdk/lib:$LD_LIBRARY_PATH
cd /home/yiming/sim/openarm-sharpa-sim/sharpa-manus-sdk/retargeting_alg_release_V4.0

python -u retargeting_manus_demo_multiprocess.py -wave -filter_alpha 0.2
```

**判据**:持续打印每只手 22 个关节角(rad)。

### 2.4 Pilot — T3 `常驻`

```bash
/opt/sharpa-pilot/sharpa-pilot --no-sandbox
```

GUI 里把 **Control Source 设成 `GLOVE`**。不做这步,重定向器算得再对手也不动。

**判据**:动手指,实体手跟着动。

### 2.5 确认手的指令真的上了线

```bash
ss -tinp | grep -A1 '127.0.0.1:2044'
```

**判据**:`bytes_received` 在涨,`lastrcv` 是个小数字。两条都冻住 = 手套流断了(见文末"坑 2")。

---

## 3 · Quest

USB 接头显,头显里同意 adb 授权:

```bash
adb devices
```

**判据**:状态是 `device`,不是 `unauthorized`。

> 站位或朝向变了才需要重做轴标定,平时跳过。当前值 `-x:-z:-y`。

---

## 4 · 臂 bring-up — T4 `常驻`

> ⚠️ **真机会动**:两条臂都会使能电机并跑 `return_to_zero()`。**先清空双臂工作空间。**

```bash
source /opt/ros/jazzy/setup.bash
source /home/yiming/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=0

ros2 launch /home/yiming/openarm/openarm_track/revised/revised_bringup.launch.py \
    use_fake_hardware:=false right_can_interface:=can0 left_can_interface:=can1
```

**判据**:日志出现 `Reached zero position (worst residual ...)`,残差很小(实测 0.017 / 0.023 rad)。
另开终端确认控制器:

```bash
ros2 control list_controllers
# 两个 *_joint_trajectory_controller  active
# 两个 *_forward_position_controller  inactive
```

🚫 **这个进程从此不要 Ctrl-C。** 它会走 `on_deactivate → disable_all()` 断力矩,手臂直接砸下去。
有序退出见第 8 节。

---

## 5 · 走到 home

> ⚠️ 真机会动。

```bash
/home/yiming/openarm/openarm_track/vr/scripts/both_to_home.sh
```

不给参数默认只走左臂;`right` / `both` 选别的。数值从 `arm_model.Q_HOME` 现查,不硬编码。

**判据**:脚本自己会先把预期的 `ss_err` 打出来。**`ss_err` 不是 0 是正常的** ——
下发的是 `Q_HOME + g(Q_HOME)/KP`,稳态误差会收敛到那个偏置本身。臂若冲过了 home,
加 `--grav-ff 0.63`。

---

## 6 · 采集 — T5

端口先清干净(每次起遥操作前都要跑,没有例外):

```bash
/home/yiming/openarm/openarm_track/vr/scripts/preflight_udp.sh
```

**判据**:退出码 0。

然后就是这一条,它把 tap / fanout / bridge / teleop 全起起来,preflight 全绿了才会往下走,
最后进采集界面:

```bash
V=/home/yiming/openarm/openarm_track/vr
$V/scripts/collect_up.sh up --task "grab a sandwhich bag out of the box"
```

`--task` 那句话会**存进每一帧**,照实写。

采集界面的按键(这个终端独占键盘):

| Quest 右手柄 | 键盘 | 动作 |
|---|---|---|
| A | `空格` | 开始 / 停止并保存一个 episode |
| B | `d` | 丢弃正在录的这个 |
| — | `u` | 把**上一个已保存**的标记为丢弃 |
| — | `s` | 打印状态 |
| — | `q` | 结束本次 session |
| 右扳机 | — | 遥操作 deadman(按住才动,跟上面几个键无关) |

> 第一个 episode 存完之后,看它下面那行 `deform_live=`。
> **启动横幅里那个 deform_live 不作数**(那是触觉第一帧还没到时拍的快照)。
> 保存行上是 `false` 才是真有问题 —— 去 Pilot 里打开触觉视图,然后重开一个 session。

**再采一轮**(栈还开着,不用重起):

```bash
$V/scripts/collect_up.sh collect --task "put the sandwhich bag back in the box"
```
其他任务的采集命令：
```bash
$V/scripts/collect_up.sh collect --task "screw lightbulb"
```


数据落在 `~/openarm/data/sessions`。

若报头显未连的错，请运行：
```bash
adb shell monkey -p com.rail.oculus.teleop -c android.intent.category.LAUNCHER 1
```
来唤醒头显。

其它:

```bash
$V/scripts/collect_up.sh status                    # 什么在跑
$V/scripts/collect_up.sh logs bridge               # 看某个后台件的日志
$V/scripts/collect_up.sh preflight                 # 只检查,什么都不起
```

---

## 7 · 收工回 home

> ⚠️ 真机会动。只收回 home,**不断力矩**,收完可以直接重新啮合接着采。

```bash
/home/yiming/openarm/openarm_track/vr/scripts/after_teleop_home.sh
```

不给参数会自动检测哪条臂还在 forward controller 下。

---

## 8 · 关机

**先手后臂**:Pilot 里把 Control Source 设成 **NONE**(或 APP),**然后**才跑:

```bash
/home/yiming/openarm/openarm_track/vr/scripts/shutdown_all.sh
```

它会:停遥操作 → 切回 JTC → 两条臂各 8 秒斜坡到零 → 确认都 SUCCESSFUL 了才停 bring-up 断力矩。

**判据**:最后打印 `Safe to power the robot off now.`

如果它 `ABORTING BEFORE POWER-DOWN`,那是硬门禁在保护你 —— 说明某条臂没真的走到零,
bring-up 会继续留着让臂保持力矩。**别绕过它**,先查为什么斜坡没成(多半是坑 3)。

---

## 踩过的坑

### 坑 1 · 手亮黄灯,Pilot 报 `0x703 - tactile flash read error`

上电自检时读指尖传感器板载 flash(ftsid / 曝光 / 标定矩阵 / 弹性体版本)失败。
**触觉流本身通常还是好的** —— 看 Pilot 日志里有没有 `Tactile is ready`,有就能正常采。
彻底断电重上电一般能清掉。固定复现且能定位到某一根手指,就是那个指尖的 I2C/flash 坏了,找 Sharpa。

### 坑 2 · preflight 报 `NO hand command on :50020`

手套链路的 socket 还 ESTAB、进程还活着、CPU 还在烧,但**字节计数器冻住了**。
手套休眠或 dongle 掉链路,重定向器没输入,自然不广播。

```bash
ss -tinp | grep -A1 '127.0.0.1:2044'      # bytes_received 必须在涨
```

先动动手指唤醒手套。还是冻着就重起 T1/T2 两个终端的进程(2.2 / 2.3)。

### 坑 3 · CAN 的三段病程

同一个根因(PCAN 从 USB 掉线),按严重程度会表现成三种完全不同的样子。
**先对症状,再往下查。**

#### 症状 A — ROS 全绿,臂纹丝不动

最阴的一个。控制器切换成功、轨迹跑满 8 秒、`joint_states` 一直有读数、日志干干净净,
但电机一动不动。签名是 `track_test` 报告里 **`start` 和 `final` 完全相等、`ss_err` 恰好等于起始角**:

```
openarm_left_joint4    start 1.9942   target 0.0000   final 1.9942   ss_err -1.9942
```

原因:盒子掉线重连过,内核把 can0/can1 重建成了**新的 netdev**,
bring-up 手里那个 socket 还指着已经消失的旧接口。指令写进去了,但没人收。

**重配 bitrate 不够 —— bring-up 必须重起**,因为要重建的是它进程内的 socket。

#### 症状 B — `Cannot find device "can0"`

盒子已经彻底从 USB 上掉下来,连 netdev 都没了。这时候任何 `ip link` 命令都没意义,
**只能先解决物理连接**。

#### 症状 C — `after_teleop_home.sh` 报 `CAN 没在收数据`

```
left   can1  rx +0 pkt/s   <<< 冻结!
!!! 中止:CAN 没在收数据,/joint_states 的值是陈的,规划出来的轨迹不可信。
```

这是脚本的硬门禁在保护你,**别绕过它**。它说的恢复顺序(先杀 bring-up、再拉 CAN、
最后重启 bring-up)是对的,但前提是 CAN 设备还存在 —— 如果是症状 B,先插线。

#### 定位命令

```bash
lsusb | grep 0c72                  # 不在 = 症状 B;Device NNN 在涨 = 还在反复掉
ip -br link show can0 can1         # index 变了 = 盒子重新枚举过
ip -s link show can1 | tail -4     # RX/TX 常年 0 = 根本没流量
lsusb -t | grep -B4 peak_usb       # 看它挂在 root hub 下还是某个 hub 下
```

`lsusb -t` 的缩进能看出层级。**挂在 hub 下面就是病根**:

```
|__ Port 001: Dev 018, Class=Hub, Driver=hub/4p, 480M    ← Genesys hub
    |__ Port 002: Dev 021 ... Manus dongle
    |__ Port 004: Dev 027 ... peak_usb                   ← PCAN 缩进在 hub 里 = 有病
```

#### 解决 — 顺序不能错

> ⚠️ **CAN 永远在 bring-up 之前。** 唯一排在 CAN 前面的 bring-up 动作是**杀掉**它,不是启动它。
> 反过来做的话,bring-up 启动时 `can0` 还没 UP,它打不开设备就直接退了。

```
杀 bring-up  →  插稳 USB  →  配 CAN  →  验稳定  →  起 bring-up  →  both_to_home
   ^^^^                                              ^^^^
   清掉持有死 socket 的旧进程                          这才是启动
```

**为什么必须先杀**:旧 bring-up 手里那个 CAN socket 绑在已经消失的 netdev 上,
它自己不会发现,也不会重连。你把 CAN 拉起来它照样用不了 —— 这就是症状 A 的由来。
要重建的是**进程内的 socket**,只能靠重起进程。

**第 1 步 · 杀掉旧 bring-up 和空跑的 teleop 栈**

```bash
V=/home/yiming/openarm/openarm_track/vr
pkill -INT -f revised_bringup.launch.py
sleep 5
pkill -f "controller_manager/ros2_control_node"
$V/scripts/collect_up.sh down          # 遥操作栈还在跑的话
$V/scripts/preflight_udp.sh            # 收掉赖着 :9871 的 fanout
```

> 正常情况下杀 bring-up = 断力矩 = 臂砸下来,**但这个场景下 CAN 已经断了,
> 电机早就收不到指令、臂已经失去力矩自然下垂了**,所以没有额外风险。
> 动手前还是看一眼臂有没有被桌沿架住。

**第 2 步 · 插稳 USB**

PCAN 直插机身,不走 hub。插完按第 1 节判据 B 记基线、等 5 分钟验编号不涨。

**第 3 步 · 配 CAN**

```bash
sudo ip link set can0 down 2>/dev/null
sudo ip link set can0 type can bitrate 1000000 dbitrate 5000000 fd on
sudo ip link set can0 up
sudo ip link set can1 down 2>/dev/null
sudo ip link set can1 type can bitrate 1000000 dbitrate 5000000 fd on
sudo ip link set can1 up

ip -details link show can1 | sed -n '1,4p'    # 判据 A:UP / ERROR-ACTIVE / berr 0 / bitrate
```

**第 4 步 · 起 bring-up**(它自带 `return_to_zero()`,臂自己回零)

> ⚠️ 真机会动。先清空双臂工作空间。

```bash
source /opt/ros/jazzy/setup.bash
source /home/yiming/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=0

ros2 launch /home/yiming/openarm/openarm_track/revised/revised_bringup.launch.py \
    use_fake_hardware:=false right_can_interface:=can0 left_can_interface:=can1
```

判据:`Reached zero position (worst residual ...)`,残差要小。**此后不要 Ctrl-C。**

**第 5 步 · 回 home,继续采集**

```bash
$V/scripts/both_to_home.sh
$V/scripts/preflight_udp.sh
$V/scripts/collect_up.sh up --task "..."
```

#### ⚠️ 盒子还在反复掉的时候不要起 bring-up

它一启动就使能电机跑回零,走到一半 USB 再掉一次,左臂就在半途失去力矩砸下来,
而它末端挂着 1.44 kg 的手。**先插稳,再上电。**

> 附带:bring-up 崩了之后臂会失去力矩自然下垂,而**全零下垂位本来就接近 zero pose**,
> 所以从这个姿态重起 bring-up 反而是最安全的起始状态,不用手动去摆。

---

### 坑 4 · 腕部相机 `not-negotiated`,episode 起不来

```
[collect] wrist camera failed to start: camera did not deliver a frame:
Internal data stream error. ... streaming stopped, reason not-negotiated (-4)
[collect] episode NOT started -- fix it, or restart with --wrist-camera off
```

**跟 CAN 无关,是相机插在 USB 2.0 总线上。** D435i 采集用的是 `960x540@60fps`,
480M 的带宽排不出这个模式组合,于是 GStreamer 直接协商失败。

```bash
lsusb | grep 8086:0b3a                       # 在不在 USB 上
lsusb -t | grep -A6 uvcvideo | grep 5000M    # 必须是 5000M,480M 就是这个病
```

**解决:插到 USB3 口**(蓝色 / 带 `SS` 标)。判据是 `lsusb -t` 里那几行 `Class=Video`
显示 **5000M** 而不是 480M。

线也要对 —— D435i 原装是 USB-C 到 USB-C,中间串 A 转 C 的转接头很容易掉到 480M 甚至不认。

**临时绕过**(只是这批 episode 没有腕部视角):

```bash
$V/scripts/collect_up.sh up --wrist-camera off --task "..."
```

> **`/dev/videoN` 会在每次插拔后重新编号**(实测 Orbbec 从 `video14` 跑到 `video2` 又跑回
> `video14`)。这个**不用管** —— 采集脚本按设备名找相机,不按节点号
> (`collect/openarm_collect.py:1328`)。别看到编号变了就以为坏了。
