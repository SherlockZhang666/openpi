# `tactile/deform` 的编码语义和训练时的取值

记录于 2026-09-19（同日更新：转换脚本已修）。结论：**盘上的数据是对的，deform 和力也是对应的。**
真正要小心的是转换脚本那一步的有损视频编码。

## 编码：常量 2 是"静止"，不是占位

`deform` 是 uint8，用 SDK 的 `deform_map_value()` 换算成 mm。从
`libsharpa-wave-sdk.so` 里把这个 LUT 取出来看，它是分段线性的：

```
code   0 -> 0.000 mm          code 0..100  : 每 code 0.005 mm
code   2 -> 0.010 mm          code 100..255: 每 code 0.030 mm   (粗 6 倍)
code 100 -> 0.500 mm
code 255 -> 5.150 mm
```

**整张图是一个绝对量表，受力时不是在 2 上"叠加"。** code 2 只是触觉垫静止时的读数
（0.01 mm，实质为零形变）。受力帧里 2 依然是众数：

```
                    峰值帧 |F|   众数    >2 的像素   最大 code (mm)
bag/193151/ep5      14.72 N    2 (94.9%)   4.51%    156 (2.69 mm)
bag/193151/ep1      13.22 N    2 (95.6%)   3.72%    162 (2.87 mm)
bulb/173356/ep0     13.52 N    2 (92.3%)   7.10%    157 (2.72 mm)
```

所以形变只占 3~7% 的像素，其余 93~97% 停在 2。

## 均匀图是真实读数，不要丢

`collect/tactile.py:504-528` 把这件事写得很清楚，而且是**特意**这么存的：

> A pad with no deformation anywhere IS a uniform map [...] Dropping it left the frame at
> its all-zero template, which is what a finger that never sent a packet also looks like.
> [...] In the 2026-08-19 egg sessions that erased 83.5% of the (frame, finger) samples
> and left ZERO no-load rows.

区分方式是**掩码，不是像素值**：

| 情况 | 像素 | `deform_valid` |
|---|---|---|
| 垫子静止，无形变 | 均匀平面（code 2） | **True** —— 真实读数 |
| 该手指这帧没来包 | 全零模板（code 0） | False |

⚠️ **`tactile_final` 里的计数器 `deform_placeholder_frames` 名字有误导性**：它统计的是
`n_deform_constant`，也就是均匀图的数量。**那些是真实的无接触读数，不是丢失的帧。**
看到"占位率 78%"不要以为数据坏了——那只是那根手指全程没碰到东西。

## deform 和力本来就是对应的

index 通道，Σ(形变 mm 超过基线的部分) 对 `tactile/f6` 合力：

```
bag/192042/ep0  +0.885     bag/193151/ep3  +0.863     bag/193151/ep5  +0.803
bag/193151/ep1  +0.934     bag/193151/ep4  +0.968     bulb/173356/ep0 +0.972
bag/193151/ep2  +0.658
```

接触面积（>2 的像素占比）对力的相关是 +0.70 ~ +0.99。原始数据不需要任何修正。

## 曾经的坑：转换脚本用 h264 crf23 编触觉图（**已修**，2026-09-19）

`convert_sharpa_data_to_lerobot.py` 原先对**所有**视频流一刀切 `vcodec="h264", crf=23`，
包括 5 路 deform。在一路真实 index 通道（524 帧，193151/ep_0005，|F| 峰值 14.7 N）
上实测往返误差：

| 编码 | 大小 | code 误差 均值/最大 | mm 误差 最大 | 均匀帧存活 |
|---|---|---|---|---|
| h264 crf23 yuv420p（**旧**） | 0.04 MB | 0.072 / **50** | **1080 µm** | **69 / 96 ← 坏了 27 帧** |
| **h264 crf0 yuv444p（现用）** | 0.34 MB | 0.004 / 1 | 30 µm | **96 / 96** |
| ffv1 gray | 0.37 MB | 0 / 0 | 0 µm | 96 / 96 |

crf23 坏在两处：

1. **最大误差 1.08 mm**，而峰值形变才 2.7 mm —— 局部误差到信号的 40%。
2. **28% 的无接触帧被破坏**：均匀平面被压出噪声后不再均匀，"垫子报告无形变"这个
   干净信号就没了。而这正是 dead-band 研究唯一的对照组。

聚合量倒是扛得住（`corr(Σdeform, |F|)` 从 +0.8031 只掉到 +0.8025），所以只用整体
接触强度的话 crf23 也能凑合；逐像素形变图不行。

ffv1 虽然比特精确，但 LeRobot 的 `encode_video_frames` 只接受
`{h264, hevc, libsvtav1}`，容器还是 `.mp4`，所以用不了。选 **h264 crf0 + yuv444p**：
yuv444p 是必须的，yuv420p 的色度下采样会把形变图和邻域平均掉。
试过加 `-color_range pc` 想消掉残留误差，**反而更糟**——解码端不认这个 tag，会重复
套用 tv→pc 展开，平白多出 +16 code 的偏移。

### 现在的实现

- `use_h264()` 不再全局 partial，改为装 `_encode_video_frames` 包装器，按 video key
  分流（LeRobot 的路径是 `videos/chunk-NNN/<video_key>/episode_NNNNNN.mp4`，取父目录名）。
  带 `tactile` 的用 `_TACTILE_ENCODING`，其余用 `_CAMERA_ENCODING`。
- `deform_code_to_mm()` + 三个常量硬编在同一个文件里，训练侧不需要装 SDK。
- `meta/sharpa_conversion.json` 多了 `video_encoding` 和 `deform_code_to_mm` 两个字段。

### 实跑验证（6 条 grab_a_sandwhich_bag，2549 帧）

逐帧比对 ep_0005 的解码结果和原始 hdf5：

```
           往返比特精确   max|err|   均匀帧 源→解
  index       False          1        90 → 90
  thumb       False          1         1 →  1
  pinky       True           0       519 → 519

head 相机: h264 yuv420p 410x256  1.82 MB   (crf23，未改)
触觉流  : h264 yuv444p 240x240  0.44 MB   (crf0)
```

**均匀帧 100% 保住**（旧编码会破坏 28%）。残留的 ±1 code 是 RGB↔YUV444 的取整，
不是编码器；换算成 mm 是 knee 以下 0.005 mm、以上 0.03 mm。
体积代价：6 集 × 5 指 = 4.3 MB（同批相机视频 15 MB）。

## 训练时还要注意

1. **要物理量就先过 LUT**。视频里存的是 SDK 原始 uint8 码，不是毫米——把 mm 再
   量化回 uint8 会丢掉 knee 以下 0.005 mm 的精细步长。直接把 uint8 归一化到 [0,1]
   则会带进 code 100 处 6 倍的非线性，deform 就和 `observation.tactile_force` 对不上了。
   用 `deform_code_to_mm()` 换成 mm **再**归一化。
2. **不要过滤均匀帧**。那是无接触的正样本，去掉会让模型只见过有接触的分布。
3. **`deform_valid` 可以照常当掩码用**，它的语义是"这帧这根手指解出了一张图"，
   转换脚本里那道断言是对的，保留。
4. **尾部/脱开帧不用管**。转换脚本用 `_longest_true_run(teleop/engaged)` 只保留最长
   的连续 engaged 段，尾部 hold、头部和中间断裂全部自动排除，原始 hdf5 不需要改。
   ⚠️ 但它只保留**最长**那一段且**不发警告**：中途脱开的 episode 会被悄悄砍掉一大块
   （实测 2 条分别只剩 245/521 和 236/493 帧，日志照样打 `kept 2 of 2`）。
   中途脱开的 episode 应该在入库前就剔除，不要指望这里兜底。

## 附：同批数据里两个未修的 attr bug

仅做记录，采集端代码没动：

- `with_contact` —— 全部 33 个 episode 恒为 `False`，与是否真的有触觉数据无关，别拿它做判据。
- `duration_s` —— 全部 33 个 episode 都比 `t_mono[-1] - t_mono[0]` 大 4.28~4.49 s。
  真实时长用 `t_mono` 跨度或 `num_frames / 30`。
