# MiniMax H3 Block Cache (T8)

简体中文 | [English](README_EN.md)

面向 ComfyUI 原生 MiniMax H3 的实验性 F1B0 Block Cache 节点。它在每次模型调用中计算 Block 0，并在目标音频与目标视频都足够稳定时复用后续 Block 的 residual，直接跳过 Block 1–49。

## 安装

在 ComfyUI Manager 中搜索 `minimax-h3-block-cache-t8` 或 `MiniMax H3 Block Cache (T8)` 并安装，然后重启 ComfyUI。

也可以手动克隆到 `ComfyUI/custom_nodes`：

```powershell
git clone https://github.com/T8mars/comfyui-minimax-h3-blockcache-T8.git
```

本节点要求 ComfyUI `0.30.0` 或更高版本，并使用 ComfyUI 原生 MiniMax H3；没有额外 Python 依赖。

## 使用

重启 ComfyUI 后，在 `advanced/model_patches` 中添加 `MiniMax H3 Block Cache (T8)`：

```text
Load Diffusion Model
        │
        ▼
MiniMax H3 Block Cache (T8)
        ├────────► Basic Scheduler
        └────────► Basic Guider
```

默认参数是保守起点：

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `residual_diff_threshold` | `0.12` | 越高越容易命中，也越可能改变结果 |
| `start_percent` | `0.08` | 此进度前只完整计算并预热缓存 |
| `end_percent` | `0.95` | 此进度后关闭缓存 |
| `max_consecutive_hits` | `2` | 连续命中上限，之后强制刷新 |
| `cache_device` | `cpu` | CPU 节省显存；GPU 减少传输但占显存 |
| `metric_stride` | `8` | 音频/视频稳定性指标的抽样步幅 |

节点只接受 ComfyUI 原生 `MiniMaxH3Model`，并拒绝 EasyCache、LazyCache、Spectrum Apply MiniMax H3、原生 Block Sparse Attention 或已有的 H3 `double_block` replacement。Block Cache 不能与 Spectrum 或原生 Block Sparse Attention 串联；后两者会修改 H3 Block 执行或维护逐 Block 状态，但没有与 Block Cache 共享跳层、回放或失效契约。普通 SageAttention 不替换 H3 Block，可以继续作为 attention backend 使用。节点返回克隆后的 MODEL，不修改输入模型。

## 实现边界

- 音频与视频分别计算 Block 0 residual diff，任一超过阈值都执行完整 50 层。
- 缓存只保存目标音频/视频段，不保存 text、condition 或 reference rows。
- 命中时在 Block 0 后短路 H3 block 循环，再走原生 H3 final layer 与 unpatchify；当前 ComfyUI 会传入 PDD 所需的 sigma、完整采样 schedule 和音视频 shift，旧版继续使用四参数 final layer。
- 当前 ComfyUI 返回原始音频速度，旧版 ComfyUI 保留音频 schedule slope。
- 完整步保留 ComfyUI 的 dynamic vbar 预取；命中时清理本次尚未消费的预取状态，并兼容新旧预取清理接口及当前 malloc graph Block scope。
- 缓存状态分配会临时暂停当前 Comfy Compiler malloc graph，避免跨 Block 或跨采样步持有的缓存张量进入分配图；没有该接口的旧版 Core 继续使用原路径。
- 所有 tensor 状态限定在一次 sampling 内，正常结束、报错或取消都会清理。
- 不修改 ComfyUI 核心文件、不修改 Comfy Kitchen `.pyd`、不增加依赖或网络请求。

这是近似缓存，不保证同 seed 无损。不同 prompt、分辨率、帧数、步数、scheduler、attention backend 和量化格式都可能改变命中率与质量。

## 许可证

本项目采用 [Apache License 2.0](LICENSE)。允许个人及商业使用、修改和再分发，但须遵守许可证中关于保留许可证文本、修改声明和归属声明等条件。

## ComfyUI 新旧版本兼容

ComfyUI 提交 `bdcb886a4` 将 MiniMax H3 切换到 `ModelType.FLOW_AV` / `ModelSamplingAV`，并把 H3 音频输出从 slope-scaled velocity 改为 raw audio velocity。本节点在运行时检测 `ModelSamplingAV`：

- 当前 ComfyUI 在缓存命中后返回 raw audio velocity，并保留 H3 模型外层原生 `audio_scale` / carry transform 的处理边界。
- 旧版 ComfyUI 通过可选的 `time_shift_slope` 保持原有 slope-scaled velocity 行为。

当前新版已在 ComfyUI `a464ac335` 上完成白名单冷启动、完整启动，以及 FL2VA INT8 H3、512×512、22 帧、1 步真实前向验证；Block Cache 成功挂载执行，日志为 `cached 0/1 model forwards`，CPU cache 约 19.1 MiB。旧版分支已通过模拟旧接口的数值测试，但尚未在独立旧版 ComfyUI checkout 上进行真实模型回归。该测试工作流在采样完成后触发的核心 `SaveLatent` / `NestedTensor.contiguous` 错误与 Block Cache 无关。

ComfyUI 提交 `efd4e951a` 为预取引入 CUDA Graph 状态，并将 `cleanup_prefetched_modules` 从单参数改为 `(module, comfy_modules)`。本节点以同次引入的 `GRAPH_MODULES` 能力为边界：当前 ComfyUI 同时传入预取 Block 和模块列表，旧版继续使用单参数清理。

ComfyUI 提交 `2504e68d4` 为 H3 final layer 增加 PDD LoRA 采样参数。本节点检测 final layer 协议：当前 Core 在缓存命中时传递 `sigma`、`sample_sigmas` 和音视频 shift，旧 Core 继续调用原四参数接口。普通 H3 与 PDD head bank 都走 Core 自己的 final layer 实现。

ComfyUI 提交 `e308cc73b` 将 Sol-Attn、top-k/SLA 和 VSA/FastVideo 合并到原生 `Block Sparse Attention` 节点。该节点会替换 H3 的全部 DiT Block，并维护逐 Block 的跨步统计，因此本节点会在两种连接顺序下都于采样前明确拒绝组合。提交 `804eb5513` 引入 Comfy Compiler 后，本节点也会在缓存分配期间暂停 malloc graph，并在命中短路时闭合当前 Block scope；旧版 Core 没有这些接口时不会调用它们。

版本 `1.0.4` 在 ComfyUI `fbed745c8`（`0.34.0`）上通过 22 项测试、Ruff、compileall、白名单冷启动、新版七参数 final layer 命中探针，以及原生 Block Sparse Attention 两种连接顺序的实际 Core API 测试。兼容探针还分别从 ComfyUI `a464ac335`、上一发布基线 `187eda8ef` 和 raw-audio 协议更新前的 `2eb609766` 独立 checkout 导入本节点，确认旧版四参数 final layer、新旧预取清理、无 malloc-scope 路径和旧版 slope-scaled audio 协议仍可识别。为控制 33B 模型测试风险，本次发布前没有重复运行完整 H3 生成；最近一次真实 FL2VA 前向记录仍是上文的 `a464ac335` 基线。

## 本机冒烟基准

环境：RTX 4060 Ti 16GB、MiniMax H3 FL2VA INT8 ConvRot、`res_multistep` 20 步、256×160、5 帧、固定 prompt/seed。只统计 tqdm denoise 时间，不含文本编码、模型加载和 VAE decode。

| 配置 | 命中 | denoise | 相对基线 |
| --- | ---: | ---: | ---: |
| 无缓存 | 0/20 | 485.8s | 1.00× |
| CPU cache，默认 stride 8 | 7/20 | 443.8s | 1.09× |
| CPU cache，stride 9 探针 | 8/20 | 403.2s | 1.20× |
| GPU cache，默认 stride 8 | 7/20 | 471.6s | 1.03× |

单个命中步约 1–2 秒，完整步约 24–39 秒。该桌面环境存在明显负载波动，且样例远小于正式 H3 视频；这些数据只证明节点存在真实净加速路径，不应外推成固定宣传倍数。stride 9 会改变边界判定，因此不作为默认值。

## 验证

```powershell
python -m unittest discover -s tests -v
python -m ruff check .
```

当前覆盖 22 项测试，包括音视频联合判定、真实跳层、连续命中上限、sampling window、UUID/shape/sigma 失效、target-only 独立存储、执行清理、模型克隆、patch 冲突、Spectrum 与 Block Sparse Attention 前后两种连接顺序的互斥、新旧预取及 malloc-scope 接口、Comfy Compiler 缓存分配隔离、新旧 H3 final layer 协议、H3 音视频输出 shape/dtype，以及当前 raw / 旧版 slope-scaled 两种音频速度终结。

正式性能结论仍需完成官方 124 帧 T2V、FL2VA、Ref2VA 的音画质量矩阵与多次稳定计时。
