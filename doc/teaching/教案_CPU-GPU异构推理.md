# KTransformers CPU-GPU 异构推理 教案

> 面向对象：本科二/三年级学生，已初步接触过大语言模型（LLM）、PyTorch、CUDA 与计算机体系结构基础概念。
> 课程实验目标：理解 KTransformers 如何在**单张 GPU** 上运行**千亿参数级 MoE 大模型**，并能在本仓库上进行二次开发。

---

## 一、教学总览

### 1.1 一个核心问题贯穿全课

> DeepSeek-V3 有 **6710 亿（671B）参数**，光是把权重以 BF16 存下来就需要 ~1.3 TB 显存，而一张 RTX 4090 只有 24GB 显存。**为什么 KTransformers 能用一张卡跑起来？**

这是本节课要回答的唯一核心问题。所有技术点都是这个问题的答案的分解。

### 1.2 课时安排建议（共 4 学时）

| 学时 | 主题 | 关键产出 |
|------|------|----------|
| 第 1 学时 | 背景与核心思想：MoE + 异构计算 | 能用自己的话讲清"为什么单卡能跑千亿模型" |
| 第 2 学时 | 框架架构：注入机制与设备映射 | 能看懂 YAML 规则，知道哪段代码跑在哪个设备 |
| 第 3 学时 | 关键算子精讲：CPU MoE 与 GPU Attention | 能定位 CPU/GPU 各算子的代码位置与调用流程 |
| 第 4 学时 | 实验任务与二次开发指引 | 能完成实验作业，规划自己的改造方向 |

### 1.3 前置知识自检（课前下发）

学生应能回答下列问题；不能回答的部分建议先自学：
- 什么是 Transformer 的 Self-Attention？Q/K/V 是什么？
- 什么是 KV Cache？为什么解码阶段比预填充阶段快？
- 显存（VRAM）和内存（DRAM）的区别？带宽量级差异？
- 什么是量化（INT4/INT8）？为什么能省内存？
- CPU 的 SIMD 指令（AVX/AMX）大致是做什么的？

---

## 二、第 1 学时：背景与核心思想

### 2.1 先建立直觉：为什么千亿模型装不进单卡

用一张表让学生有"量级感"：

| 模型 | 参数量 | BF16 显存 | 一张 24GB 卡够吗 |
|------|--------|-----------|------------------|
| Llama-3-8B | 8B | ~16GB | 勉强够 |
| Qwen2-72B | 72B | ~144GB | 远远不够 |
| DeepSeek-V3 (MoE) | 671B 总 / 37B 激活 | ~1.3TB / ~74GB | 还是不够 |

**结论**：单纯靠"把模型塞进显存"这条路，单卡永远走不通。

### 2.2 第一个关键点：MoE——"用 671B 的身体，但每次只动 37B 的肌肉"

MoE（Mixture of Experts，专家混合）是理解 KTransformers 的前提。用一个比喻讲解：

> 一个医院（模型）有 256 个科室（专家 experts），每个病人（token）来了只挂 **8 个科室**的号。医院的"总编制"是 256 个科室的所有医生，但任意时刻在岗接诊的只是其中 8 个科室。

对应到代码层面：
- 每个 token 经过一个 **gate（路由/分诊台）**，gate 决定"激活哪几个 expert"。
- DeepSeek-V3 每 token 只激活 **8 个** expert（`num_experts_per_tok=8`），但总共有 **256 个** expert。

**这就是单卡能跑千亿 MoE 的第一个原因**：虽然权重总共有 671B，但每次推理真正参与计算的只有约 37B 参数。问题被转化为：**只要能高效地让"没被激活的专家"待在便宜的存储里就行**。

### 2.3 第二个关键点：异构计算——"GPU 干 GPU 擅长的，CPU 干 CPU 擅长的"

KTransformers 的核心思想可以用一句话概括：

> **把"每次都要算"的算子放 GPU，把"权重很大但每次只用一小部分"的算子放 CPU。**

具体到 MoE 模型，分工如下：

| 算子 | 特点 | 放在哪里 | 为什么 |
|------|------|----------|--------|
| Attention（含 KV Cache） | 每层每 token 都算，访存受限于 KV Cache 带宽 | **GPU** | GPU 显存带宽高（~1TB/s），适合反复读 KV Cache |
| MoE Gate（路由） | 参数小，但要快速算 | **GPU** | 计算密集、延迟敏感 |
| MoE Experts（专家 FFN） | 权重巨大（占模型 90%+），但每 token 只用 8/256 | **CPU** | CPU 内存便宜大容量（几百 GB DRAM），靠量化+指令集加速 |
| LayerNorm / Embedding | 参数小 | CPU 或 GPU | 看配置 |

### 2.4 第三个关键点：量化——"把权重压到 1/4 甚至 1/8"

CPU 内存虽然大，但如果不量化，671B 权重仍要 1.3TB DRAM。KTransformers 用 GGUF 格式做量化：

- **INT4（如 Q4_K）**：每个权重 4 bit，相比 BF16（16 bit）省 **4 倍**。
- **INT8（Q8_0）**：省 2 倍，精度更好。
- 还有 IQ1_S、IQ2_XS 等更激进的 1~2 bit 量化。

量化类型定义见 [custom_gguf.py](file:///workspace/archive/ktransformers/util/custom_gguf.py)（`GGMLQuantizationType` 枚举，L40-L101）。

### 2.5 第四个关键点：CPU 指令集加速——"让 CPU 算矩阵乘法不慢得离谱"

CPU 算矩阵乘法天然不如 GPU，但现代 CPU 有专用指令：
- **AVX2 / AVX-512**：SIMD 向量指令，一次处理多个数据。
- **AMX（Intel 高端至强才有）**：专门为矩阵乘法设计的指令，有专门的"矩阵寄存器"和 tile 计算，性能远超 AVX。

KTransformers 针对不同 CPU 提供不同后端，见 [kt-kernel/operators/amx/](file:///workspace/kt-kernel/operators/amx) 与 [kt-kernel/operators/avx2/](file:///workspace/kt-kernel/operators/avx2)。

### 2.6 本学时小结：单卡跑千亿模型的"四板斧"

用一张图（板书）总结：

```
千亿模型单卡运行 = 
   ① MoE 稀疏激活（只算 37B）       → 让"必须算"的量变小
   ② 异构分工（GPU/CPU 各司其职）   → 让"装得下"
   ③ 量化（INT4/INT8）              → 让"占的内存"变小
   ④ CPU 指令集加速（AMX/AVX）      → 让"CPU 算得不至于太慢"
```

> 课堂提问：如果去掉①MoE（改成 dense 模型），单卡还能跑千亿吗？为什么？
> （答：不能，因为 dense 模型每个 token 都要算全部参数，无法靠"稀疏激活"减少计算量。）

---

## 三、第 2 学时：框架架构——注入机制与设备映射

### 3.1 总体推理流程（先看全局）

一次推理从用户输入到输出 token，经过以下步骤。请对照代码理解：

```
用户输入文本
   │
   ▼
[local_chat.py] local_chat()        ← 入口函数
   │   1. 加载 tokenizer / config
   │   2. with torch.device("meta"): 构建模型骨架（不占内存）
   │   3. optimize_and_load_gguf()   ← 关键：注入异构算子 + 加载 GGUF 权重
   │   4. 创建 StaticCache (KV Cache)
   │   5. prefill_and_generate()     ← 推理主循环
   ▼
逐 token 输出
```

入口代码见 [local_chat.py](file:///workspace/archive/ktransformers/local_chat.py)：
- `local_chat()` 函数（L76-L139）：加载模型与注入。
- `default_optimize_rules` 字典（L58-L64）：模型类型 → YAML 规则文件的映射，这是"哪个模型用哪套异构规则"的总表。

### 3.2 核心机制一：YAML 驱动的"算子注入"（Injection）

这是 KTransformers 最核心的设计，也是二次开发最常打交道的部分。

#### 3.2.1 什么是"注入"

PyTorch 模型本质上是一棵 `nn.Module` 树。例如 DeepSeek-V3 的结构（简化）：

```
DeepseekV3ForCausalLM
├── model
│   ├── embed_tokens          (Embedding)
│   ├── layers[0..N]
│   │   ├── self_attn         (Attention)
│   │   ├── mlp
│   │   │   ├── gate          (路由)
│   │   │   └── experts       (256 个专家 FFN)  ← 重量级
│   │   └── ...
│   └── norm
└── lm_head
```

KTransformers 的做法：**不改原始模型代码，而是在运行时把这棵树里的某些节点"换成"自己写的优化实现**。这就叫"注入"（inject）。

#### 3.2.2 YAML 规则长什么样

以 DeepSeek-V3 为例，规则文件见 [DeepSeek-V3-Chat.yaml](file:///workspace/archive/ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat.yaml)。重点看这几条规则，它们直接决定了 CPU/GPU 分工：

```yaml
# 规则1：MoE experts 放 CPU（generate 阶段）
- match:
    name: "^model\\.layers\\..*\\.mlp\\.experts$"
  replace:
    class: ktransformers.operators.experts.KTransformersExperts
    kwargs:
      prefill_device: "cuda"      # 预填充阶段放 GPU
      prefill_op: "KExpertsTorch"
      generate_device: "cpu"      # 解码阶段放 CPU ← 关键！
      generate_op: "KExpertsCPU"
      out_device: "cuda"          # 输出结果拷回 GPU
  recursive: False
```

> 课堂讲解要点：`generate_device: "cpu"` 这一行就是"专家放 CPU"的源头。`prefill_device` 和 `generate_device` 的区别对应大模型推理的两个阶段（见 3.3）。

```yaml
# 规则2：Attention 放 GPU
- match:
    name: "^model\\.layers\\..*\\.self_attn$"
  replace:
    class: ktransformers.operators.attention.KDeepseekV2Attention
    kwargs:
      generate_device: "cuda"
      prefill_device: "cuda"
```

```yaml
# 规则3：普通 Linear（如 lm_head）放 GPU，用 Marlin 量化内核
- match:
    name: "^lm_head$"
    class: torch.nn.Linear
  replace:
    class: ktransformers.operators.linear.KTransformersLinear
    kwargs:
      generate_device: "cuda"
      prefill_op: "KLinearTorch"
      generate_op: "KLinearMarlin"
```

> 课堂练习（5 分钟）：让学生打开 [DeepSeek-V3-Chat-multi-gpu.yaml](file:///workspace/archive/ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat-multi-gpu.yaml)，找出"多卡时 experts 放哪"。

#### 3.2.3 注入的代码实现

核心函数 `inject()` 在 [optimize.py](file:///workspace/archive/ktransformers/optimize/optimize.py)（L28-L54）：

```python
def inject(module, local_optimization_dict, model_config, gguf_loader, prefix=''):
    for name, child in module._modules.items():
        child_prefix = prefix + name
        if child_prefix in local_optimization_dict:
            inject_module_meta = local_optimization_dict[child_prefix]
            if inject_module_meta["class"] != "default":
                # 动态导入 YAML 里指定的类，例如 ktransformers.operators.experts.KTransformersExperts
                import_class_name = import_path[-1]
                module_cls = getattr(__import__(...), import_class_name)
                # 实例化注入模块，传入 gguf_loader、config、原模块
                inject_module = module_cls(key=..., gguf_loader=..., config=..., orig_module=child, **kwargs)
                # 用注入模块替换原模块（树节点替换）
                set_module(module, name, inject_module)
```

一句话总结：**遍历模型树 → 命中 YAML 规则 → 动态 import 替换类 → `set_module()` 做节点替换**。

入口函数 `optimize_and_load_gguf()`（L129-L163）串起整个流程：
1. 读 YAML → 2. `gen_optimize_config()` 生成注入配置 → 3. `with torch.device("meta"):` 在 meta 设备上构建模型（不占真实内存） → 4. `inject()` 替换节点 → 5. `load_weights()` 加载 GGUF 量化权重到对应设备。

### 3.3 核心概念：prefill 与 generate 两个阶段

大模型推理分两阶段，KTransformers 给每个算子都配了两个实现：

| 阶段 | 俗称 | 输入 | 计算量 | KTransformers 对应字段 |
|------|------|------|--------|------------------------|
| prefill | 预填充 / 上下文 | 整个 prompt（成百上千 token） | 大，类似 batch 矩阵乘 | `prefill_device` / `prefill_op` |
| generate | 解码 / 逐 token | 每次 1 个 token | 小，但延迟敏感 | `generate_device` / `generate_op` |

**为什么 experts 在 prefill 时放 GPU、generate 时放 CPU？**
- prefill 是"大批量"计算，GPU 算得快，但需要权重都在显存——而 experts 权重太大放不下，所以这里其实是个权衡。
- generate 是"每次 1 个 token"，GPU 算 MoE 反而因访存效率低而不划算；放 CPU 用量化+指令集，配合 KV Cache 在 GPU 上的 attention，整体吞吐更高。

> 这个设计是 KTransformers 论文的核心贡献，建议作为课后阅读作业。

---

## 四、第 3 学时：关键算子精讲

### 4.1 GPU 侧：Attention 算子

文件：[attention.py](file:///workspace/archive/ktransformers/operators/attention.py)

`KDeepseekV2Attention`（L48-L67）继承自原始 DeepseekV2Attention，但：
- 通过 `BaseInjectedModule` 接收 `prefill_device`、`generate_device`。
- 实现 DeepSeek 独有的 **MLA（Multi-head Latent Attention）**，通过压缩 KV 维度大幅减小 KV Cache 体积（这也是 DeepSeek 能支持长上下文的关键）。

```
MLA 核心：把 K/V 压缩到低秩空间（kv_lora_rank），
         KV Cache 只存压缩表示，query 时再"展开"
```

相关代码 `get_absorbed()`（L69-L75）：把 `kv_b_proj` 拆成 `q_absorb` 和 `out_absorb`，这是 MLA 的"吸收"技巧，可以进一步减少计算。

attention 的前向有 `forward_chunck`（分块 prefill，L77 起）和解码路径两条，分别调用：
- `triton_attention_prefill.context_attention_fwd`（prefill）
- `triton_attention.decode_attention_fwd_grouped`（decode）

这些是 Triton 写的 GPU kernel，见 [triton_attention.py](file:///workspace/archive/ktransformers/operators/triton_attention.py) 和 [triton_attention_prefill.py](file:///workspace/archive/ktransformers/operators/triton_attention_prefill.py)。

> 对本科生的要求：知道 attention 在 GPU 跑、用的是 MLA 减小 KV Cache、有 prefill/decode 两条路径即可，不要求看懂 Triton kernel 细节。

### 4.2 GPU 侧：Gate 路由算子

文件：[gate.py](file:///workspace/archive/ktransformers/operators/gate.py)

`KMoEGateQwen2Moe.forward()`（L159-L175）：
```python
def forward(self, hidden_states):
    # 计算 gating logits（哪个 expert 权重高）
    logits = F.linear(hidden_states.float(), self.weight.float(), None)
    # grouped_topk：分组取 top-k，决定每个 token 激活哪几个 expert
    return grouped_topk(hidden_states, logits,
                        self.top_k, self.norm_topk_prob,
                        self.n_group, self.topk_group)
```

输出是 `(expert_ids, weights)`，交给后面的 experts 算子去执行。

> 讲解要点：gate 是"分诊台"，输出"哪些 expert 被激活 + 各自权重"。gate 本身参数小，放 GPU。

### 4.3 CPU 侧：MoE Experts 算子（本课重中之重）

文件：[experts.py](file:///workspace/archive/ktransformers/operators/experts.py)

#### 4.3.1 类继承结构

```
KExpertsBase (抽象基类, L68)
   ├── KExpertsCPU      (CPU 实现, L143)  ← generate 阶段用
   └── KExpertsTorch    (GPU 实现)        ← prefill 阶段用
```

`KTransformersExperts`（见 YAML）是个"调度壳"，根据 `generate_op`/`prefill_op` 在 CPU/GPU 实现间切换。

#### 4.3.2 KExpertsCPU 的 `load()`：把权重搬到 CPU 并初始化后端

L169-L258，核心逻辑：
1. 从 GGUF 加载 gate/up/down 三个权重张量（量化格式）。
2. 取出权重在内存中的指针（`ctypes.addressof`）。
3. 根据 `self.backend` 选择 CPU 计算后端：
   - `"llamafile"`：通用后端，基于 llamafile 的量化 GEMM。
   - `"AMXBF16"`：用 Intel AMX 指令算 BF16 MoE。
   - `"AMXInt8"`：用 AMX 算 INT8 MoE（运行时再量化）。
4. 预分配 CPU 侧的输入/输出 buffer，并设置 `pin_memory=True`（锁页内存，加速 CPU↔GPU 传输）。

```python
# L201-L219 llamafile 后端
if self.backend == "llamafile":
    moe_config = MOEConfig(
        n_routed_experts,             # 专家总数
        self.config.num_experts_per_tok,  # 每 token 激活数
        self.config.hidden_size,
        self.config.moe_intermediate_size,
        ...
        gate_ptr, up_ptr, down_ptr,  # 权重指针
        self.gate_type, self.up_type, self.down_type,  # 量化类型
        hidden_type,
    )
    self.moe = MOE(moe_config)
```

#### 4.3.3 KExpertsCPU 的 `forward()`：CPU 算完拷回 GPU

L320-L362 是解码阶段的核心。配合 CUDA Graph 时（L325-L334）：
```python
if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
    # 1. GPU 上的输入拷到 CPU 锁页内存
    KExpertsCPU.input_tensor_cpu[idx].copy_(input_tensor, non_blocking=True)
    KExpertsCPU.expert_ids_cpu[idx].copy_(expert_ids, non_blocking=True)
    KExpertsCPU.weights_cpu[idx].copy_(weights, non_blocking=True)
    # 2. 提交给 CPU 推理线程池（与 CUDA stream 同步）
    self.cpu_infer.submit_with_cuda_stream(
        torch.cuda.current_stream().cuda_stream,
        self.moe.forward(...))
    # 3. 等 CPU 算完
    self.cpu_infer.sync_with_cuda_stream(torch.cuda.current_stream().cuda_stream)
    # 4. 结果从 CPU 拷回 GPU
    KExpertsCPU.output_gpu_map[self.out_device][idx].copy_(
        KExpertsCPU.output_cpu[idx], non_blocking=True)
    return KExpertsCPU.output_gpu_map[self.out_device][idx]
```

**这就是 CPU-GPU 异构推理的最关键 4 步**，建议让学生在代码上标出来。

#### 4.3.4 CPU 推理线程池：CPUInfer

`KExpertsCPU.CPU_INFER = CPUInfer(Config().cpu_infer)`（L151），见 [cpuinfer.py](file:///workspace/archive/ktransformers/operators/cpuinfer.py)。

`CPUInfer` 是一个 CPU 侧的线程池/任务队列，作用是：
- 让 CPU 计算与 GPU 计算并行（CPU 算 experts 时 GPU 可以算 attention）。
- 支持与 CUDA stream 同步（`submit_with_cuda_stream` / `sync_with_cuda_stream`），避免 CPU 算完等 GPU、或 GPU 等 CPU。

> 课堂讲解要点：`submit_with_cuda_stream` + `sync_with_cuda_stream` 是让 CPU/GPU "重叠计算"的关键。这是异构计算性能能起来的根本原因——不是"CPU 替代 GPU"，而是"两者并行"。

### 4.4 整体数据流（一次 decode 步骤）

把所有算子串起来，一个 token 的解码流程：

```
[cur_token (GPU)]
      │
      ▼
embed_tokens (CPU) → inputs_embeds → 拷到 GPU
      │
      ▼ (逐层)
layer_i:
  ├── input_layernorm (GPU)
  ├── self_attn (GPU, MLA + KV Cache)          ← GPU 算
  ├── post_attention_layernorm (GPU)
  └── mlp:
        ├── gate (GPU) → 输出 expert_ids, weights
        └── experts (CPU, KExpertsCPU)         ← CPU 算
              1. 输入拷到 CPU 锁页内存
              2. cpu_infer.submit_with_cuda_stream(...)
              3. sync_with_cuda_stream(...)
              4. 输出拷回 GPU
      │
      ▼
lm_head (GPU) → logits → 采样下一个 token
```

> 课堂任务：让学生在 [DeepSeek-V3-Chat.yaml](file:///workspace/archive/ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat.yaml) 上，给每个 match 规则标注"GPU/CPU"。

### 4.5 CPU 后端的硬件适配（选讲，供深入学生）

新版 [kt-kernel](file:///workspace/kt-kernel) 把 CPU 算子按指令集拆分得更细：
- [kt-kernel/operators/amx/](file:///workspace/kt-kernel/operators/amx)：AMX 后端，`bf16-moe.hpp`、`fp8-moe.hpp`、`fp4-moe.hpp` 等。
- [kt-kernel/operators/avx2/](file:///workspace/kt-kernel/operators/avx2)：AVX2 后端，`gptq_int4-moe.hpp`、`mxfp4-moe.hpp` 等。
- 自动选择逻辑见 [kt-kernel/python/utils/amx.py](file:///workspace/kt-kernel/python/utils/amx.py)（L117-L180），根据 CPU flags 选后端。

> 选讲部分对一般本科生不强制，作为"想深入做 CPU kernel 优化"的同学的入口。

---

## 五、第 4 学时：实验任务与二次开发指引

### 5.1 实验环境准备

见仓库根目录 [install.sh](file:///workspace/archive/ktransformers/install.sh) 与 [doc/en/install.md](file:///workspace/doc/en/install.md)。
- 硬件最低要求：一张支持 CUDA 的 GPU（8GB+ 即可跑小模型实验）、CPU 支持 AVX2（最好有 AMX）。
- 推荐用 DeepSeek-V2-Lite（16B，小模型）做实验，权重见官方文档。

### 5.2 必做实验任务

#### 任务 1：跑通并观察日志（基础）

运行 `local_chat.py`，观察启动日志里的 `Injecting xxx as ...` 输出。
- **交付**：截图启动日志，标出至少 5 处 `Injecting` 行，说明每行对应 YAML 里的哪条规则、注入到 GPU 还是 CPU。

#### 任务 2：修改设备映射（理解注入机制）

在 [DeepSeek-V3-Chat.yaml](file:///workspace/archive/ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat.yaml) 的 copy 上做实验：
1. 把 `mlp.experts` 的 `generate_device` 从 `"cpu"` 改成 `"cuda"`，观察是否 OOM（显存不够）。
2. 改回 CPU，把 `self_attn` 的 `generate_device` 改成 `"cpu"`，观察速度变化。
- **交付**：两次实验的运行结果（成功/失败+报错）+ 用自己的话解释为什么。

#### 任务 3：定位代码（代码导航）

在仓库中找出并给出文件路径+行号：
1. `KExpertsCPU` 类定义在哪？`forward` 方法在哪几行？
2. `inject` 函数定义在哪？它调用 `set_module` 的那一行是？
3. 哪个 YAML 规则决定 experts 在 decode 阶段放 CPU？
4. `CPUInfer` 的 `submit_with_cuda_stream` 在 `forward` 里被调用的那一行是？
- **交付**：一份带文件链接和行号的清单。

#### 任务 4：画数据流图（综合）

画出一次 decode 步骤中，一个 token 在 GPU 与 CPU 之间的完整流转图（含 embed → attn → gate → experts → lm_head），标注每一步在哪个设备、数据何时跨设备拷贝。
- **交付**：一张图（手绘或软件画均可）。

### 5.3 选做/进阶方向（二次开发建议）

适合本科生做课程项目级别的二次开发：

1. **新增一个算子的注入规则**：给一个还没被注入的子模块（如某个 LayerNorm）写一条 YAML 规则，让它跑在指定设备，测量速度变化。
2. **新增 CPU 后端**：参考 [kt-kernel/operators/avx2/](file:///workspace/kt-kernel/operators/avx2) 下的某个 moe.hpp，实现一个简化版后端（可只支持小规模、固定形状），跑通 correctness 测试。
3. **Profiling 与瓶颈分析**：用 `torch profiler` 或 `nsys` 分析 prefill 和 decode 阶段的 CPU/GPU 时间占比，找出瓶颈，写一份分析报告。
4. **量化精度对比**：用 Q4_K vs Q8_0 vs BF16 跑同一组 prompt，对比生成质量和速度，写实验报告。
5. **多卡规则探索**：阅读 [DeepSeek-V3-Chat-multi-gpu.yaml](file:///workspace/archive/ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat-multi-gpu.yaml)，理解多卡时 experts 如何分片，画图说明。

### 5.4 二次开发常用入口文件清单

| 想做什么 | 看哪里 |
|----------|--------|
| 改 CPU/GPU 分工 | [optimize_rules/*.yaml](file:///workspace/archive/ktransformers/optimize/optimize_rules/) |
| 改注入逻辑 | [optimize/optimize.py](file:///workspace/archive/ktransformers/optimize/optimize.py) |
| 改 CPU experts 实现 | [operators/experts.py](file:///workspace/archive/ktransformers/operators/experts.py) 的 `KExpertsCPU` |
| 改 GPU attention | [operators/attention.py](file:///workspace/archive/ktransformers/operators/attention.py) |
| 改 CPU MoE kernel | [kt-kernel/operators/amx/](file:///workspace/kt-kernel/operators/amx)、[kt-kernel/operators/avx2/](file:///workspace/kt-kernel/operators/avx2) |
| 改 CPU 线程池 | [operators/cpuinfer.py](file:///workspace/archive/ktransformers/operators/cpuinfer.py)、[kt-kernel/cpu_backend/](file:///workspace/kt-kernel/cpu_backend) |
| 改推理主循环 | [util/utils.py](file:///workspace/archive/ktransformers/util/utils.py) 的 `prefill_and_generate` |
| 加新模型支持 | [models/](file:///workspace/archive/ktransformers/models/) 下加 modeling_xxx.py + 对应 YAML |

---

## 六、考核方式建议

| 环节 | 占比 | 形式 |
|------|------|------|
| 课堂参与 + 提问 | 10% | 课堂提问与讨论 |
| 必做实验任务 1-4 | 50% | 代码 + 报告 |
| 选做方向报告 | 30% | 实验报告 + 简短答辩 |
| 课后阅读理解 | 10% | KTransformers 论文读后感（1 页） |

---

## 七、课后阅读与参考

- KTransformers 论文：*KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models*（ACM SOSP 2025）。仓库首页有 BibTeX。
- DeepSeek-V3 技术报告（MLA 与 MoE 设计）。
- GGUF 量化格式说明：https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- Intel AMX 指令简介（Intel 官方文档）。
- 仓库自带文档：[doc/en/](file:///workspace/doc/en/) 下各 model tutorial，特别是 [DeepseekR1_V3_tutorial.md](file:///workspace/doc/en/DeepseekR1_V3_tutorial.md)。
- 官方在线 book：https://kvcache-ai.github.io/ktransformers/

---

## 八、教师备课要点提示

1. **不要一上来就讲代码**。先用第 1 学时建立"为什么单卡能跑千亿"的直觉，学生有了问题再去看代码才有目标感。
2. **YAML 注入机制是二次开发的"主战场"**。本科生做课程项目，90% 的情况是在改 YAML 或基于 YAML 加新算子，而不是动 C++/CUDA kernel。务必让学生熟练掌握 YAML 规则与 `optimize.py` 的对应关系。
3. **prefill vs generate 的区分**是理解所有 `*_device`/`*_op` 双字段的关键。建议用"批量算 vs 一个一个算"的比喻反复强调。
4. **CPU↔GPU 数据拷贝那 4 步**（拷入/提交/同步/拷出）是性能关键，也是最容易出 bug 的地方。建议让学生在 `KExpertsCPU.forward` 上动手加 print 调试。
5. **量级感很重要**：本科生对"671B""24GB""1TB/s 带宽"往往没有直觉，多用对比表和具体数字。
6. 进阶学生（想做 kernel 优化）应引导到 [kt-kernel](file:///workspace/kt-kernel) 目录，那里是新版、更模块化的 CPU kernel 实现。
