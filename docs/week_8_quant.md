# Week 8 — bitsandbytes 4-bit specialists

*31 Aug 2026. Same product router (rules v2 + few-shot), same suite (`mixed_ua_v4` 192), T=0, seed=42, max_tokens=256, **1 repeat**.*

Serve Mamay-4B / Lapa / Aya with vLLM `--quantization bitsandbytes --load-format bitsandbytes` (same HF checkpoints as bf16). Ports unchanged (`:8003` / `:8001` / `:8005`). `gpu_memory_utilization=0.90` and `max-model-len=8192` match the bf16 serves so KV-cache reservation is comparable.

Scripts: `cluster/serve_{mamay4,lapa,aya}_bnb.sbatch`.

## Result vs bf16 rules v2 (0.848)

| System | Overall | p50 | chat | code | translate | instruct | knowledge | alignment | GPU-s |
|--------|--------:|----:|-----:|-----:|----------:|---------:|----------:|----------:|------:|
| Rules v2 bf16 (3×) | **0.848** | **633** | 1.000 | 0.984 | 0.820 | **0.781** | **0.750** | 0.750 | ~295 |
| **Rules v2 4-bit bnb (1×)** | **0.830** | 1115 | 1.000 | **1.000** | 0.795 | 0.719 | 0.719 | 0.750 | **473** |

HTTP 192/192. Traffic unchanged: aya 63 · lapa 65 · mamay4 64.

Drops: instruct **−0.062** (2/32), knowledge **−0.031** (1/32), translate **−0.025**. Alignment holds at 0.750. Chat stays 1.000.

**Latency got worse**, not better: p50 633 → 1115 ms, GPU-s 295 → 473. bitsandbytes on vLLM 0.25.1 / Ada is a slower matmul path than fused bf16.

**nvidia-smi still ~44 GB/GPU.** With `gpu_memory_utilization=0.90` the allocator fills the card with KV cache. 4-bit weights do not show up as a smaller reservation. Fair weight-VRAM would need a lower util (or an isolated load); that was not this run.

## One GPU / packing — not this experiment

Quantization **yes**; colocating the pool on **one GPU** **no**.

This run kept the same three-process layout as bf16 (one vLLM per specialist, one GPU each, util 0.90). That answers “does 4-bit bitsandbytes beat bf16 at matched serving settings?” (it does not). It does **not** answer “can Mamay-4B + Lapa + Aya share a 48 GB card?”

A packing measurement would need at least one of:

- lower `gpu_memory_utilization` (and/or `max-model-len`) so weight size is visible in nvidia-smi;
- two vLLM processes on one GPU, or one process with multiple models;
- a weight-only VRAM snapshot before KV reservation.

None of those were run. Do not cite week 8 as a packing or “fits on one GPU” result.

## Verdict

Keep the **bf16** rules v2 product. 4-bit bitsandbytes here is a **quality and speed regression** at matched serving settings. It is not the packing story (`gpu_memory_util≈0.9` hides weight size). AWQ/FP8 with fused kernels, or a real colocation run, would be a different experiment.

## Follow-up — one GPU, online FP8 (27 Sep 2026)

Different experiment from the table above. Three vLLM processes on one RTX 6000 Ada, `--quantization fp8`, `--enforce-eager`, `max-model-len 4096`. `gpu_memory_utilization` is a fraction of the whole card: Lapa 0.42, Aya 0.28, Mamay-4B 0.18. After load, nvidia-smi showed **45 668 / 49 140 MiB**, split 21 392 + 14 530 + 9 726 MiB.

Rules v2 + few-shot, `mixed_ua_v4`, T=0, seed=42, max_tokens=256, **3 repeats**. HTTP 192/192 on each. The three repeats agree:

| System | Overall | p50 | chat | code | translate | instruct | knowledge | alignment |
|--------|--------:|----:|-----:|-----:|----------:|---------:|----------:|----------:|
| bf16 rules v2 (3×, three GPUs) | **0.848** | **633** | 1.000 | 0.984 | 0.820 | 0.781 | **0.750** | 0.750 |
| **FP8, one GPU (3×)** | **0.842** | 723 | 1.000 | 0.984 | 0.816 | 0.781 | 0.688 | **0.781** |

The 1× screen that gated the 3× was 0.842 / p50 725 ms, above the bitsandbytes bar of 0.830. Overall drop vs bf16 is 0.006. The bucket that moves is knowledge (0.750 → 0.688). p50 is slower (633 → 723 ms) and still far from the bitsandbytes 1115 ms. Resident GPUs go from 3 to 1. This does not replace the bf16 product number.

Artifacts: `results/week_11_pack_fp8/`. Script: `cluster/eval_pack_fp8.sbatch`.

Artifacts for the bitsandbytes run stay in `results/week_8_quant/`.
