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

## Verdict

Keep the **bf16** rules v2 product. 4-bit bitsandbytes here is a **quality and speed regression** at matched serving settings. It is not the packing story (`gpu_memory_util≈0.9` hides weight size). AWQ/FP8 with fused kernels would be a different experiment.

Artifacts: `results/week_8_quant/`.
