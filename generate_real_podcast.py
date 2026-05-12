#!/usr/bin/env python3
"""
Generate real conversational podcast audio using Claude (via CLI) + edge-tts.

Pipeline:
  1. Extract section text from index.html
  2. Send to `claude -p` to generate a two-host conversation script
  3. Parse the script into speaker turns
  4. Voice each turn with edge-tts (Microsoft Azure voices)
  5. Concatenate into final MP3

Usage:
    python3 generate_real_podcast.py --sections tokenization
    python3 generate_real_podcast.py --all
    python3 generate_real_podcast.py --all --force                # Regenerate audio (keeps cached transcripts)
    python3 generate_real_podcast.py --all --force-transcript     # Also re-run Claude
    python3 generate_real_podcast.py --list                       # List sections

Requires: claude CLI, beautifulsoup4, edge-tts
  pip install edge-tts beautifulsoup4
"""

import asyncio
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    from bs4 import BeautifulSoup
except ImportError:
    sys.exit("Install beautifulsoup4: pip install beautifulsoup4")

BASE_DIR = Path(__file__).parent
SOURCE = BASE_DIR / "index.html"
PODCAST_DIR = BASE_DIR / "podcast"
TRANSCRIPT_DIR = BASE_DIR / "podcast" / "transcripts"

# Voices for the two hosts — edge-tts
EDGE_VOICE_HOST = "en-US-GuyNeural"       # Alex — the explainer
EDGE_VOICE_COHOST = "en-US-JennyNeural"   # Jenny — the curious questioner

AUDIENCE = """The listener is a senior engineering leader at AWS — strong background in distributed systems, cloud infrastructure, and GPU workload scaling at cloud scale, but relatively new to ML internals. He learns best through concrete analogies mapped to infrastructure concepts he already knows: auto scaling, bin packing, memory hierarchies, distributed systems, EC2 capacity management."""

# Map section IDs to their chapter for context
CHAPTER_MAP = {
    'tokenization': 'ch1', 'embeddings': 'ch1', 'qkv': 'ch1', 'multihead': 'ch1',
    'layers': 'ch1', 'params': 'ch1', 'moe': 'ch1', 'numformats': 'ch1', 'sampling': 'ch1',
    'mla': 'ch2', 'attnvariants': 'ch2', 'ssm': 'ch2', 'mtp': 'ch2', 'codesign': 'ch2',
    'distill': 'ch2', 'lora': 'ch2',
    'nvlink': 'ch3', 'parallelism': 'ch3',
    'batching': 'ch4', 'kvcache': 'ch4', 'prefixcache': 'ch4', 'chunked': 'ch4',
    'flashattn': 'ch4', 'quant': 'ch4', 'marlin': 'ch4', 'loratp': 'ch4', 'specdecode': 'ch4', 'optlist': 'ch4',
    'reqlen': 'ch5', 'scheduling': 'ch5', 'outputpred': 'ch5', 'reqseqbatch': 'ch5',
    'multiturn': 'ch5', 'agentic': 'ch5', 'toolcalling': 'ch5', 'disagg': 'ch5', 'dynamo': 'ch5',
    'traintoserve': 'ch6', 'engines': 'ch6', 'sglang': 'ch6', 'benchmarking': 'ch6',
    'costeconomics': 'ch6', 'mfu': 'ch6',
    'cuda': 'ch7', 'numlibs': 'ch7', 'fusion': 'ch7', 'fileformats': 'ch7',
    'dynamo-orch': 'ch7', 'profiling': 'ch7',
    'gpuinternals': 'ch8', 'cachehier': 'ch8', 'archroadmap': 'ch8', 'superchips': 'ch8',
    'mig': 'ch8', 'otheraccel': 'ch8', 'formfactors': 'ch8',
    'containers': 'ch9', 'coldstart': 'ch9', 'autoscaling': 'ch9', 'multicloud': 'ch9',
    'security': 'ch9', 'observability': 'ch9', 'clientside': 'ch9',
    'sglang-dd': 'ch10',
    'trtllm-dd': 'ch10',
    'moe-dd': 'ch10',
    'sizing-dd': 'ch10',
}

CHAPTER_NAMES = {
    'ch1': 'The Model (how transformers work internally)',
    'ch2': 'Architecture Frontier (cutting-edge model designs)',
    'ch3': 'The Hardware (GPUs, memory, interconnects)',
    'ch4': 'Optimization (batching, caching, quantization, kernels)',
    'ch5': 'Serving at Scale (scheduling, routing, disaggregation)',
    'ch6': 'Operations (engines, benchmarking, cost economics)',
    'ch7': 'Software Stack (CUDA, kernel libraries, compilers, profiling)',
    'ch8': 'Hardware Depth (GPU internals, generations, accelerators)',
    'ch9': 'Production (containers, autoscaling, multi-cloud, security, observability)',
    'ch10': 'DeepDives (long-form, NotebookLM-style explorations of single topics)',
}

# Analogies pre-generated for each chapter to seed the conversation
CHAPTER_ANALOGIES = {
    'ch1': [
        'Tokenizer = the codec/serializer at the API gateway — encodes raw input into the internal representation the system processes',
        'Embedding table = a lookup cache mapping IDs to rich feature vectors, like a DNS resolver mapping names to IP addresses',
        'Attention = a load balancer that routes each token\'s query to the most relevant keys across the full sequence',
        'Multi-head attention = multiple independent load balancers, each specializing in different routing criteria (syntax, semantics, position)',
        'Layers = stages in a processing pipeline — each layer refines the representation, like successive middleware in a request pipeline',
        'Parameters = the total state the system must load into memory to serve — directly determines instance sizing',
        'MoE = capacity pooling with sparse routing — like having 8 specialized instance types but only activating 2 per request',
        'Number formats (FP16, BF16, FP8) = compression ratios for the model weights — smaller formats trade precision for throughput, like JPEG quality levels',
        'Sampling = the policy that turns probabilities into a chosen action — temperature/top-p are knobs on the policy, like exploration parameters in a load-balancing algorithm',
        'Constrained decoding = schema validation enforced at token-emission time, so output always parses by construction — no parse-retry loop',
    ],
    'ch2': [
        'MLA = compressing the KV cache by projecting keys/values into a smaller latent space — like compressing session state before storing it',
        'Attention-variant family = different cache strategies for different access patterns — windowed, sparse, latent, linear are like LRU/ARC/sliding-window cache eviction policies, each tuned for a different workload shape',
        'Mamba / SSMs = a fixed-size recurrent state instead of a growing KV cache — like replacing per-session storage with a constant-size session token',
        'Multi-token prediction = speculative execution — predict multiple future steps in parallel instead of one at a time',
        'Hardware co-design = designing your application architecture around the constraints of your infrastructure (memory bandwidth, compute ratios)',
        'Distillation = capability transfer via synthetic data — the teacher generates a workload, the student learns to handle it, like training a smaller cheaper service to replicate the behavior of a flagship one',
        'LoRA = a tiny low-rank patch on top of a frozen base model — multi-tenant fine-tunes are like per-customer config layered onto a shared service binary, with LoRA-aware routing standing in for tenant-aware request affinity',
    ],
    'ch3': [
        'GPU HBM = main memory, SRAM = L1 cache — the entire optimization game is keeping data close to compute',
        'NVLink = a high-bandwidth backplane between GPUs, like a dedicated low-latency network between instances in a placement group',
        'Tensor parallelism = sharding a model across GPUs like sharding a database across nodes — each shard processes its portion and they synchronize',
        'Pipeline parallelism = assembly-line processing — each GPU handles a different stage of the model, like microservices in a pipeline',
    ],
    'ch4': [
        'Continuous batching = a city bus that picks up and drops off passengers at every stop, vs. a charter that waits until full before departing',
        'KV cache = a warm instance pool — pre-computed state you keep around to avoid re-doing work on every decode step',
        'PagedAttention = virtual memory paging for the KV cache — allocate in fixed-size blocks, no fragmentation, can evict and reload',
        'Chunked prefill = rate limiting long prompts so they don\'t starve decode requests — like admission control for bursty workloads',
        'FlashAttention = a kernel that fuses multiple memory-bound operations into one pass — like combining multiple API calls into a batch request',
        'Quantization = lossy compression of model weights — trading bits of precision for 2-4x throughput, like downsampling images for faster CDN delivery',
        'Speculative decoding = optimistic concurrency — draft multiple tokens cheaply, then verify in one batch, accepting the ones that match',
        'Prefix caching = a CDN edge cache for the KV state of shared prompt prefixes — first request pays the full prefill, every subsequent request that shares the prefix gets near-zero TTFT',
        'Hash-chained vs radix-tree prefix cache = consistent hashing on fixed-size blocks vs a trie of token spans — the trie wins when traffic branches (parallel sampling, agent fan-out, ToT)',
        'Cache-aware routing = session affinity at the gateway — route a request to the replica that already holds its prefix instead of round-robin, or you trash your own cache',
    ],
    'ch5': [
        'Request scheduling = the scheduler in an auto-scaling group deciding which instance handles which request',
        'Admission control = a circuit breaker — reject requests early when the system is at capacity rather than letting them queue and timeout',
        'Output length prediction = capacity planning — estimate how much resource a request will consume before committing to serve it',
        'Disaggregation = separating prefill (compute-heavy) from decode (memory-heavy) onto different instance types, like splitting reads and writes to different database replicas',
        'Tool-call parser = a per-model-family protocol decoder — same idea as a content-type-specific deserializer at the API layer, since every model family encodes tool calls in its own dialect',
        'Reasoning tokens = hidden internal compute that still bills against your KV budget and decode bandwidth — like background jobs running on the same instance as your serving traffic',
        'Constrained decoding = schema validation enforced at token-emission time via an FSM mask — provably valid output by construction, no parse-retry loop',
        'NVIDIA Dynamo = the Kubernetes layer above inference engines — KV-aware routing + tiered KV memory + prefill/decode autoscaler, with vLLM/TRT-LLM/SGLang as the worker runtimes',
        'KVBM tiered KV memory = HBM/DDR/SSD/RDMA tiers, exactly the L1/L2/L3/disk hierarchy you already know, applied to conversation state',
        'NIXL = the RDMA bus that makes prefill/decode disaggregation cheap — without it the cross-pool KV handoff becomes the slowest hop',
    ],
    'ch6': [
        'Engine comparison = choosing between ECS, EKS, and Lambda — each inference engine makes different tradeoffs for different workloads',
        'SGLang = the inference engine that bet on RadixAttention and structured-output decoding — best-in-class for prefix-heavy and agentic workloads',
        'RadixAttention = a trie-keyed KV cache where branching generations (parallel sampling, ToT, agent fan-out) automatically share ancestor state in GPU memory',
        'Compressed FSM / jump-forward decoding = when the grammar forces a deterministic span of tokens, skip the model entirely for those tokens — like pre-computing a static portion of the response',
        'Benchmarking = load testing your inference endpoint — the metrics that matter and how to avoid misleading results',
        'Cost economics = unit economics of inference — cost-per-token is the new cost-per-request',
        'Roofline / arithmetic intensity = the same compute-vs-bandwidth framing you use for storage tiers, applied to a GPU — every operation lives somewhere on a curve and you optimize by moving it across the breakeven',
        'MFU = the utilization metric for GPUs — like CPU utilization but for matrix math throughput, and much harder to max out',
    ],
    'ch7': [
        'CUDA kernel = a function pinned to a specific compute resource pool — the lowest layer where work actually runs, like a Lambda handler running on a specific instance type',
        'GEMM = the central matrix multiply, ~95% of inference compute — every other op is incidental',
        'cuBLAS / cuDNN / CUTLASS / FlashInfer / DeepGEMM = the layered library tower from "general-purpose primitive" to "narrowly specialized fast path"',
        'Kernel fusion = combining a chain of memory-bound ops into one kernel so intermediate data stays in registers/SRAM — like collapsing a microservice chain into one binary to skip network hops',
        'CUDA Graphs = a pre-recorded DAG of kernel launches replayed in one shot — eliminates per-kernel host overhead, exactly like batching API calls',
        'safetensors = the universal LLM serving format — memory-mapped, sharded, zero code execution; ONNX = the cross-framework graph format; GGUF = the CPU/edge format',
        'NVIDIA Dynamo = the orchestration layer above engines, treating vLLM/SGLang/TRT-LLM as worker pools and adding cluster-wide KV-aware routing and disaggregation',
        'Nsight Systems / Nsight Compute = the timeline profiler and the per-kernel deep-dive — like distributed tracing vs single-service flame graphs',
        'DCGM = production GPU telemetry exporter — the equivalent of node_exporter but for GPUs',
    ],
    'ch8': [
        'SM = the basic execution unit of a GPU, like a single core in a CPU — H100 has 132 of them',
        'Tensor Core = the MMA unit (D = A·B + C) — where ~95% of inference FLOPs happen',
        'CUDA core / Tensor Core / SFU = scalar / matrix / transcendental units — three specialized execution pools per SM',
        'Cache hierarchy = registers → SRAM/L1 (256KB/SM) → L2 (50MB) → HBM (80GB) → host RAM — each tier 10× slower and 10× larger than the one above',
        '2:4 structured sparsity = hardware-accelerated sparsity pattern (2 zeros in every 4 weights) doubling Tensor Core throughput — only works because the hardware understands the exact pattern',
        'Architecture roadmap = each generation adds a smaller precision (BF16 → FP8 → FP4) and a new FlashAttention generation; quantization research follows hardware',
        'NVL72 / NVL144 = rack-scale NVLink domains where 72+ GPUs all reach each other at NVLink speed — turns "node + InfiniBand" deployments into "rack as a single unit"',
        'Grace / Vera = ARM CPUs on-package with the GPU via NVLink-C2C at 900 GB/s — makes host RAM a usable spillover tier for KV cache',
        'MIG = hardware partitioning that turns one big GPU into up to 7 isolated smaller GPUs — like EC2 dedicated-host slicing for GPUs',
        'PCIe vs SXM = same chip, different package — but SXM gets full HBM bandwidth, NVLink, and higher power; PCIe is the discount tier with no NVLink, unsuitable for TP',
        'Non-NVIDIA accelerators = TPU, Trainium, Groq, Cerebras, AMD MI300X — the silicon is competitive but the CUDA software moat keeps the LLM serving market on NVIDIA',
    ],
    'ch9': [
        'Cold start = GPU procurement → image pull → weight load → engine warmup, totaling 1–5 minutes — autoscaling on lagging metrics is too slow because of this',
        'NIM = pre-baked containers with quantized weights and compiled engines — pull-and-run instead of build-and-warm',
        'Five autoscaling knobs = min/max replicas, autoscaling window, scale-down delay, concurrency target — each has a default trap',
        'Queue depth as the leading indicator = the metric that grows before latency does, so autoscaling has time to react before SLO violations',
        'KV-aware + LoRA-aware routing = session-affinity at the load balancer for cache locality — round-robin is the wrong default for cached LLM serving',
        'Multi-cloud bin packing = scheduling replicas across hyperscalers, neoclouds, and reserved capacity — necessary at scale because no single provider has enough flagship GPUs',
        'Llama 3 paper failure rate = 1 GPU failure per 50,000 GPU-hours — a 1,000-GPU fleet sees ~3–4 incidents a week, so active-active is mandatory',
        'SOC 2 / HIPAA / data residency = the certifications and constraints customers actually ask for — region-pinned tenancy is often the biggest deployment constraint',
        'Tenant isolation in shared inference = per-tenant cache scoping, authoritative adapter selection, sandboxed tool execution — the unique multi-tenancy hazards of LLM serving',
        'Shadow + canary = the safe model-rollout pattern; mirror traffic to validate quality, then 1%/10%/50%/100% canary for capacity — blue-green doubles GPU cost and skips quality validation',
        'TLS handshake tax = up to 140 ms of latency on a fresh HTTPS connection — connection reuse via persistent SDK clients is the highest-leverage client-side fix',
        'Streaming = HTTP/SSE for one-way (LLM standard), WebSocket for bidirectional (voice / agents), gRPC for internal service-to-service with strict schemas',
        'Async with webhooks = fire-and-forget pattern for jobs over ~30 seconds — the only sane shape for batch and long-running agent work',
    ],
    'ch10': [
        'RadixAttention = a shared filesystem cache across processes — vs. PagedAttention which is virtual memory paging within a single process',
        'SGL-Router = a sticky/session-affinity load balancer in front of a fleet of replicas, where the "session" is the cached prefix',
        'Overlapped CPU scheduler = pipelining the dispatcher: build batch N+1 while the GPU runs batch N, like a CPU pipeline hiding the fetch stage',
        'HiCache = a tiered buffer pool for KV state — HBM is the hot tier, host RAM is warm, RDMA-attached storage is cold, exactly how a database manages pages',
        'Compressed FSM / jump-forward decoding = skip the model entirely for spans the grammar forces — same idea as a query optimizer constant-folding deterministic predicates',
        'NIM Factory three-engine matrix = a workload-shape acknowledgement: TRT-LLM for peak throughput, vLLM for general-purpose, SGLang for prefix-heavy/structured/agentic',
        'NIXL = the RDMA transport that lets KV blocks travel between nodes; SGLang HiCache speaks NIXL natively, which is why Dynamo + SGLang composes cleanly',
        'API-tier prompt caching (Anthropic, OpenAI) = the same content-addressable KV trick exposed at the API; RadixAttention is the open-source primitive shape',
        'TensorRT-LLM build step = AOT compile vs vLLM JIT — engine is shape/hardware/precision-specific, like a statically-linked C++ binary vs a Python script with an interpreter',
        'TensorRT engine = the serialized, pre-optimized artifact (kernel selection + quantization calibration + graph rewrites baked in); loaded by the PyExecutor runtime (Scheduler / KVCacheManager / ModelEngine / Sampler)',
        'CUDA Graph capture = recording a kernel-launch DAG once and replaying it per iteration — like batching API calls to skip per-call overhead; CUDA Graph padding rounds the dynamic batch to a captured size so the recording is reusable',
        'In-flight batching in TRT-LLM = continuous batching compiled into the engine, not a Python scheduler outside it — same primitive as vLLM, but the scheduler decisions can be fused with kernel launches',
        'Wide expert parallelism (Wide-EP) = sharding hundreds of experts across many GPUs so all-to-all between MLP-down and MLP-up dominates cost, then redesigning the AlltoAll kernel to fuse hidden-state + scaling + routing fields and quantize per-token outputs to NVFP4 on the wire',
        'PDL (Programmatic Dependent Launch) = device-level fine-grained pipelining where kernel B starts as soon as A produces enough output, without waiting for A to fully finish — a finer-grained version of the overlap scheduler',
        'MLA optimization in TRT-LLM = eliminating Q/K/V concat copies (PR 6538), FP8 Context FMHA matched to FP8 KV cache, 4-way LM-head TP — kernel-level surgery driven by DeepSeek R1 as the target workload',
        'FP8 vs NVFP4 vs INT4 AWQ in TRT-LLM = throughput tier (Hopper/Blackwell) vs Blackwell-specific aggressive tier vs small-batch memory-bound escape hatch — three layers of the same compression stack',
        'Speculative decoding family in TRT-LLM = N-Gram, Lookahead, Medusa, Eagle, Eagle3, MTP — six drafters, all attacking the autoregressive serial chain, ~1.5–2× on compatible workloads',
        'Multiblock Attention = long-context attention split into blocks that run in parallel — paged-attention-aware kernel that delivers >3× throughput on long sequences',
        'NIM Factory = pre-compiled TRT-LLM engine artifacts shipped per (model × precision × hardware × parallelism) cell; the build pipeline\'s hard problem is the combinatorial validation matrix, not the build itself',
        'Pre-compiled NIM engine vs customer-compiled = fast cold start + predictable perf + fixed validation surface, at the cost of inflexibility on shapes/precisions outside the matrix; vs flexibility at the cost of weeks of build/calibration time and tracking TRT-LLM\'s release cadence',
        'MoE all-to-all = the network shuffle phase of map-reduce, but executed twice per layer, synchronously, with dynamic per-rank payload sizes — that is why it dominates inference latency at scale, often 40-60% of runtime',
        'Expert parallelism (EP) = sharding a sparse model by which expert lives on which GPU — like sharding a database by tenant ID rather than by row hash, except the routing key (the gate output) is recomputed every token',
        'Wide-EP (EP32, EP64, EP320) = the "use the whole rack as one MoE" answer, only sane when you have an NVLink fabric (GB200 NVL72 at 130 TB/s) wide enough to absorb the all-to-all',
        'Synchronous all-to-all = a global barrier — the slowest expert pins everyone to its tail latency, so dynamic routing skew turns into idle GPU time on the popular experts\' victims',
        'EPLB (Expert-Parallel Load Balancer) = a placement and replication scheduler for hot experts — same idea as hot-shard replication in a sharded database, plus periodic remapping based on observed traffic',
        'Redundant experts (32 in DeepSeek prefill) = the MoE version of read replicas for hot keys — duplicate the popular experts across GPUs so the dispatch step has options',
        'DualPipe (DeepSeek-V3) = bidirectional pipeline that overlaps attention compute, MoE compute, and MoE all-to-all in one schedule — eliminates pipeline bubbles the way hyperthreading hides memory stalls',
        'MLA (Multi-head Latent Attention) = compress K and V jointly into a small latent — KV cache shrinks by ~10×, at the cost of a tiny up-projection on every read; the reason DeepSeek can fit context this large at this throughput',
        'Auxiliary-loss-free load balancing (DeepSeek) = balance experts via a learned routing bias, not via an auxiliary loss term — keeps the model quality clean while still hitting balanced expert utilization',
        'MTP (Multi-Token Prediction) at inference = repurpose the training-time MTP head as a built-in speculative drafter — DeepSeek-R1 on B200 gets 2.82× acceptance with 3 stacked MTP layers',
        'DeepEP = DeepSeek\'s open-source EP all-to-all kernels, designed for the H800 topology (NVLink 160 GB/s intra-node + IB 50 GB/s inter-node) with 20 SMs allocated to cross-node transfer',
        'No-token-dropping inference = unlike training, DeepSeek does not drop overflow tokens at inference, which forces the deployment to use redundant experts and dynamic balancing instead of capacity factor',
        'Disaggregated prefill/decode for MoE = the two phases stress different experts — prefill is a wide compute sweep (TP4+SP, EP32 on 32 GPUs), decode is wide EP (EP320 on 320 GPUs) — same disaggregation idea as for dense, but the EP rank is part of the split',
        'One-sided AlltoAll over NVLink = use NVLink one-sided semantics so the receiver does not stall waiting for a matching send — like RDMA one-sided put/get vs MPI Send/Recv',
        'NVFP4 wire format on the AlltoAll combine = quantize the expert outputs before they cross NVLink so the bytes-on-the-wire shrink, like compressing payloads before a network shuffle in a distributed sort',
        'EPLB caveat (LMSYS) = the load balancer is tuned on in-distribution traffic; production traffic is not in-distribution, so EPLB is a starting point not a finished product',
        'LMSYS 96-H100 reproduction = the public proof point for DeepSeek-class MoE inference on commodity H100 — 52.3k input / 22.3k output tokens/sec/node, TTFT 2-5s, ITL ~100ms, within 5.6%/6.6% of DeepSeek\'s own numbers, but throughput-optimized not latency-optimized',
        '~700 GB FP8 weights for 671B = even at FP8, the model does not fit on one node — every deployment is a multi-node aggregation problem before any optimization starts',
        'Arithmetic intensity = FLOPs per byte loaded from memory — the single number that decides whether a kernel is compute-bound or memory-bound, like ops-per-IO in a storage system',
        'Roofline model = the throughput ceiling as a function of arithmetic intensity — flat (compute) above the ridge, linear (bandwidth) below, the same plot you would draw for any storage tier',
        'Critical batch size (B_crit) = the smallest batch where a matmul becomes compute-bound — ~240 tokens on TPU v5e, ~280 on H100 in bf16; the magic number that decides whether more batching helps',
        'Prefill = compute-bound matmul soup over the whole prompt; decode = memory-bandwidth-bound per-token KV streaming — same model, opposite regimes, the entire reason for PD disaggregation',
        'Per-token KV cache size (bytes) = 2 × bytes_per_float × num_kv_heads × head_dim × num_layers — the formula that decides how much context you can serve at what batch size, period',
        'MLA per-token KV (DeepSeek-V3) = 70.272 KB at BF16, vs 327.68 KB Qwen-2.5 72B GQA (4.66×) and 516.10 KB Llama-3.1 405B GQA (7.28×) — the only architecture-level KV win that compounds at long context',
        'GQA / MQA / MLA = three points on the same Pareto frontier — how aggressively to share KV heads across query heads, trading quality for cache size',
        'PagedAttention block size = 16 tokens per block in vLLM by default — picks the granularity of KV allocation, like a page size in virtual memory',
        'KV cache fraction = the share of GPU memory you hand to the KV pool after weights — the single knob with the largest throughput impact in any engine',
        'TRT-LLM max_batch_size / max_num_tokens = the two-knob grid search every TRT-LLM deployment runs — Llama-3.3 70B on 4×H100 finds the optimum at 512 / 2048, 21% over default 2048',
        'Saturation batch (B_sat) = the batch where extra tokens stop improving throughput because step time grows with B — the natural cap for any engine',
        'Theoretical step time floor = (B × KV_size + param_bytes) / HBM_bandwidth — the memory-bound lower bound every benchmark has to live above',
        'Disaggregated serving (PD split) = run prefill on a low-batch compute-bound pool, decode on a high-batch memory-bound pool, ship KV across — separates the two regimes so each is tuned for its own roofline',
        'Continuous batching priority = decode-then-prefill — admit a new prompt only when there is room without starving in-flight decodes, exactly the elevator-scheduling problem',
        'Speculative decoding economics = trade FLOPs for memory-bandwidth wins — a 1.8× win on a memory-bound decoder is essentially free FLOPs, because the GPU was idle on bandwidth anyway',
        'KV quantization (INT8 / FP8) = compress the cache 2× — gives you 2× the batch at the same memory, sliding straight up the roofline',
        'Character.AI scale = 20k QPS, <1¢/hour conversation, 33× cost reduction since 2022, 13.5× cheaper than competitor APIs — the cited public proof that radical KV reduction works at consumer scale',
        'Two-batch / micro-batch overlap (DeepSeek-V3) = while one micro-batch computes MLA+MoE, the other does all-to-all — dual-pipeline pattern, the production version of ScMoE',
        'DeepSeek-V3 TPOT math = 61 layers × 241.92μs all-to-all (CX7 400Gbps IB) = 14.76ms TPOT theoretical upper bound = 67 tok/sec; same on GB200 NVL72 (900GB/s) drops to 0.82ms TPOT = 1200 tok/sec',
    ],
}

def build_prompt_for_section(section_text: str, section_title: str, sec_id: str) -> str:
    """Build a tailored prompt for each section based on its chapter and content."""
    chapter = CHAPTER_MAP.get(sec_id, 'ch1')
    chapter_name = CHAPTER_NAMES.get(chapter, 'LLM Inference')
    analogies = CHAPTER_ANALOGIES.get(chapter, [])

    analogies_block = '\n'.join(f'  - {a}' for a in analogies)

    # Long-form NotebookLM-style deep dives target 45-60 minutes of audio.
    if chapter == 'ch10':
        return f"""You are writing a long-form, NotebookLM-style podcast script for a two-host show about LLM inference infrastructure.

AUDIENCE:
{AUDIENCE}
The listener already knows vLLM well (PagedAttention, continuous batching, prefix caching basics). Do NOT re-explain those — reference them as known and build on them.

HOSTS:
- **Alex** (male): The main explainer. Deep ML inference expertise. Authoritative but accessible. Uses analogies constantly, especially distributed-systems / cloud-infra ones.
- **Jenny** (female): Sharp, infra-systems-background co-host. Asks the questions the listener would ask. Pushes Alex to go deeper. Reacts genuinely. Connects new ideas back to vLLM, paging, caches, load balancers, buffer pools.

EPISODE TOPIC:
"{section_title}" — a single-topic deep dive in the DeepDives chapter.

SOURCE MATERIAL (this is the structured outline; the section is already organized into five segments — preserve that segment structure in the conversation):
---
{section_text}
---

PRE-GENERATED ANALOGIES (use, adapt, extend):
{analogies_block}

STRUCTURE:
The source is organized as five segments. Walk through them in order, but make the transitions feel natural — Jenny's questions should bridge between segments. Roughly:
- Segment 1 (~5 min): What it is and why it exists. Hook the listener with the problem the system solves.
- Segment 2 (~10 min): The headline technical idea. Go deep. This is where the listener should leave with a real mental model.
- Segment 3 (~10 min): The complementary techniques. Cover each one as its own mini-arc.
- Segment 4 (~5 min): The decision framework. When to pick this vs alternatives. Be honest about tradeoffs.
- Segment 5 (~5 min): The real-world / interview-ready framing. Productization, ecosystem, what to actually say.

DEPTH INSTRUCTION:
For EVERY concept, explain WHY it works the way it does. Jenny pushes: "Why not just...?", "What breaks if you don't?", "How does that play out at scale?", "Wait, isn't that just...?". Go slow. Three important aspects of one concept get three exchanges, not one paragraph. Attribute numbers when you cite them ("the paper reports", "the LMSYS blog says", "per the CMU lecture"). When a number is workload-dependent, say so.

CONVERSATION INSTRUCTION:
This must sound like two real people talking, not reading. Include:
- Natural reactions, mid-sentence interruptions, "wait, hold on" moments
- Jenny connecting back to infra concepts she already knows (paging, caches, load balancers, buffer pools, sticky sessions, RDMA, pipelining)
- Moments where Alex refines an analogy after Jenny pushes back
- Occasional humor, surprise, "I had to think about this for a while too"
- Building on each other's points, not alternating monologues
- Brief recaps when transitioning between segments

LENGTH:
Target **7,000–9,000 words** of dialogue, which produces **45–60 minutes** of audio at conversational pace (~150 wpm). This is a long-form deep dive, NOT a quick overview. If you find yourself running short, go deeper on the WHY — there is always more to unpack.

FORMAT:
Output ONLY the dialogue lines. No stage directions, no notes, no segment headers, no "[pause]" markers.
Format each line EXACTLY as:
Alex: <what Alex says>
Jenny: <what Jenny says>

Start with a hook — something that connects to the listener's existing vLLM knowledge — then build outward."""

    return f"""You are writing a podcast script for a two-host show about LLM inference infrastructure.

AUDIENCE:
{AUDIENCE}

HOSTS:
- **Alex** (male): The main explainer. Has deep expertise in ML inference. Speaks with authority but accessibly. Uses analogies constantly — especially ones drawn from distributed systems and cloud infrastructure.
- **Jenny** (female): A sharp, curious co-host. Comes from an infrastructure/systems background (like the listener). Asks the exact questions the listener would ask. Pushes Alex to go deeper, not just define terms. Reacts genuinely ("Oh wait, that's exactly like...", "So you're saying...", "Hang on, why not just...").

CONTENT TO COVER:
This episode covers "{section_title}" from the chapter "{chapter_name}".

Here is the source material:
---
{section_text}
---

PRE-GENERATED ANALOGIES (use these as a starting point, adapt and extend them):
{analogies_block}

NARRATIVE INSTRUCTION:
Do NOT survey the concepts one by one like a textbook. Build a connected story where each concept leads naturally to the next. Start with a hook that connects to something the listener already understands, then build outward. Each new concept should feel like the natural next question.

DEPTH INSTRUCTION:
For EVERY concept, explain WHY it works the way it does — not just WHAT it is. When Alex explains something, Jenny should push: "But why not just...?", "What breaks if you don't do that?", "How does that play out at scale?" Go slow. Let the concepts breathe. If a concept has 3 important aspects, give each one its own exchange — don't compress three ideas into one paragraph.

CONVERSATION INSTRUCTION:
This must sound like two real people talking — not reading from a script. Include:
- Natural reactions and interruptions
- Jenny connecting new ideas back to infrastructure concepts she knows
- Moments where Alex corrects a misconception or refines an analogy
- Occasional humor or surprise
- Building on each other's points rather than alternating monologues

CLOSING:
After covering the section content, close with a 2-minute "connect forward" segment where Alex and Jenny preview what comes next and why it matters — tease the listener about where these concepts lead.

LENGTH:
Aim for 8-12 minutes when spoken aloud (~1200-1800 words). This is longer than a quick overview — take the time to go deep.

FORMAT:
Output ONLY the dialogue lines. No stage directions, no notes, no headers.
Format each line EXACTLY as:
Alex: <what Alex says>
Jenny: <what Jenny says>

Start the conversation directly."""


# ── Text extraction (reused from generate_podcast.py) ──

SKIP_CLASSES = {'prev-next', 'resource-bar', 'quiz', 'breadcrumb', 'kbd-hint', 'new-badge'}
VIZ_CLASSES = {'viz-flow', 'viz-timeline', 'viz-grid', 'viz-mem', 'fmt-row', 'fmt-bits',
               'fmt-legend', 'token-legend', 'token-row', 'legend-item'}


def extract_sections(html_path: Path) -> dict[str, list[str]]:
    with open(html_path, encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')

    sections = {}
    for div in soup.find_all('div', id=lambda x: x and x.startswith('sec-')):
        sec_id = div['id'].replace('sec-', '')
        paragraphs = []
        for el in div.children:
            if not hasattr(el, 'get'):
                continue
            classes = set(el.get('class', []))
            if classes & SKIP_CLASSES or classes & VIZ_CLASSES:
                continue
            tag = el.name
            if tag == 'table':
                text = table_to_text(el)
            else:
                text = el.get_text(separator=' ', strip=True)
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) > 10:
                paragraphs.append(text)
        sections[sec_id] = paragraphs
    return sections


def table_to_text(table) -> str:
    headers = [th.get_text(strip=True) for th in table.find_all('th')]
    rows = [r for r in table.find_all('tr') if r.find('td')]
    if not rows:
        return ''
    lines = []
    for row in rows[:8]:
        cells = [td.get_text(strip=True) for td in row.find_all('td')]
        parts = [f"{headers[j]}: {c}" if j < len(headers) else c for j, c in enumerate(cells)]
        lines.append('; '.join(parts))
    return '\n'.join(lines)


# ── Conversation generation via Claude CLI ──

def generate_conversation(section_text: str, section_title: str, sec_id: str = '') -> str:
    """Call claude -p to generate a podcast conversation script."""
    prompt = build_prompt_for_section(section_text, section_title, sec_id)

    result = subprocess.run(
        ['claude', '-p', prompt, '--output-format', 'text'],
        capture_output=True, text=True, timeout=900
    )

    if result.returncode != 0:
        print(f"    Claude CLI error: {result.stderr[:200]}")
        return ""

    return result.stdout.strip()


def parse_conversation(script: str) -> list[dict]:
    """Parse 'Alex: ...' / 'Jenny: ...' lines into structured turns."""
    turns = []
    current_speaker = None
    current_text = []

    for line in script.split('\n'):
        line = line.strip()
        if not line:
            continue

        # Match speaker prefix
        match = re.match(r'^(Alex|Jenny)\s*:\s*(.*)', line, re.IGNORECASE)
        if match:
            # Save previous turn
            if current_speaker and current_text:
                turns.append({
                    'speaker': current_speaker,
                    'text': ' '.join(current_text)
                })
            current_speaker = match.group(1).capitalize()
            current_text = [match.group(2)] if match.group(2) else []
        elif current_speaker:
            # Continuation of current speaker's turn
            current_text.append(line)

    # Don't forget the last turn
    if current_speaker and current_text:
        turns.append({
            'speaker': current_speaker,
            'text': ' '.join(current_text)
        })

    return turns


# ── Audio generation: edge-tts backend ──

async def tts_with_retry(text: str, voice: str, output_path: Path, max_retries: int = 3):
    """Call edge-tts with retry on 503/rate limit errors."""
    import edge_tts
    for attempt in range(max_retries):
        try:
            communicate = edge_tts.Communicate(text, voice, rate="+5%")
            await communicate.save(str(output_path))
            return
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)  # 5s, 10s, 15s
                print(f'    TTS retry {attempt+1}/{max_retries} after {wait}s ({e})')
                await asyncio.sleep(wait)
            else:
                raise


async def generate_audio_edge(turns: list[dict], output_path: Path):
    """Generate MP3 from conversation turns using edge-tts."""
    temp_dir = output_path.parent / '.temp'
    temp_dir.mkdir(exist_ok=True)

    temp_files = []
    for i, turn in enumerate(turns):
        voice = EDGE_VOICE_HOST if turn['speaker'] == 'Alex' else EDGE_VOICE_COHOST
        temp_path = temp_dir / f'{output_path.stem}_{i:03d}.mp3'

        await tts_with_retry(turn['text'], voice, temp_path)
        temp_files.append(temp_path)

    # Concatenate all clips
    with open(output_path, 'wb') as out:
        for tf in temp_files:
            with open(tf, 'rb') as inp:
                out.write(inp.read())

    # Cleanup
    for tf in temp_files:
        tf.unlink()
    try:
        temp_dir.rmdir()
    except OSError:
        pass


# ── Main ──

async def process_section(sec_id: str, paragraphs: list[str], section_names: dict):
    """Full pipeline for one section: text → conversation → audio."""
    title = section_names.get(sec_id, sec_id)
    output_mp3 = PODCAST_DIR / f'{sec_id}.mp3'
    transcript_path = TRANSCRIPT_DIR / f'{sec_id}.txt'

    # Step 1: Check if transcript already exists (skip Claude call)
    if transcript_path.exists():
        print(f'  [{sec_id}] Using cached transcript')
        script = transcript_path.read_text()
    else:
        # Generate conversation via Claude
        section_text = '\n\n'.join(paragraphs)
        print(f'  [{sec_id}] Generating conversation via Claude CLI...')
        script = generate_conversation(section_text, title, sec_id)
        if not script:
            print(f'  [{sec_id}] Failed to generate conversation')
            return False
        # Cache the transcript
        transcript_path.write_text(script)

    # Step 2: Parse turns
    turns = parse_conversation(script)
    if len(turns) < 4:
        print(f'  [{sec_id}] Warning: only {len(turns)} turns parsed (expected more)')
        if len(turns) == 0:
            return False

    alex_turns = sum(1 for t in turns if t['speaker'] == 'Alex')
    jenny_turns = sum(1 for t in turns if t['speaker'] == 'Jenny')
    total_words = sum(len(t['text'].split()) for t in turns)
    print(f'  [{sec_id}] {len(turns)} turns ({alex_turns} Alex, {jenny_turns} Jenny, ~{total_words} words)')

    # Step 3: Generate audio with edge-tts
    print(f'  [{sec_id}] Generating audio (edge-tts)...')
    await generate_audio_edge(turns, output_mp3)
    size_kb = output_mp3.stat().st_size / 1024
    print(f'  [{sec_id}] -> {output_mp3.name} ({size_kb:.0f} KB)')
    return True


def get_section_names(html_path: Path) -> dict:
    """Extract section display names from nav buttons."""
    with open(html_path, encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
    names = {}
    for btn in soup.find_all('button', class_='nav-item'):
        sec = btn.get('data-sec', '')
        if sec:
            names[sec] = btn.get_text(strip=True)
    return names


async def main():
    parser = argparse.ArgumentParser(
        description='Generate real conversational podcast using Claude CLI + edge-tts')
    parser.add_argument('--all', action='store_true', help='Generate all sections')
    parser.add_argument('--sections', nargs='*', help='Specific section IDs')
    parser.add_argument('--list', action='store_true', help='List sections')
    parser.add_argument('--force', action='store_true',
                        help='Regenerate even if MP3 exists (keeps cached transcripts)')
    parser.add_argument('--force-transcript', action='store_true',
                        help='Also regenerate transcripts (re-run Claude)')
    args = parser.parse_args()

    if not any([args.all, args.sections, args.list]):
        parser.print_help()
        return

    try:
        import edge_tts  # noqa: F401
    except ImportError:
        sys.exit("Install edge-tts: pip install edge-tts")

    sections = extract_sections(SOURCE)
    section_names = get_section_names(SOURCE)

    if args.list:
        for sid, paras in sections.items():
            words = sum(len(p.split()) for p in paras)
            name = section_names.get(sid, sid)
            print(f'  {sid:20s} {name:30s} ~{words:,} words')
        return

    PODCAST_DIR.mkdir(exist_ok=True)
    TRANSCRIPT_DIR.mkdir(exist_ok=True)

    target_ids = args.sections if args.sections else list(sections.keys())
    generated = 0

    for sec_id in target_ids:
        if sec_id not in sections:
            print(f'  Skipping {sec_id} (not found)')
            continue

        output = PODCAST_DIR / f'{sec_id}.mp3'
        transcript = TRANSCRIPT_DIR / f'{sec_id}.txt'

        if args.force_transcript and transcript.exists():
            transcript.unlink()

        if output.exists() and not args.force and not args.force_transcript:
            print(f'  [{sec_id}] Skipping (already exists)')
            continue

        try:
            success = await process_section(sec_id, sections[sec_id], section_names)
            if success:
                generated += 1
        except Exception as e:
            print(f'  [{sec_id}] ERROR: {e}')
            print(f'  [{sec_id}] Skipping, will retry on next run')
            continue

    print(f'\nDone! {generated} podcast episodes generated in {PODCAST_DIR}/')
    print(f'Transcripts cached in {TRANSCRIPT_DIR}/')


if __name__ == '__main__':
    asyncio.run(main())
