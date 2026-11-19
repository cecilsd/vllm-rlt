# Dory traversal-local RoPE reuse — validation evidence

Code: baseline `bed34006ea8dc5b6ba1a3abd99a9fd81bd91a31b`, candidate `99509928f237d2e011d2f4b4181f74cbf07d0a27`.

The production patch changes only the Dory model traversal and related tests. No new flags are required. One sine/cosine pair per active attention type is reused within PRELUDE, each recurrent invocation, and CODA. Default two-loop evaluation count is56->8; Q/K rotation is unchanged in every attention layer.

## Model and measurement

Official NVIDIA Dory4B BF16 revision `ee517dd388a9b348890ae1e00dacc5eb4d0c750d`; model SHA256 in checkpoint-verified.json. One RTX5090; BF16 weights/ordinary activations/KV; existing FP32 intermediate reductions. Software/source versions: environment.json.

The main unprofiled matrix uses two loops,768 KV blocks,16 tokens/block, max sequences4, chunk/token budget128,32 generated tokens, ignoreEOS, greedy; inputs128/B1,2048/B1,128/B4. Three warmed independent processes per arm, interleaved A/B,B/A,A/B. Exact workload text and input construction are in benchmark.py; prompt hashes, all raw outputs, engine drain windows and graph counters are in performance-*.json and correctness-*.json.

Throughput equals output tokens / engine drain window. TTFT and token intervals are engine-side, excluding network. Tail percentiles from short runs are descriptive. Median/ranges are in summary.json. These are finite batch experiments, not saturated server capacity results.

## Results and precision

Eager sync/async throughput median improves3.73%..5.61%, CUDA Graph2.21%..2.50%, sync packed4.49%..4.69% across the tested shapes. These do not establish a universal speedup. Peak allocated memory rises0.12..0.25MiB due to sin/cos lifetime.

All36 matched real-checkpoint correctness requests (432 token IDs),90 performance requests (2880 token IDs), exit depths and finish reasons agree. Teacher-forced logits/hidden/KV compare exactly for8,294,400 tensor elements across1/2/4 loops; selected exact-state counts are in summary.json. Large state dumps stay local; these checks are not a task-quality benchmark. Both arms use the repository tokenizer unchanged and emit the same checkpoint tokenizer-regex warning.

## Complete-request timeline

request-timeline.png aligns both arms at request start on the same time scale. request-decode-zoom.png aligns their first decode step with a fixed6ms window. CPU operator, CUDA runtime, matching GPU stream and RoPE region tracks are shown.

This added diagnostic uses128 input /4 output, two loops, sync eager. It intentionally shortens output length to bound trace size. Each arm warms the exact shape. Separate ops-only/full captures cover the complete request including prefill and all decode, with a final GPU synchronization. Full enables shapes,memory,and Python call-stack events. The three short unprofiled diagnostic repeats are retained in request-*-summary.json but are not included in the primary interleaved32-output speedup claim.

The first-token phase includes final-prompt processing and sampling, not only pure attention prefill. Main->candidate: prefill RoPE64->10, kernels9421->9043; decode RoPE168->24,kernels24186->23178. Combined kernel count33607->32221. Shared math remains per stage; do not interpret these engine schedule totals as the per-token56->8 count.

The ops-only host windows are129.07->101.82ms for prefill and293.68->265.94ms for decode. GPU kernel-time sums are31.49->31.17ms and52.16->51.09ms respectively. Runtime synchronization counts remain1/18, and memcpy counts4/33. Fewer host-dispatched short operators/launches account for the dominant observed difference in these profiled runs; profiler overhead amplifies host cost. Diagnostic duration changes do not quantify production speedup or prove that every non-kernel gap is eliminable. The larger unprofiled matrix supports the speed claim.

The original recurrent-traversal captures show RoPE20->2 and kernels2877->2751. Raw full traces contain Python call/parent events; folded export_stacks was empty in this runtime.

## Trace privacy and replay

All published *.json.gz files are sanitized diagnostic exports: timestamps become relative, process/thread IDs are remapped, host paths and memory addresses are removed, concrete tensor inputs are omitted, and arguments are allow-listed. CPU/CUDA operator names,durations,correlation IDs,shapes,memory byte counts and Python source call/parent events remain. Download/decompress with gzip, then open the JSON in Perfetto or another Chrome trace viewer. Original unsanitized traces remain local and are not published.

## Reproduction

Set up isolated `baseline/` and `candidate/` checkouts at the pinned SHAs, a verified checkpoint at `$EXPERIMENT_ROOT/weights/dory`, and a suitable torch/Triton environment. Select an idle GPU with CUDA_VISIBLE_DEVICES and set PYTHONPATH to the arm checkout. Run each arm in a fresh process.

Engine startup/config: benchmark.py:engine_for and DoryForCausalLM.from_pretrained(...,dtype=torch.bfloat16,loops=2).

Single/batched engine request matrix:
```
python benchmark.py --root "$EXPERIMENT_ROOT" --arm baseline --phase correctness
python state_probe.py --root "$EXPERIMENT_ROOT" --arm baseline
python benchmark.py --root "$EXPERIMENT_ROOT" --arm baseline --phase performance --round 0
```
Repeat candidate and rounds0/1/2 with arm order A/B,B/A,A/B (`--round 0`, with a space). Each performance invocation includes B1 and B4 cases. Keep warmup separate from measured runs.

Diagnostic profiles:
```
python profile_probe.py --root "$EXPERIMENT_ROOT" --arm baseline
python request_profile.py --root "$EXPERIMENT_ROOT" --arm baseline
```
Repeat with candidate. Analyze preserved results using analyze.py and plot_timeline.py.

## Tests and scope

Linux pre-commit run --all-files passes all3 hooks (log included). Final Linux full CPU result1225passed/873skipped (834 GPU). Related supported GPU selection201passed/15deselected. The expanded selection has15 identical baseline/candidate failures:12 unsupported pinned PagedFA4/SM120 combinations and3 existing key_scale fake-cache fixtures; IDs in baseline-failures.json. This is not a full GPU pass.

Latest Dory/speculation/shared-RoPE selection122passed. Windows other-model regression baseline/candidate both98passed/18skipped/1identical Huginn numeric tolerance failure. Other-model implementations and shared RoPE/runner paths are unchanged.

Not qualified by these experiments: other GPUs/backends, FP16 real-weight GPU, real-weight self-speculation, all KV layouts, live service tails or task-quality improvements. Scratch experiment directories were removed and GPU memory returned idle after artifacts were verified.
