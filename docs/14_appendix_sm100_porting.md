# A4: Porting Optimized SM90 (H100) Kernels to SM100 (B200)

*来源: [@_xjdr on X](https://x.com/_xjdr/status/1913703210004648385)*

本文介绍如何将高度优化的CUDA内核从NVIDIA的Hopper架构(SM90, 如H100 GPU)移植到下一代Blackwell架构(SM100, 如B200 GPU)。

---


## Introduction

Porting highly optimized CUDA kernels from NVIDIA’s Hopper architecture (SM90, e.g. H100 GPU) to the next-generation Blackwell architecture (SM100, e.g. B200 GPU) requires understanding new hardware instructions and memory hierarchies. This guide focuses on transformer inference kernels such as FlashMHA and DeepGEMM (from DeepSeek-AI) that rely on advanced tensor-core operations. We will map key SM90 instructions (like warp-group MMA and asynchronous copy) to their SM100 equivalents, and provide guidance on achieving full correctness and peak performance on SM100. The guide covers the new **Tensor Core 5th Generation (TCGEN05)** instructions, the **Tensor Memory (TMEM)** hierarchy and Tensor Memory Accelerator (TMA) usage, architectural changes in SM100 (SM count, warp schedulers, shared memory, etc.), relevant updates in CUTLASS/CUTE and CUDA 12, and practical tips for avoiding memory hazards and debugging synchronization issues. Throughout, we include low-level examples (PTX or intrinsics) to illustrate how to load tiles, perform computation, and store results on SM90 vs SM100.

## SM90 vs. SM100 Architecture Overview
**Fifth-Generation Tensor Cores:** The SM100 architecture introduces NVIDIA’s 5th-generation tensor cores (TCGEN05), which are significantly larger and faster than the 4th-gen cores in SM90 (Hopper). Empirical analysis suggests SM100 tensor cores operate as **128×128 systolic arrays** – in other words, to fully utilize them, your GEMM tile dimensions for **M** and **N** should be 128 or greater (in multiples of 128). Smaller tiles (e.g. 64×64) will only achieve a fraction of peak throughput (about 25% in the 64×64×64 case) (ThunderKittens Now Optimized for NVIDIA Blackwell GPUs
). In practice, this is a departure from H100, where even smaller shapes could saturate the tensor cores. Consequently, SM100 kernels should use larger thread-block tiles or **cooperate multiple warps (or CTAs) on a single GEMM** to reach 128×128 output sizes whenever possible. The new tensor cores also achieve roughly **2.0–2.5× higher FMA throughput** than H100’s at equivalent clock (ThunderKittens Now Optimized for NVIDIA Blackwell GPUs
), delivering a large boost in performance if fed with optimal tile sizes.
**SM Count and Warp Schedulers:** SM100 GPUs prioritize per-SM capability over SM count. Compared to H100 (which has up to 132 SMs), an SM100-based GPU may have fewer SMs, but each SM is more powerful. Each SM houses additional warp schedulers and execution resources to handle more warps concurrently. (Exact SM counts vary with specific B200 SKUs, but the trend is **fewer, beefier SMs**.) For example, if H100 had 6 schedulers per SM, SM100 might feature 8 or more warp schedulers, allowing more warps active per SM to hide latency. The **warp-group** concept from Hopper (128 threads = 4 warps executing a tensor operation together) is extended in SM100: new *CTA-wide* or even *multi-SM* warp groups are possible (discussed below under CTA pairs). More schedulers and larger tensor cores mean each SM100 can handle a larger thread-block doing more work. As a result, you may need to adjust kernel launch parameters – e.g. using larger thread blocks (more threads per block) or thread-block clusters – to best occupy an SM100. Occupancy on SM100 will often be limited by available shared memory or registers per block rather than warp count, given the expanded per-SM resources. It’s advisable to experiment with **fewer, larger thread blocks** (possibly 1–2 CTAs per SM) to leverage the full compute capacity, rather than many small blocks which might underutilize each SM.
**Shared Memory and Tensor Memory:** SM100 retains a large shared memory (on the order of ~227 KB per SM, similar to H100’s 228 KB (NVIDIA Hopper Architecture In-Depth | NVIDIA Technical Blog
)) for general thread block usage. In addition, SM100 introduces a new **Tensor Memory (TMEM)** space of roughly **256 KB per SM** dedicated to tensor core operations (ThunderKittens Now Optimized for NVIDIA Blackwell GPUs
). This is effectively a new layer of on-chip memory distinct from the traditional shared memory and register file. TMEM serves as a high-speed buffer for matrix tiles used by tensor instructions – it allows intermediate results (like partial GEMM accumulators) to reside on-chip across warp groups or even across CTAs in a cluster. Think of TMEM as an extension of the register file that’s specialized for large matrices. Using TMEM means that results of one tensor core operation can be consumed by subsequent operations without round-tripping through global memory or even standard shared memory, which is key for pipeline efficiency. We will delve into how to allocate and use TMEM in the next sections.
**Thread Block Clusters and CTA Pairs:** Hopper introduced **Thread-Block Clusters**, enabling limited coordination between independent CTAs (thread blocks) on the same GPC (General Processing Cluster) – for example, clusters could share data via multicast and use cluster-level barriers. SM100 goes further by allowing two thread blocks to effectively act as a tightly-coupled pair on the **same SM**. These are referred to as **CTA pairs** (or CTA groups) (ThunderKittens Now Optimized for NVIDIA Blackwell GPUs
) (ThunderKittens Now Optimized for NVIDIA Blackwell GPUs
). A CTA pair consists of two CTAs in the same cluster that can coordinate at an even deeper level – they can *simultaneously execute a single tensor core instruction spanning both CTAs’ data*, and they share access to each other’s TMEM regions (ThunderKittens Now Optimized for NVIDIA Blackwell GPUs
) (ThunderKittens Now Optimized for NVIDIA Blackwell GPUs
). In other words, up to **two CTAs can collaborate on one larger GEMM**, splitting the work but using the fused TMEM/TMA resources of both. This is exposed in PTX via the **cta_group::1** vs **cta_group::2** modifiers on tensor core instructions, where cta_group::2 engages two SMs (or two CTAs) for the operation (Help needed to explain tcgen05.mma_cta_group instructions - CUDA Programming and Performance - NVIDIA Developer Forums
). Practically, this means if one CTA holds part of matrix A and the other holds part of B, both CTAs’ data can be multiplied as one operation, utilizing up to 8 Tensor Cores (4 per SM) in unison (Help needed to explain tcgen05.mma_cta_group instructions - CUDA Programming and Performance - NVIDIA Developer Forums
). The CTA pair mechanism helps achieve the 128×128 tile sizes for peak throughput when a single CTA alone might be too resource-limited to do so. Developers can leverage CUDA cluster launch (with cudaLaunchCooperativeKernelMultiDevice or new cluster APIs) to schedule CTA pairs. We will later discuss how to manage synchronization in such cases (see **Cluster Synchronization and Barriers**).
**Precision Format Support:** Both SM90 and SM100 support BF16, FP16, TF32, INT8, etc., but SM100 adds *expanded support for FP8* and even experimental 4-bit/6-bit floating types. Hopper’s tensor cores introduced FP8 (E4M3 or E5M2 formats) mainly via software (Transformer Engine); Blackwell’s tensor cores natively support FP8 in matrix ops with hardware-managed scaling factors. In PTX 8.7 (SM100), new data-type suffixes like **.f8f6f4** and **.mxf8f6f4.block_scale** appear, indicating mixed 8/6/4-bit float support with scaling (The Longest Nvidia PTX Instruction | Ash's Blog
) (The Longest Nvidia PTX Instruction | Ash's Blog
). For example, **.mxf8f6f4.block_scale** denotes an MMA that multiplies FP8 matrices with block-wise scaling – the **“mx”** prefix suggests a mixed-precision accumulate (likely FP8 inputs accumulated into FP16/FP32) and “block_scale” means the instruction will apply per-block scale factors for FP8 values. We’ll see how to use these for MLP or MoE kernels that employed block FP8 on H100.

In summary, porting to SM100 means handling **new instructions** (TCGEN05 family), taking advantage of **TMEM/TMA** to keep data on-chip, possibly reorganizing kernel tiling to satisfy larger tensor cores, and using new **synchronization mechanisms** (like CTA pairs and advanced barriers). Next, we map specific SM90 instructions to their SM100 counterparts, which is a crucial first step in the porting process.

## Mapping SM90 Instructions to SM100 Equivalents

Porting the code involves replacing or reworking certain inline PTX and intrinsic calls. The SM100 PTX ISA (8.7) introduces an extensive **TCGEN05** instruction set, which replaces or augments many SM90-era idioms. Below is a mapping of key SM90 instructions/patterns to SM100 equivalents:

- **Warp-Group MMA (****wgmma.mma**
**_async → ****tcgen05.mma**
**)** – On H100, matrix multiplications across a warp-group of 128 threads were issued via wgmma.mma
_async instructions (e.g., wgmma.mma
_async.sync.aligned.m64n64k16.f16 for a 64x64x16 fragment multiply). In SM100, these are replaced by the more general **tcgen05.mma**
 instructions. The new tcgen05.mma
 supports execution at **CTA scope** (not just warp scope), using the cta_group modifiers as discussed. For instance, a Hopper code performing a BF16 matrix multiply-add might use:// SM90 example (PTX-like pseudocode)
```cpp
wgmma.mma
```
_async.sync.aligned.m64n64k16.bf16.accumulator{...} [dest_reg_fragments], [a_smem_addr], [b_smem_addr];
In SM100, the equivalent could be:// SM100 example (PTX-like pseudocode)
```cpp
tcgen05.mma
```
.cta_group::1.sync.aligned.bf16 [%tmem_ptr], %descA, %descB, %phase, p;
Here, %descA and %descB are tensor map descriptors for operand A and B (similar to what was encoded in wgmma’s use of shared memory pointers), and [%tmem_ptr] is a **Tensor Memory** destination where the accumulator (matrix D) will reside. The p is a predicate if needed (for conditional execution). The SM100 MMA has more variants: **.mma.sp** (sparse matrix A support), **.mma.ws** (per-warp split?), etc. (The Longest Nvidia PTX Instruction | Ash's Blog
) (The Longest Nvidia PTX Instruction | Ash's Blog
), but the base .mma covers dense matrix multiply-add. Key differences:
*Scope:* SM90 wgmma was implicitly a warp-group operation on one SM. SM100 tcgen05.mma
 can operate on one CTA (similar scope) or a CTA pair (cta_group::2) across two SMs (Help needed to explain tcgen05.mma_cta_group instructions - CUDA Programming and Performance - NVIDIA Developer Forums
).
*Destination:* SM90 writes the result into a **register fragment** (accumulator fragment per thread). SM100 can write directly into **TMEM** (by specifying a TMEM pointer as the destination). This means accumulation happens in a TMEM tile that is accessible by all threads in the CTA/cluster, rather than staying private in registers. It offloads the accumulation storage to TMEM. We will later see how that affects the pipeline (e.g., needing to load out from TMEM to registers at the end).
*Data types and modifiers:* SM100 tcgen05.mma
 supports a superset of types. For example, to handle **FP8 with block scaling**, you might see a PTX like:tcgen05.mma
.cta_group::2.kind::mxf8f6f4.block_scale [%tmem], %descA, %descB, %phase, [%scaleA], [%scaleB], p;
which multiplies FP8 matrices A, B (with format like e4m3/e5m2 represented as f8f6f4) and uses block scale factors located at [%scaleA] and [%scaleB] in memory (Help needed to explain tcgen05.mma_cta_group instructions - CUDA Programming and Performance - NVIDIA Developer Forums
). On H100, implementing block FP8 required manual scaling (e.g., each tile loaded as int8 then multiplied by a scale factor in code). In SM100, the *tensor core can directly apply the scale factors* if provided in TMEM or shared memory (Help needed to explain tcgen05.mma_cta_group instructions - CUDA Programming and Performance - NVIDIA Developer Forums
). This greatly simplifies FP8 ports – instead of performing separate element-wise multiply for scaling, you supply the precomputed scales to the MMA instruction (as extra operands or via descriptors). We will cover how to prepare these operands in the TMEM/TMA section.

- **Asynchronous Copy (cp.async.bulk.tensor → TMA load/store via tcgen05.cp):** H100 introduced the **Tensor Memory Accelerator (TMA)** engine for bulk data movement between global memory and shared memory, accessible in PTX as cp.async.bulk.tensor instructions (with variants for 1D–5D copy regions). For example, an H100 kernel might use cp.async.bulk.tensor.2d to asynchronously copy a 2D tile of matrix from global to shared memory, using a descriptor for the source layout (User Guide for NVPTX Back-end — LLVM 21.0.0git documentation
) (User Guide for NVPTX Back-end — LLVM 21.0.0git documentation
). On SM100, the TMA mechanism is still present but the interface is now part of the TCGEN05 family:**tcgen05.cp** – This instruction triggers an asynchronous copy of tensor data *into* or *out of* Tensor Memory. It effectively replaces cp.async.bulk.tensor on SM100 (which targeted shared or cluster memory on H100) with a variant that targets TMEM or shared memory via TMEM allocators. The LLVM NVPTX backend describes tcgen05.cp intrinsics corresponding to PTX that use TMEM (The Longest Nvidia PTX Instruction | Ash's Blog
). You will typically use tcgen05.cp after allocating a region of TMEM or obtaining a pointer to a shared memory tile. For instance:// Pseudocode: schedule asynchronous copy of A tile from global to TMEM
```cpp
cuda::memcpy_async(dest_tmem_ptr, src_global_ptr, desc, barrier);
// In PTX this would lower to something like:
tcgen05.cp.async.cluster.global.to
```
.shared::cta [...]
(Exact PTX syntax elided for brevity – the PTX doc shows variants with multicast and cache hint flags as well.)In practice, high-level CUDA 12+ APIs or CUTE/CUTLASS abstractions hide the direct PTX. For example, CUTLASS 3.9 provides TMA load utilities that behind the scenes emit tcgen05.cp and related synchronization. In porting, you will replace sequences like:// SM90 style using cp.async:
```cpp
cp.async.bulk.tensor.2d.shared::cluster(...); // copy tile into shared memory
```
... // use mbarrier to synchronize
with:// SM100 style using TMA:
```cpp
TensorCopyDesc descA = makeTensorCopyDesc(...); 
cute::copy_tma(descA, gmem_ptr_A, tmem_ptr_A, barrier); // conceptual example
```
The above line would configure and launch a TMA transfer to TMEM (or to shared memory). Under the hood, this would invoke tcgen05.cp. The **source or destination can be TMEM or shared** depending on the use case – typically for loading matrix A/B you copy from global (GMEM) to shared memory, then use tcgen05.mma
 to read from shared; but it is conceivable to copy directly to TMEM if the MMA can read from TMEM (currently, MMA reads from registers or shared, and writes to TMEM, as we’ll detail).**TMA Store:** SM100’s TMA is bidirectional like on H100 ([Deep Dive on the Hopper TMA Unit for FP8 GEMMs | PyTorch](https://pytorch.org/blog/hopper-tma-unit/#:~:text=TMA%20is%20an%20H100%20hardware,This%20is%20termed%20%E2%80%98multicast%E2%80%99
)). That means you can also asynchronously write results from shared memory to global memory. In PTX, this might be another form of tcgen05.cp or a dedicated tcgen05.st
 (store) instruction. The PTX ISA documentation enumerates tcgen05.ld and tcgen05.st
 as specialized load/store for tensor memory (User Guide for NVPTX Back-end — LLVM 21.0.0git documentation
) (User Guide for NVPTX Back-end — LLVM 21.0.0git documentation
). However, those ld/st specifically refer to moving data between TMEM and registers or shared (not global). For global stores, one would still use the copy engine. In CUTLASS, you might call something like cute::copy_tma(output_desc, tmem_ptr_output, gmem_ptr_output, barrier) to push a matrix from on-chip memory to global asynchronously. On H100, one might not have used an async store (often the result was small enough to just do a ld.shared / st.global
 loop). But on SM100, since results can reside in TMEM, you have the option to use TMA to stream them out while the next computation proceeds, overlapping output writes with compute. We recommend using TMA store if output size is large, to overlap that latency – the programming model is similar to TMA load.**Memory addressing differences:** The cp.async.bulk.tensor on SM90 required providing a *tensor descriptor* (defining dimensions, strides, etc.) and coordinates, plus a **barrier** to signal completion (User Guide for NVPTX Back-end — LLVM 21.0.0git documentation
) (User Guide for NVPTX Back-end — LLVM 21.0.0git documentation
). On SM100, tcgen05.cp also uses a descriptor (likely prepared via new intrinsics or the CUDA Graph API). It still supports up to 5D copies, multicast within a cluster, and various cache hints (these features carry over, just accessed via new instructions) (NVIDIA Hopper Architecture In-Depth | NVIDIA Technical Blog
) (NVIDIA Hopper Architecture In-Depth | NVIDIA Technical Blog
). The big difference is that the **destination can be Tensor Memory**. Typically, you will:
Allocate a TMEM tile for, say, matrix **D** (the accumulator/output tile) – we cover allocation shortly.
Use one or more tcgen05.cp instructions to bring matrix **A** and **B** tiles into shared memory (or registers) for the compute. (In current usage, TMEM is mainly used for outputs/accumulators, not as the primary buffer for A/B, because the tensor core reads A/B from shared memory.)
After computation, use tcgen05.cp (or tcgen05.ld + traditional store) to move the result out of TMEM to global memory.Summarizing naming: You might see the term **“TMA load/store”** to describe these operations. In code, this could correspond to tcgen05.cp (with arguments indicating gmem→smem or smem→gmem direction), or high-level calls like cuda::memcpy_async which internally choose the right PTX. The important conceptual shift is treating bulk copies as **asynchronous tasks offloaded to hardware** with explicit management (barriers or waits) for when the data is ready, rather than manual loop loads/stores in the kernel.

- **Asynchronous Barrier & Synchronization (mbarrier → tcgen05.wait, tcgen05.commit, etc.):** On SM90, coordinating asynchronous copies and compute was done with **memory barriers (mbarrier)**. For example, one would use cp.async to start copies, then later issue cp.async.wait_group N or use mbarrier.await to wait for N previous copy operations to complete, and finally mbarrier.arrive or bar.sync to signal threads can proceed (NVIDIA Hopper Architecture In-Depth | NVIDIA Technical Blog
) (NVIDIA Hopper Architecture In-Depth | NVIDIA Technical Blog
). SM100 still provides these mechanisms but also adds new **tcgen05-specific sync instructions** that simplify certain patterns:**tcgen05.wait** – This instruction serves as a fine-grained wait for the completion of prior pipelined operations issued by the same thread (or warp). The PTX ISA defines both *mbarrier-based completion* and *tcgen05.wait-based completion* as options for implicitly pipelined instructions (Contents — PTX ISA 8.7 documentation
) (Contents — PTX ISA 8.7 documentation
). In essence, if you launch a series of tcgen05.cp or tcgen05.mma
 that operate asynchronously, you can insert tcgen05.wait to stall until the needed data is ready. Unlike a full barrier that syncs all threads, tcgen05.wait is more like a **per-thread or per-task wait** – think of it as waiting on an “event” that the hardware triggers when a certain stage finishes. For example, if one warp initiates a TMA load, that same warp (or a designated thread in it) might later do tcgen05.wait to ensure the copy finished before proceeding to use the data.**tcgen05.commit** – Another new instruction in PTX 8.7, likely used to commit the results of asynchronous operations. For instance, after issuing an async copy to TMEM, one might use tcgen05.commit to indicate that the data in TMEM is ready for consumption (or to finalize an accumulator state before another warp uses it). The combination of commit and wait in TCGEN05 can supplant the older sequence of cp.async.commit_group and cp.async.wait_group. Essentially, **tcgen05.commit marks a transaction complete, and tcgen05.wait waits for it** (Contents — PTX ISA 8.7 documentation
) (Contents — PTX ISA 8.7 documentation
).**tcgen05.fence** – A memory fence for ordering, analogous to membar instructions. This ensures all memory operations (LD/ST, TMEM ops) issued before the fence are visible before those after. You might use tcgen05.fence if you mix standard memory operations with TMEM operations to avoid reordering hazards. For example, if you do some regular shared memory stores and then a tcgen05.cp that reads shared, a fence can ensure the stores are seen by the copy.**mbarrier and ClusterBarrier:** The classic use of mbarrier (memory barrier object) is still valid. In CUDA 12, there are C++ APIs like cuda::barrier<cuda::thread_scope_block> which implement an arrive/wait pattern that maps to these hardware barriers. On SM100, if you prefer, you can continue to use an **mbarrier** to coordinate between threads: one thread (or warp) does mbarrier.arrive when it finishes an async copy or finishes producing data, and other threads do mbarrier.wait to block until the count is reached (NVIDIA Hopper Architecture In-Depth | NVIDIA Technical Blog
) (NVIDIA Hopper Architecture In-Depth | NVIDIA Technical Blog
). The SM100 adds an **asynchronous transaction barrier** that accelerates this mechanism at hardware level (NVIDIA Hopper Architecture In-Depth | NVIDIA Technical Blog
) (NVIDIA Hopper Architecture In-Depth | NVIDIA Technical Blog
). For cross-CTA (cluster-wide) sync, there is **ClusterBarrier** available (via cooperative launch or NVPTX barrier.cluster instructions), which can synchronize threads across CTAs in a cluster – for instance, if using CTA pairs on two SMs, you may need a cluster barrier at certain points. We will detail best practices for using these barriers in a later section (Debugging & Hazards).In summary, **SM90’s cp.async...; ... mbarrier.wait/bar.sync sequence maps to SM100’s tcgen05.cp...; ... tcgen05.wait/commit** (or an equivalent high-level barrier). Many of these details are abstracted if you use C++20 <cuda/barrier> or CUTLASS’s pipeline abstractions. For example, CUTLASS 3.9 defines a cute::Barrier that wraps tcgen05.wait in its arrive_wait() method ([BUG]Is tcgen05.fence supported by Cutlass-3.8.0 ? · Issue #2098 · NVIDIA/cutlass · GitHub
) ([BUG]Is tcgen05.fence supported by Cutlass-3.8.0 ? · Issue #2098 · NVIDIA/cutlass · GitHub
). When porting, ensure that any mbarrier usage in your SM90 code is updated to either use the new barrier primitives or is correctly mapped to the underlying PTX. If you had inline PTX for Ampere/Hopper barriers (like mbarrier::complete_tx etc.), these should be replaced by or used in conjunction with the new TC instructions.

The table below summarizes the mapping:

- *Matrix Multiply:* **SM90:** wgmma.mma
_async → **SM100:** tcgen05.mma
 (with CTA group and new type modifiers) (The Longest Nvidia PTX Instruction | Ash's Blog
) (Help needed to explain tcgen05.mma_cta_group instructions - CUDA Programming and Performance - NVIDIA Developer Forums
).

- *Async Copy (global→shared/cluster):* **SM90:** cp.async.bulk.tensor.{1D-5D} → **SM100:** tcgen05.cp (TMA load to TMEM/shared) (NVIDIA Hopper Architecture In-Depth | NVIDIA Technical Blog
) (NVIDIA Hopper Architecture In-Depth | NVIDIA Technical Blog
).

- *Async Copy (shared→global):* **SM90:** (no direct analog, used loop or cp.async if supported) → **SM100:** tcgen05.cp or tcgen05.st
 (TMA store from TMEM/shared).

- *Async Sync Barrier:* **SM90:** cp.async.wait_group + mbarrier or bar.sync → **SM100:** tcgen05.wait + tcgen05.commit (or use cuda::barrier which under the hood uses these) (Contents — PTX ISA 8.7 documentation
) (Contents — PTX ISA 8.7 documentation
).

- *Memory Fence:* **SM90:** membar.gl/membar.cta
 as needed → **SM100:** tcgen05.fence (for ordering TMEM ops with others) (The Longest Nvidia PTX Instruction | Ash's Blog
).

- *Cross-CTA sync:* **SM90:** barrier.cluster (Hopper introduced) → **SM100:** similar, plus CTA pair specific sync if needed (often handled by hardware when using cta_group instructions, but explicit cluster barrier can be used around those).

By applying these mappings, you address the primary differences at the instruction level. Next, we delve deeper into **Tensor Memory (TMEM)** and how to structure pipelines around it, since that is a fundamental feature of SM100 that underpins these new instructions.

## Tensor Memory (TMEM) and TMA Pipeline on SM100

One of the biggest architectural changes from SM90 to SM100 is the introduction of **Tensor Memory (TMEM)** and its integration with the TMA copy engine. Proper use of TMEM is key to achieving high performance on SM100, as it enables the async pipeline of load-compute-store to run with minimal stalling. In this section, we explain how to allocate and use TMEM, the rules for addressing it, how it interacts with shared memory and TMA, and how to manage synchronization with TMEM in mind.
**What is TMEM?** TMEM is a chunk of on-chip memory dedicated to storing tensor data for the new TCGEN05 operations. It can be thought of as a special **tile buffer** that exists outside of the standard shared memory (but still on the SM). The SM100 PTX defines TMEM as **address space 6** (whereas shared memory is address space 3) and uses 32-bit offsets for it (User Guide for NVPTX Back-end — LLVM 21.0.0git documentation
) (User Guide for NVPTX Back-end — LLVM 21.0.0git documentation
). Each CTA (or CTA pair) can dynamically allocate sections of TMEM at runtime to hold matrices. Notably, TMEM can be larger than the register file of any single warp, enabling large accumulator matrices to persist. As mentioned earlier, roughly 256 KB per SM is available for TMEM (ThunderKittens Now Optimized for NVIDIA Blackwell GPUs
), which is on par with (or in addition to) the shared memory size.
**Allocating TMEM:** In PTX, TMEM is managed with **tcgen05.alloc and tcgen05.dealloc** instructions. In CUDA C++, these are exposed via NVVM intrinsics and likely wrapped by libraries. A simplified view:

- tcgen05.alloc.cg1 allocates a TMEM region for the current CTA (cta_group of size 1) (User Guide for NVPTX Back-end — LLVM 21.0.0git documentation
). There is also tcgen05.alloc.cg2 for a CTA pair (cta_group of 2 CTAs) (User Guide for NVPTX Back-end — LLVM 21.0.0git documentation
). The .shared variant means the resulting pointer is written to shared memory (address space 3) instead of a register (User Guide for NVPTX Back-end — LLVM 21.0.0git documentation
); in practice, this just dictates where the pointer is stored for use by threads.

- The allocation takes a parameter ncols which is the number of **columns** to allocate in TMEM (User Guide for NVPTX Back-end — LLVM 21.0.0git documentation
). TMEM is conceptually organized as a 2D matrix of a certain height (the hardware-defined number of rows) and the requested number of columns. **ncols must be a power of two (****User Guide for NVPTX Back-end — LLVM 21.0.0git documentation**
**)**. Think of columns as the “width” of your allocated tile in TMEM, each column being 128 bytes (for example) – the exact granularity isn’t explicitly stated, but .aligned.b32 suggests 32-byte alignment (User Guide for NVPTX Back-end — LLVM 21.0.0git documentation
). If you allocate 8 columns, that might be 8 * 32B = 256B wide tile, etc.

- The alloc writes the base address of the allocated TMEM tile into a specified destination (usually a variable in shared memory that all threads can read, or a register for the issuing thread) (User Guide for NVPTX Back-end — LLVM 21.0.0git documentation
). This base address is what you use as the %tmem_ptr in tcgen05.mma
 and other instructions.

- Correspondingly, tcgen05.dealloc frees the TMEM region when you’re done (to avoid leaks if you allocate multiple times in a CTA) (User Guide for NVPTX Back-end — LLVM 21.0.0git documentation
). Usually you’d deallocate at the end of the kernel or after finishing a series of operations using that tile.

- There’s also tcgen05.relinquish_alloc_permit which is used in multi-CTA scenarios: it signals that a CTA is giving up its right to allocate TMEM further (User Guide for NVPTX Back-end — LLVM 21.0.0git documentation
). This is relevant in clusters where perhaps only one CTA pair at a time can allocate; others must wait or relinquish if done.

**Memory addressing and layout in TMEM:** TMEM is only accessible via the TCGEN05 instructions – you **cannot directly dereference a TMEM pointer in normal LD/ST instructions** (User Guide for NVPTX Back-end — LLVM 21.0.0git documentation
). Instead, you must use tcgen05.ld or tcgen05.st
 to move data between TMEM and registers/shared memory (User Guide for NVPTX Back-end — LLVM 21.0.0git documentation
). This restriction ensures that only the tensor core pipeline manipulates TMEM, preserving coherence. The **load/store width and alignment** are fixed (likely 128-bit or 256-bit segments). When you allocate n columns, you essentially get a tile that can hold matrices up to that width. The height of the tile is implicitly determined by the operation – for example, if you do a 128x128 MMA, the TMEM tile for the result D will be 128 (rows) × 128 (cols) elements in size (for FP32 accumulators). If ncols was insufficient to cover 128 columns, the hardware would either fail the allocation or you’d get an out-of-bounds error when writing. Thus, it’s crucial to allocate enough columns for your largest needed matrix. A good rule is to allocate to the next power of two of the matrix width. For instance, to store a 128×128 FP32 accumulator (which is 128 columns of FP32), request ncols=128 (already power of 2). For a 80×80 matrix, request 128 columns as well (since next power of 2 is 128). **Alignment**: The .aligned.b32 on alloc suggests the TMEM address returned is 32-byte aligned, and you should adhere to that for any direct loads/stores (User Guide for NVPTX Back-end — LLVM 21.0.0git documentation
). Typically, you won’t worry about manual alignment because you allocate whole columns.
**Using TMEM in the Kernel Pipeline:** Here’s how a typical load-compute-store pipeline might look on SM100, contrasted with SM90, for a GEMM inside a single CTA:

- **Prologue (Data Load):***SM90 (H100):* Allocate shared memory buffers for matrices A and B (if not static). Use cp.async.bulk.tensor to stream a tile of A from global memory to shared memory, and similarly for B. Use a circular buffer of e.g. 4 staged copies (and cp.async.wait_group to ensure at least one tile is ready). Then use wgmma.mma
_async to consume A and B from shared. Accumulate result in registers.
*SM100 (B200):* Allocate a TMEM tile for the result D using tcgen05.alloc. Also allocate shared memory for A and B tiles (still needed because the MMA reads A/B from SMEM). Launch TMA copies for A and B: one thread (or warp) uses tcgen05.cp to bring in A’s tile and B’s tile into shared memory. We can use one TMEM barrier (or the same one) to track these copies. For example, thread0 does:__shared__ cuda::barrier<cuda::thread_scope_block> tmaBarrier;
```cpp
if (threadIdx.x == 0) {
 cuda::memcpy_async(smemA, globalA + offset, descA, tmaBarrier);
 cuda::memcpy_async(smemB, globalB + offset, descB, tmaBarrier);
```
}
```cpp
// ... each copy will signal tmaBarrier on completion
Then ensure data is ready: either thread0 does tmaBarrier.arrive_and_wait() after issuing, or all threads do something like:__syncthreads(); // optionally ensure thread0 done issuing
cuda::barrier_arrive_relaxed(tmaBarrier, 2); // each copy adds one arrival
cuda::barrier_wait(tmaBarrier); // wait until 2 arrivals (A & B loaded)
```
At this point, A and B tile are in shared memory and ready.

- **Compute (MMA):***SM90:* Once A and B are in SMEM, all warp(s) execute wgmma.mma
_async to multiply and accumulate into registers. If the GEMM is large, this is done in a loop over K dimension chunks, ping-ponging between shared-memory double buffers and using wgmma.mma
```cpp
_async each iteration. Synchronization between stages is via __syncthreads() or mbarrier as appropriate to ensure new data is loaded before next MMA.
```
*SM100:* Once A and B are ready, perform the tensor core operation. Here we use tcgen05.mma
 with the TMEM pointer for D. Important: in SM100, the MMA can be issued by a **single warp** on behalf of the whole CTA (if using CTA-wide MMA) – not every warp should call it, or it would be duplicated. In CUTLASS’s scheme, typically one warpgroup (say warp 0-3 in a CTA) is designated as the “producer” of the MMA results, and only one thread (or one warp) actually issues the tcgen05.mma
 instruction with cta_group modifier, which then internally uses all required warps’ data. For example, ThunderKittens uses cute::elect_one_sync() to have only one elected thread execute the CTA-level MMA PTX (Help needed to explain tcgen05.mma_cta_group instructions - CUDA Programming and Performance - NVIDIA Developer Forums
```cpp
). So in your ported code, guard the inline PTX or intrinsic call so that it’s executed exactly once per CTA (or per CTA pair). For instance:if (threadIdx.x == 0) { // assume warp 0, lane 0 issues the CTA-wide MMA
 asm volatile("tcgen05.mma
```
.cta_group::1.sync.aligned.bf16 [%0], %1, %2, %3, p;\n"
 : /* outputs */ 
 : "r"(tmem_D_base), "l"(descA_tiled), "l"(descB_tiled), "r"(phase), "r"(pred) );
}
The above is pseudo-instruction – normally you’d use intrinsics or CUTLASS C++ function to handle descriptors. Note the use of cta_group::1 (single CTA). If we had a CTA pair, each CTA would have one thread do this, likely with cta_group::2, allowing hardware to coordinate across two SMs. After the tcgen05.mma
 is launched, the actual multiply-accumulate happens asynchronously: the hardware will read A from shared mem, B from shared mem, perform the FMA on the tensor cores, and write the result into the TMEM tile at %0 (tmem pointer). All threads in the CTA can continue executing while this is in progress, except those that are needed for the MMA (they might be stalled internally). Usually you’d immediately proceed to the next steps of loading the next tile or doing partial reduction in parallel, etc., overlapping with the MMA latency. The design of TMEM allows the accumulator to remain on chip even if we don’t immediately use it in registers.If the GEMM is split in K, you would loop: each iteration loads next A, B into shared (possibly while previous MMA is running), then issue another tcgen05.mma
 with .kind::accumulate or similar flag to accumulate into the same TMEM tile. In PTX, this might be implicit – the same %tmem_ptr used again will accumulate rather than overwrite, as long as you don’t re-initialize it. (Alternatively, there may be a flag or using tcgen05.mma
 vs tcgen05.mma
.sp or mma.ws
 for accumulate vs writeback – but details aside, the common usage is multiple MMA instructions accumulate partial sums in the TMEM D tile.)

- **Epilogue (Store/Reduction):***SM90:* After the loop, each thread had a portion of the accumulator in registers. The epilogue usually involved doing per-element operations (e.g., applying activation, or in FlashMHA, computing softmax on the output of Q*K before multiplying by V). For a plain GEMM, the epilogue would store the result to global memory: each thread writes out its registers to the output matrix in global memory (through shared mem or directly). If using cp.async for store, one warp could coalesce a write of the tile from shared to global and use cp.async.signal to indicate completion, then mbarrier.wait. However, many kernels just did a synchronized st.global
 since writing output is often the last step.
*SM100:* Now the result is sitting in TMEM (shared by the CTA). We have two main options to retrieve it:
**Direct store from TMEM:** Use the TMA engine to copy from TMEM to global memory. Since tcgen05.cp can handle shared memory as a source, and we can treat TMEM similarly, one could imagine an instruction or sequence like tcgen05.cp.shared.to.global
 [%tmem_ptr] -> [global_addr]. In practice, because TMEM is not directly addressable by normal instructions, we might first issue a tcgen05.ld to load chunks of TMEM into registers or shared memory, then do a normal global store or a cp.async from shared to global. There is indication that tcgen05.st
 can write from registers to TMEM, and by symmetry possibly from TMEM to memory, but typically the expected path is: **TMEM → (LD to registers) → ST to global** or **TMEM → (LD to shared) → cp.async to global**.**Convert TMEM to registers (or shared) then store:** The tcgen05.ld instruction will load a portion of a TMEM tile into registers (User Guide for NVPTX Back-end — LLVM 21.0.0git documentation
). For example, a warp can use tcgen05.ld to read a 128-byte segment (one “column” or part of it) from the TMEM tile into 4 registers. By iterating, threads can gather the full matrix D into registers or shared memory. Once in registers, you can perform any epilogue computation needed (like scaling, bias add, activation). For instance, in FlashMHA, after computing the attention scores (Q*K), one would need to apply softmax: this would entail reading the scores out of TMEM (maybe one row per thread), computing exp and normalization in FP32 in the threads, then writing the normalized scores (or directly computing the next GEMM with V). If the next GEMM (scores*V) is also on tensor cores, you might skip writing out and instead feed the scores into another MMA by first moving them to shared memory and having the tensor core consume them. In fact, advanced pipelines can overlap this – e.g., one warp (“consumer”) starts reading D from TMEM and doing softmax while another warp (“producer”) is loading the next A, B or computing next partial in TMEM. This concurrency is the “only one bubble” pipeline described in ThunderKittens (ThunderKittens Now Optimized for NVIDIA Blackwell GPUs
) (ThunderKittens Now Optimized for NVIDIA Blackwell GPUs
```cpp
).To keep things simple: let’s assume we just want to store the result to global. We can have a warp (or all warps collectively) load the TMEM tile into a shared memory buffer. For example:__syncthreads(); // ensure MMA complete (or use tcgen05.wait if one thread needs to wait)
if (warpId == 0) { // first warp reads out
 for(int i=laneId; i < tile_elems; i += 32) {
```
 output_smem[i] = tcgen05_ld(tmem_D_base + i*elementSize);
 }
}
```cpp
__syncthreads();
// now output_smem has the matrix
```
store_global(output_ptr, output_smem, output_desc);
In practice, you’d use vectorized loads and stores. The tcgen05.wait or barrier would ensure the MMA writes to TMEM are finished before reading. In PTX, reading from TMEM might involve an explicit tcgen05.wait if the MMA was asynchronous to the reading thread (or the hardware might handle hazard tracking if we use the same issuing thread).Alternatively, we could ask one thread to directly do a TMA store: cuda::memcpy_async(globalDst, tmem_ptr_D, descOut, tmaBarrier). If such an overload exists, it would use the hardware to read TMEM and write to GMEM. It’s likely that currently the recommended method is to stage through shared memory or registers because TMEM is not directly accessible by the copy engine without a load. The PTX tcgen05.cp seems to require a source in memory address space (global or shared), and TMEM might not qualify unless we do tcgen05.relinquish_alloc_permit which possibly makes the TMEM visible as shared? (That is speculative.)

**Interaction of TMEM with TMA and SMEM:** It’s important to clarify how TMEM, TMA, and SMEM interplay:

- TMEM is allocated and addressed within a CTA (or CTA pair). All threads in the CTA pair know the base address via a shared memory pointer written by alloc. For CTA pairs (cta_group::2), the TMEM may span two SMs; each CTA likely allocates its part and the hardware treats them as one logical contiguous TMEM space. The ThunderKittens blog confirms that in CTA pairs, each CTA can access the other’s TMEM (ThunderKittens Now Optimized for NVIDIA Blackwell GPUs
) (ThunderKittens Now Optimized for NVIDIA Blackwell GPUs
).

- Data is typically brought from global memory **into shared memory** (not directly into TMEM for A/B matrices). The rationale is that the MMA operation on SM100 still *reads* operand matrices from shared memory (or register fragments) and *writes* the result to TMEM. The design is asymmetric: input operands aren’t read from TMEM; TMEM is primarily for outputs (or large accumulators) that would otherwise reside in registers on SM90. So you will still use shared memory double-buffering for input matrices.

- You can also use TMEM for intermediate accumulation. For example, in an MoE with multiple experts’ partial outputs, you could accumulate partial results from several MMAs into one TMEM tile before final output.

- **Memory addressing rules:** Each TMEM pointer is only valid within the CTA or cluster that allocated it. You cannot use it outside (if you do, you’ll get illegal memory access). TMEM pointers are 32-bit offsets – essentially tokens into the SM’s TMEM pool (User Guide for NVPTX Back-end — LLVM 21.0.0git documentation
) (User Guide for NVPTX Back-end — LLVM 21.0.0git documentation
). Do not treat them as normal pointers; do not do pointer arithmetic except perhaps adding offsets provided by descriptors. The API will provide descriptor transformations (like tcgen05.shift) to move a pointer to the next coordinate if needed. The PTX tcgen05.shift can adjust a tensor map’s coordinate in TMEM (for sliding windows in tiled loops) (The Longest Nvidia PTX Instruction | Ash's Blog
), but if using CUTE, it will handle that when you increment your loop.

- **Synchronization models for TMEM:** Since TMEM operations are asynchronous, you have to ensure that:
Data producers have finished writing to TMEM before consumers read from it.
TMEM allocations from different CTAs do not conflict (the hardware and CUDA runtime handle this by granting a CTA exclusive permission to allocate a segment, and the relinquish_alloc_permit instructs when it’s done so others could reuse if necessary).
Within a CTA, if you reuse the same TMEM for multiple sequences (e.g., compute two different GEMMs back-to-back in the same CTA using the same TMEM space), ensure you either use separate allocations or that the first result is completely used and written out before reusing. A tcgen05.dealloc followed by another tcgen05.alloc can recycle the memory.
Use tcgen05.wait or cuda::barrier to avoid read-after-write hazards. For example, don’t launch a second tcgen05.mma
 writing to the same TMEM tile before the first one has at least started (unless intentionally pipelining partial accumulations, which is fine if they are meant to accumulate).
When using CTA pairs: the two CTAs must coordinate so that, say, CTA0 doesn’t deallocate TMEM while CTA1 is still using it, etc. Typically, you’ll treat the pair as a single unit and dealloc at the very end of their work, and use cluster barrier (or the inherent sync in the paired instructions) to ensure they move in lockstep for alloc/dealloc phases.

- **Diagrams & Example:** The figure below illustrates a 2D tensor in memory and how a region (tile) is specified for copying – this concept is used when setting up TMA descriptors for cp.async (the copy engine computes addresses based on tensor strides, avoiding manual loop logic) (NVIDIA Hopper Architecture In-Depth | NVIDIA Technical Blog
) (NVIDIA Hopper Architecture In-Depth | NVIDIA Technical Blog
):

(image
) *Figure 1: Specifying a 2D tile (pink region) to copy from a larger tensor in memory using a TMA descriptor. The TMA unit uses tensor width/height and block dimensions to compute addresses automatically, including any padding (**NVIDIA Hopper Architecture In-Depth | NVIDIA Technical Blog*
*) (**NVIDIA Hopper Architecture In-Depth | NVIDIA Technical Blog*
*).*

In practice, to use TMA, you create a descriptor (with the base address, leading dimension, etc.), then call cuda::memcpy_async (or the PTX tcgen05.cp). The hardware will copy the block (pink) from global memory into the destination (shared memory or TMEM) in a single operation. This contrasts with older GPUs where threads had to calculate addresses and loop for each line (LDGSTS on A100) (NVIDIA Hopper Architecture In-Depth | NVIDIA Technical Blog
) (NVIDIA Hopper Architecture In-Depth | NVIDIA Technical Blog
). SM100’s TMA works similarly to SM90’s, but ensure your descriptors account for any changes (like alignment requirements or new transpose options).

**Putting it all together:** For a FlashMHA-like sequence on SM100 (simplified):

1. **Q*K^T (Attention Scores) using BF16** – allocate TMEM for the scores matrix (D). Stream Q and K tiles from global to shared using TMA. Issue tcgen05.mma
 to compute Q*K^T, result stays in TMEM. Immediately begin streaming next Q, K if looping, while in parallel another warp starts softmax on the portion of scores that is ready:
One warp (or thread) uses tcgen05.ld to load a row of the scores from TMEM, computes exp() and accumulates sums for softmax.
It may need to coordinate via tcgen05.wait to ensure at least one column of the score tile is done. But since the MMA writes the whole tile at once (for that phase of K), we might simply wait for the entire MMA to finish for that tile. Some implementations do softmax in chunks to overlap with the next compute.
Use a shared reduction or atomic to find the max or sum across threads if needed.
Write back the normalized scores into either registers or shared memory.

1. **Scores * V (Output computation) using FP8** – Now treat the normalized scores as matrix A (in shared memory, possibly in FP32 or down-converted to BF16 if needed for tensor core) and V as matrix B (FP8). Allocate another TMEM tile for the output of this GEMM (if different from scores tile). If using FP8 for V with block scaling:
Ensure the scale factors for each block of V are loaded (likely via TMA as well, into shared memory or constant memory). The tcgen05.mma
 ... block_scale will read those scales from memory (the PTX snippet showed passing the addresses of scale factors as operands (Help needed to explain tcgen05.mma_cta_group instructions - CUDA Programming and Performance - NVIDIA Developer Forums
)).
Issue tcgen05.mma
 for Scores*V, with kind .mxf8f6f4.block_scale. This will multiply the (BF16 or FP16) scores by (FP8 values * scale), accumulating in (FP32 or BF16) output in TMEM.
Finally, read out the output TMEM tile and store to global memory (the final hidden states or logits).

1. Throughout these steps, use ClusterBarrier or CTA synchronization where necessary. For instance, ensure that the warp doing softmax has finished reading needed data from TMEM before the next tcgen05.mma
 overwrites it (if reusing the same TMEM for double-buffering the scores, which you might do in a double-buffered pipeline). More likely, you allocate two TMEM buffers for scores (ping-pong), to overlap softmax on one while computing the next into the other. In that case, a barrier between those stages is needed.

The TMEM and TMA capabilities of SM100 allow such overlapping of stages with minimal explicit synchronization. However, it adds complexity in bookkeeping (multiple buffers, more asynchronous events). The next section will discuss how CUTLASS/CUTE and CUDA 12.x help manage some of this complexity.

## CUTLASS/CUTE 3.9 and CUDA 12.x Updates for SM100

NVIDIA’s CUTLASS library (and its subcomponent CUTE) have been updated to support SM100 (Blackwell) and the TCGEN05 instructions. Leveraging these can speed up your porting effort by providing high-level abstractions for the low-level behavior we described. Here we outline relevant changes and how to use them:

```cpp
- **Architecture Tags:** CUTLASS 3.9 introduces an architecture tag for SM100 (e.g., cutlass::arch::Sm100 or similar) and uses it to conditionally compile the new pipeline. When targeting SM100, ensure you compile with the correct -arch=sm_100 flag so that the PTX assembler enables the TCGEN05 instructions. (CUDA 12.x supports sm_100 target with PTX ISA 8.7 – you may need CUDA 12.2 or newer for full support.) The PTX ISA 8.7 includes all TCGEN05 ops (The Longest Nvidia PTX Instruction | Ash's Blog
```
). In CUTLASS, checks like #if
 (__CUDA_ARCH__ >= 1000) (for sm_100) enable code paths that call the new intrinsics.

- **Tensor Core Warp Specialization:** In previous generations, CUTLASS defined “warp-level MMA” shapes like 16x16 or 64x64. With SM100, warp-level operations are replaced or augmented by CTA-level operations. CUTE provides templates in cute/arch/ for SM100 MMA. For example, it has cute::arch::mma_sm100_umma.hpp which likely defines the template for an **“universal MMA”** that maps to tcgen05.mma
 ([BUG]Is tcgen05.fence supported by Cutlass-3.8.0 ? · Issue #2098 · NVIDIA/cutlass · GitHub
) ([BUG]Is tcgen05.fence supported by Cutlass-3.8.0 ? · Issue #2098 · NVIDIA/cutlass · GitHub
). The library abstracts whether the operation is per-warp or per-cta. As a developer, you will specify tile sizes (M,N,K) and the number of warps per CTA. For SM100, you might choose 4 warps in a CTA to handle a 128x128 tile (since each warp could take a 64x64 portion, but the cta_group MMA will join them). CUTLASS now supports **CTA-wide epilogues** too, because the result is in TMEM not in individual warp registers. So it provides new **collective operations** to handle epilogue reduction from TMEM.In
```cpp
 short, expect changes in how you configure ThreadblockShape, WarpShape, etc., for SM100. For instance, a typical CUTLASS GEMM kernel template might look like:using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
using WarpShape = cutlass::gemm::GemmShape<128, 128, 64>; // might match threadblock for CTA-wide
using InstructionShape = cutlass::gemm::GemmShape<128, 128, 64>; // since one MMA covers the whole tile
```
This is different from SM80/90 where WarpShape was smaller (e.g., 64x64) and multiple warps tiled the threadblock.

- **CUTLASS Pipeline Stages:** On H100, CUTLASS pipelines often had 4- staged or 5-staged pipelines (number of cp.async stages) to overlap global memory latency. The MLIR targeting slide even mentions 7 pipeline stages with mbarriers ([PDF] Targeting H100 in MLIR (ODM)
). On SM100, because of larger on-chip memory, one might increase pipeline depth to further overlap loads with compute. However, Blackwell’s massive bandwidth and compute might make the optimal pipeline shallower or similar – it depends on kernel specifics. CUTLASS likely includes a default pipeline configuration for SM100 kernels. Check CUTLASS documentation (e.g., *Blackwell functionality* in their repository, which likely explains usage of tcgen05 instructions (cutlass/media/docs/blackwell_functionality.md at main - GitHub
)). Some changes:
```cpp
**Barrier integration:** CUTLASS’s GroupScheduleBarrier or similar is updated to use tcgen05.commit/wait. If you previously inserted cutlass::arch::fence_view_async_copy(); or cutlass::arch::cp_async_wait_group() calls, those might now map to tcgen05.wait automatically for SM100. Ensure you use the latest CUTLASS API for barrier (like cutlass::PipelineBarrier if provided).
```**TMA in CUTLASS:** CUTLASS 3.x introduced TMA macros for Hopper (enabled via #define
 CUTLASS_ENABLE_TMA). In SM100, these are fully utilized. The library can create a TensorTransferPlan given the problem size, which internally uses either cp.async (for SM90) or tcgen05.cp (SM100). If your code was manually using cp.async, consider switching to CUTLASS’s **Tensor Memory Pipeline** abstraction. For example, cute::make_tma_copy with a Tensor and a SmemTile will generate the code to copy that tile using the proper method.
```cpp**CUTLASS GEMM Kernel changes:** The new tensor cores mean the minimum scheduling unit might be a CTA pair. If using CTA pairs, CUTLASS might expose a mechanism to launch kernels in cluster mode (e.g., a new launch type or an extension of cutlass::Kernel<...> to indicate 2 CTAs per cluster). In one of the search results, a blackwell_cluster_launch_control.md was present, indicating documentation for launching clusters with dynamic scheduling. This likely pertains to how to configure grids such that CTA pairs land on the same SM. It might be handled by cooperative groups or new CUDA cluster APIs.
```

- **Memory Residency and Multi-Buffering:** Because SM100 has more on-chip memory (shared + TMEM), CUTLASS kernels might hold more tiles at once. For instance, on H100 a kernel might double-buffer two stages of K. On SM100, one could triple-buffer: one tile in compute (in TMEM), one in flight via TMA, and one being read out or used in epilogue. This could reduce stalls but at the cost of using more shared memory for buffers. If your kernel was close to using the shared memory limit on H100, you might be able to allocate a bit more on SM100 since each SM has similar shared memory but also now offloads accumulators to TMEM (freeing some shared memory that would have been used to store partial results). So, *memory residency* of intermediate data is improved – you can keep e.g. the entire accumulator matrix resident in TMEM throughout, rather than dumping it to shared or global between phases.

```cpp
- **FP8 and Rounding in CUTLASS:** With hardware FP8 support, you’ll find new data type tags in CUTLASS (e.g., cutlass::half_t for FP16, we might see cutlass::float_e4m3_t or similar for FP8). CUTLASS likely handles the packing/unpacking of FP8 with scales. Pay attention to any new **epilogue** helpers: e.g., to apply the block scale to output or to prepare the scale arrays. SM100’s MMA can apply the scale during multiply for inputs, but for outputs, if you need to quantize back to FP8, you might need to divide by scale and round. NVIDIA could have added an MMA variant that directly produces a scaled int output (some .satfinite or rounding modes were hinted (The Longest Nvidia PTX Instruction | Ash's Blog
```
)). If not, you will do that in an epilogue in registers. CUDA 12.2+ might also introduce device functions to support FP8, like fast conversions.

- **Software example:** If you opt not to use CUTLASS directly, you can still use intrinsics provided by <cuda_fp8.h> or similar. For example, there might be an intrinsic _mma_m64n128k32_fp8_fp8_fp16 etc., but more likely you’ll use inline PTX or CUTLASS. Given the complexity, using CUTLASS’s structures is recommended for large GEMMs, while you might directly write PTX for smaller custom kernels like FlashMHA (as DeepSeek-AI likely did, since they mention inline PTX usage).

In summary, update your build to use CUDA 12.x for SM100, use CUTLASS 3.9 or later for reference code (even if you don’t adopt it, it’s a great reference on how to implement things like CTA pairs, barriers, etc. as they are likely already solved in their examples (ThunderKittens Now Optimized for NVIDIA Blackwell GPUs
)). Keep an eye on CUTLASS’s GitHub for any open issues (for instance, full support for tcgen05.fence and tcgen05.shift might be in progress ([BUG]Is tcgen05.fence supported by Cutlass-3.8.0 ? · Issue #2098 · NVIDIA/cutlass · GitHub
) ([BUG]Is tcgen05.fence supported by Cutlass-3.8.0 ? · Issue #2098 · NVIDIA/cutlass · GitHub
)). The library is still catching up to expose every new instruction (as of early 2025, some intrinsics were present like tcgen05.wait and mma, but not all ([BUG]Is tcgen05.fence supported by Cutlass-3.8.0 ? · Issue #2098 · NVIDIA/cutlass · GitHub
) ([BUG]Is tcgen05.fence supported by Cutlass-3.8.0 ? · Issue #2098 · NVIDIA/cutlass · GitHub
)). If something is missing, you might need to write a small piece of inline PTX.

## Debugging Memory Hazards and Synchronization on SM100

Porting to such low-level code inevitably introduces the risk of tricky bugs: race conditions, incorrect synchronization, and memory access violations. This section provides practical tips to debug and ensure correctness of your ported kernels, focusing on shared memory/TMEM hazards and barrier usage.
**1. Illegal Memory Access (Misuse of TMEM or Shared Memory):**
If you see errors like *“unspecified launch failure”* or memory access exceptions when running the kernel, it could be due to incorrect TMEM usage. Common pitfalls:

- **Using TMEM pointer like a regular pointer:** Remember, TMEM pointers (the value written by tcgen05.alloc) are not pointers into global or shared address space – they are essentially tokens valid only with TC instructions (User Guide for NVPTX Back-end — LLVM 21.0.0git documentation
). If you accidentally use a TMEM pointer in a normal load/store or pass it to a normal CUDA API expecting a global pointer, it will be invalid. Ensure any pointer arithmetic on TMEM addresses is done through tcgen05.shift or by adjusting descriptor coordinates, not by adding byte offsets manually (unless you are very certain of the layout).

- **Not allocating enough TMEM:** If your tcgen05.mma
 writes outside the allocated columns, it will cause a fault. Double-check the sizes: For example, if you allocated 64 columns but ended up doing a 128-wide MMA, you have a problem. It’s safer to allocate a bit more than needed if uncertain (the overhead is just on-chip memory).

- **Mis-aligned TMEM usage:** While tcgen05.alloc ensures alignment, issues can arise if you try to treat parts of TMEM differently. Stick to using it as one contiguous tile per allocation. Do not allocate one large TMEM and manually subdivide for multiple uses without understanding the alignment constraints – it’s better to allocate separate TMEM sections for separate purposes (since you can dealloc and alloc smaller pieces as needed).

- **Shared memory bank conflicts or out-of-bounds:** This is not new to SM100, but if your port changes shared memory layout (for instance, because you adjusted tile shapes), ensure you’re not indexing out of the declared array bounds. Use tools like cuda-memcheck with the race detection mode; it might catch out-of-bounds shared memory accesses. Note that cuda-memcheck might not understand TMEM accesses, but it will catch if you misuse shared memory.

- **Check predicate logic around TMEM ops:** If you use predicates (the .pred p in the asm) to guard an instruction, ensure that in all threads of the warp-group the predicate is set consistently to avoid divergence in a warp-group MMA. Ideally, for CTA-wide instructions, run them converged (one thread does it or all do in unison). A bug could be if one warp in a CTA pair enters the code and the other doesn’t – this could hang or crash the kernel. Use collective conditions (like if(blockIdx.x == 0) or sync-based election) rather than arbitrary thread conditions for these.
**2. Data Races and Hazards (Ordering Issues):**
SM100’s asynchronous nature means it’s easy to introduce use-before-ready errors.

- **Forgetting tcgen05.wait or barrier:** If you launch a TMA load and then immediately use the data in shared memory, you *must* wait for the copy to complete. On H100, failing to do cp.async.wait_group would lead to reading stale or uninitialized data. On SM100, if you neglect a cuda::barrier_wait or tcgen05.wait, the symptoms might be incorrect values or sporadic correctness issues. Always pair your memcpy_async with a barrier.wait() before using the copied data. If using inline PTX, after a tcgen05.cp, do:tcgen05.commit [mbarrier_ptr]; // or barrier.arrive
```cpp
tcgen05.wait [mbarrier_ptr]; // or barrier.wait
```
(Where [mbarrier_ptr] is a memory barrier you set up – or use tcgen05.wait without an arg if it’s implicitly tracking the pipeline.)

```cpp
- **Multiple writers to shared memory or TMEM:** If two warps produce data that will be consumed by a single MMA, ensure they don’t overwrite each other’s results. For example, maybe one warp computes partial of matrix A into shared memory while another warp computes another part; if the MMA reads the whole tile, you must sync them (with __syncthreads() or barrier) before launching the MMA. Similarly, if two different MMAs write to the same TMEM area (accumulating), that’s fine if they are meant to (like partial accumulation), but ensure they are orchestrated by the same CTA-wide instruction sequence. If you inadvertently launch two CTA-wide MMAs in parallel on overlapping TMEM (like in an attempt to pipeline two different GEMMs in the same CTA without separation), you’ll corrupt data. The solution is to use separate TMEM allocations or to serialize them with a barrier between.
```

- **Using cluster (CTA pairs) without proper sync:** CTA pairs operate on the same SM and share TMEM. The good news is that a tcgen05.mma
```cpp
.cta_group::2 inherently synchronizes the pair for that operation (since hardware locks step them). But if you have any code before or after that needs both CTAs to align (like loading A by CTA0 and loading B by CTA1, then the MMA uses both), you must use a **Cluster Barrier**. In CUDA, you can use __clusterBarrierArrive() and __clusterBarrierWait() (if available) or the cooperative launch API ensures a barrier at kernel launch for cluster. NVIDIA provides an API in cooperative groups: e.g., cooperative_groups::this_cluster().sync(). Use it after both CTAs have finished their copy, before they start the MMA. Similarly, after the MMA, if CTA0 is supposed to dealloc TMEM or do something that CTA1 depends on, sync them.
```

- **Memory fence when mixing async and regular memory ops:** If you do a normal st.shared storing something that will be read by the TMA unit (or vice versa), consider using tcgen05.fence or at least __sync_threads() to enforce ordering. One example: you compute a scale factor in registers and store it to shared memory, then you launch a tcgen05.mma
```cpp
...block_scale that will read that scale from shared. You need to ensure the store is visible to the MMA unit. A __syncthreads() would suffice if the same CTA’s threads are doing it, because it’s a CTA fence (all threads will sync and memory is visible). If a single thread wrote it, you might do __threadfence_block() to flush it to shared for other warps. tcgen05.fence is a heavier hammer that ensures all prior operations by that thread are complete relative to the CTA. Generally, prefer high-level sync functions unless you specifically want to only fence memory and not threads.
```
**3. Debugging Tools and Strategies:**

- **Use warp-level printouts or flags:** Debugging inline PTX is tricky since you can’t easily print from device code. One method is to use conditional shared memory writes as debug flags. For example, dedicate an integer in global memory or shared memory for debugging. Have each thread write a certain ID at certain milestones, then after kernel, examine that memory on host. This can tell you how far the code reached before crashing. It’s crude but effective to pinpoint roughly where a fault occurs. Similarly, for data validation, copy small sub-matrices to global and use cudaMemcpy to inspect if they look correct mid-way (in a debug build).

```cpp
- **Check for divergence in CTA-wide calls:** If you suspect not all threads are participating correctly in a CTA-wide operation, force a divergence check. For instance, set a shared variable to 0, then have each thread do if(predicate) atomicAdd(shared_var,1); __syncthreads(); if(threadIdx.x==0) printf("Pred count = %d\n", shared_var);. If the count is not the full warp-group or CTA size when it should be, you know some threads had a different predicate value.
```

```cpp
- **Memory hazard detection:** As of now, there is no official tool that fully understands TMEM hazards. However, you can sometimes catch issues by running on very small problem sizes and varying the timing. Introduce artificial delays (e.g., if(threadIdx.x==0) for(int i=0;i<1000;i++); at various points) to see if a race condition becomes more apparent (maybe producing wrong results or a crash only when delayed). This can indicate missing sync.
```

```cpp
- **Simplify the pipeline for testing:** Try disabling overlapping and run things sequentially to verify correctness, then add overlap back. For example, first test that your kernel produces correct results if you do: load A, load B, __syncthreads(), do MMA, __syncthreads(), softmax (if any), __syncthreads(), next MMA, etc. This will be slower but easier to reason about. Once correct, gradually move loads earlier (before the previous compute finishes) and insert the proper waits. Each step, compare results with the sequential version to ensure no change.
```

- **Leverage CUTLASS’s correctness tests:** If you build a CUTLASS kernel or use their FP8 MMA intrinsics, use their unit tests (if available) as a starting point. For instance, they often have a reference GEMM and will compare the output of the CUTLASS kernel to it. You can adapt those tests for your kernel.

- **Floating-point agreement:** When porting from SM90 to SM100, differences in accumulation order (e.g., CTA-wide vs warp-wide accumulate) might cause slight numeric differences. This is expected (rounding differences), but it shouldn’t be catastrophically different. If you see large differences, suspect a bug in scaling or synchronization. Check that your FP8 scales are applied the same way as before – with hardware doing it now, ensure you didn't double-scale or skip scale. Also ensure that any accumulation in TMEM uses the same precision as before (e.g., if you expected FP32 accumulate and hardware did FP32, you’re fine; but if by mis-specifying you ended up doing BF16 accumulate, that could lower accuracy).
**4. Using ClusterBarrier and mbarrier correctly:**

- **ClusterBarrier:** Only use cluster synchronization when needed, as it can be expensive. For CTA pairs that use tcgen05.mma
```cpp
.cta_group::2, you typically won’t need to manually sync right around the MMA because the instruction itself synchronizes the two CTAs for that operation. But you might need a barrier before the MMA to ensure both CTAs have issued their part of data movement. Use cooperative_groups::this_cluster().sync() if available, or the PTX barrier.cluster arrivel/wait. One approach: use the cluster’s built-in barrier count. If you launched the cluster with 2 CTAs, you can do:cooperative_groups::grid_group grid = cooperative_groups::this_grid();
cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
// both CTAs do:
```
cluster.sync(); // or cluster.arrive_then_wait() 
if(cluster.thread_rank() == 0) { tcgen05.mma
.cta_group::2 ... } 
cluster.sync();
This ensures both CTAs hit the sync before and after the MMA. The middle if(thread_rank==0) ensures only one thread in the cluster (could be CTA0’s thread0) issues the instruction, which is enough to engage both CTAs’ tensor cores.

- **mbarrier for fine-grained pipeline:** If you explicitly use mbarrier, be careful to reset it properly. Example: mbarrier.init(count) to set the number of producers for an async copy group. On Hopper, an mbarrier was often tied to cp.async: we didn’t directly handle it, as cp.async.wait_group implicitly used an internal barrier. On SM100, if you use cuda::barrier, it abstracts mbarrier. If using PTX, use barrier.arrive and barrier.wait with the correct mbarrier handle. One common bug is calling wait too many times (draining the barrier count incorrectly) or not arriving enough. The PTX as referenced in the Hopper blog uses an arrive for each launched copy and a wait for the required number (NVIDIA Hopper Architecture In-Depth | NVIDIA Technical Blog
) (NVIDIA Hopper Architecture In-Depth | NVIDIA Technical Blog
). For multiple stages, you might use multiple mbarriers (like double-buffering: barrier0 for one buffer, barrier1 for the next). Ensure to alternate correctly and not mix them up.
**5. Example: Diagnosing a Softmax Integration Issue:**
Suppose after porting FlashMHA, you find the attention scores are sometimes correct, but occasionally the softmax’d values contain NaNs or infinities on SM100, whereas on SM90 they were fine. This could indicate a synchronization issue where softmax is reading the score tile before it’s fully written (so maybe it read some uninitialized large values). To debug:

- Check if the softmax warp is waiting for the tcgen05.mma
```cpp
 to finish. If you used tcgen05.wait in the thread that issued MMA but the softmax is done by another warp, you actually need a CTA-level sync. Perhaps add an __syncthreads() after MMA (this will force all warps to wait until the issuing warp (thread 0) finishes the MMA instruction). Alternatively, have the issuing thread write a flag to shared memory after tcgen05.wait, and make other warps wait until that flag is set (spin-wait or so). A cluster barrier could also be used if softmax is in a separate CTA (consumer CTA vs producer CTA).
```

- Check if the TMEM tile for scores is double-buffered and if so, ensure softmax warp is operating on the correct buffer (e.g., index flip). If you inadvertently have a bug where both MMA and softmax used the same TMEM buffer concurrently for different tiles, you will definitely get corruption.

- Confirm that the ranges of values are as expected. Maybe SM100’s tensor core did accumulate in BF16 whereas SM90 did in FP32, causing more rounding error and thus a different max index, which in turn changes some exponent to Inf. If that’s the case, ensure you requested the correct accumulator type in the MMA (some instructions allow choosing accumulator type, e.g., .bf16 vs .bf16.tf32 modes). If needed, do an extra __half2float conversion or adjust algorithm (though ideally, keep it in FP32 accumulate).
**6. Always verify with known-good implementations:** Finally, when you think your port is working, test it on real data and compare outputs with the original SM90 kernel or a high-level reference (like PyTorch’s attention or cuBLAS GEMM). Performance tuning is important, but correctness first. After verifying correctness, profile the kernel. If you see underutilization (e.g., achieved occupancy low, or tensor core utilization not 100%), that’s a hint to adjust block sizes or use CTA pairs. Use Nsight Compute’s metrics: It can show if tensor pipelines are stalled waiting on memory (then your TMA pipeline could be lengthened or you need more stages) or if there’s unused TC issue slots (perhaps your blocks are too small).

By following these tips and carefully mapping each aspect of the kernel from SM90 to SM100, you can achieve a successful port that harnesses SM100’s full potential. It’s a non-trivial undertaking – Hopper to Blackwell is a bigger leap in programming model than previous gen upgrades – but the reward is substantial performance gains for transformer inference workloads.

## Conclusion

Porting FlashMHA and DeepGEMM kernels from H100 to B200 (SM90 to SM100) involves a thorough rewrite of low-level operations to use NVIDIA’s new **TCGEN05 instructions, TMEM, and TMA features**. We mapped the old warp-group MMA and cp.async patterns to the new CTA-group MMA and TMA load/store calls, explained how to allocate and utilize the new Tensor Memory for holding large intermediate matrices, and discussed adjustments needed in kernel tiling and launch configuration given SM100’s architectural shifts (fewer but more powerful SMs, larger tensor core tiles, CTA pairing). We also highlighted changes in tools like CUTLASS 3.9 that can simplify using these features, and provided debugging strategies to handle the increased complexity of asynchronous pipelines.

In summary, to achieve **full correctness and peak performance** on SM100:

- Embrace **CTA-wide and paired-CTA** operations to fully utilize 128×128 tensor cores, using TMEM to store and share results.

- Use the **TMA (Tensor Memory Accelerator)** to its fullest: asynchronously load data tiles from global memory and store results, overlapping with compute. Rely on tcgen05.commit/wait or CUDA barriers to manage these transfers.

- Adapt your **kernel tiling and scheduling** to SM100’s resource profile – often larger thread blocks (or clusters) and careful use of on-chip memory to keep those tensor cores busy at all times.

- Leverage **CUTLASS/CUTE** as a reference or foundation, since it provides battle-tested implementations of many of these patterns (including FP8 support and synchronization).

- Pay attention to new **synchronization primitives** (Cluster Barrier, mbarrier, etc.) especially if using multi-CTA cooperation. Always ensure that data in TMEM or shared memory is produced before it’s consumed to avoid subtle races.

- Finally, iteratively test and optimize, because the highest performance will come from a deep understanding of both the algorithm and the hardware. SM100 offers new opportunities (e.g., reducing global memory traffic via TMEM re-use, fusing operations by sharing TMEM data between them) – consider if you can restructure the kernel to take advantage of that (for example, in FlashMHA, keep the attention scores in TMEM and directly use them for the next GEMM with V, rather than writing out to shared and reading again).

Armed with this guide, an experienced CUDA developer should be able to methodically approach the porting process. Start by establishing baseline correctness with straightforward mappings, then incrementally apply optimizations unique to SM100. With careful tuning, you can expect the SM100 kernels to significantly outperform their SM90 counterparts, reflecting the architectural advancements of the Blackwell generation. Good luck, and happy coding on NVIDIA’s SM100!
**Sources:**

- NVIDIA Hopper Architecture and TMA details (NVIDIA Hopper Architecture In-Depth | NVIDIA Technical Blog
) (NVIDIA Hopper Architecture In-Depth | NVIDIA Technical Blog
) (NVIDIA Hopper Architecture In-Depth | NVIDIA Technical Blog
)

- PTX ISA 8.7 (SM100) documentation on TCGEN05 instructions (The Longest Nvidia PTX Instruction | Ash's Blog
) (Contents — PTX ISA 8.7 documentation
)

- NVIDIA Developer Forums – discussions on tcgen05 usage (Help needed to explain tcgen05.mma_cta_group instructions - CUDA Programming and Performance - NVIDIA Developer Forums
) (Help needed to explain tcgen05.mma_cta_group instructions - CUDA Programming and Performance - NVIDIA Developer Forums
)

- Ash Vardanian’s blog on PTX (Blackwell insights) (The Longest Nvidia PTX Instruction | Ash's Blog
) (The Longest Nvidia PTX Instruction | Ash's Blog
)

- Together AI ThunderKittens blog (Blackwell kernel optimizations) (ThunderKittens Now Optimized for NVIDIA Blackwell GPUs
) (ThunderKittens Now Optimized for NVIDIA Blackwell GPUs
)

上午5:16 · 2025年4月20日
·
6.8万
 查看

6

29

346

342

查看 6 条回复