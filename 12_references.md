# References

*From [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)*

## References

### Landmark LLM scaling papers

Introduces tensor parallelism and efficient model parallelism techniques for training large language models.

Describes the training of a 530B parameter model using a combination of DeepSpeed and Megatron-LM frameworks.

Introduces Google's Pathways Language Model, demonstrating strong performance across hundreds of language tasks and reasoning capabilities.

Presents Google's multimodal model architecture capable of processing text, images, audio, and video inputs.

Introduces the Llama 3 herd of models.

DeepSeek's report on the architecture and training of the DeepSeek-V3 model.

### Training frameworks

Our framework for training large language models, featuring various parallelism strategies.

NVIDIA's framework for training large language models, featuring various parallelism strategies.

Microsoft's deep learning optimization library, featuring ZeRO optimization stages and various parallelism strategies.

A PyTorch extension library for large-scale training, offering various parallelism and optimization techniques.

An integrated large-scale model training system with various optimization techniques.

A PyTorch native library for large model training.

EleutherAI's framework for training large language models, used to train GPT-NeoX-20B.

Lightning AI's implementation of 20+ state-of-the-art open source LLMs, with a focus on reproducibility.

An open source framework for training language models across compute clusters with DiLoCo.

A GPipe implementation in PyTorch.

The Open Source for Large-scale Optimization framework for large-scale modeling.

### Debugging

Official PyTorch tutorial on using the profiler to analyze model performance and bottlenecks.

Comprehensive guide to understanding and optimizing GPU memory usage in PyTorch.

Guide to visualizing and understanding GPU memory in PyTorch.

Guide to using TensorBoard's profiling tools for PyTorch models.

### Distribution techniques

Comprehensive explanation of data parallel training in deep learning.

Introduces the Zero Redundancy Optimizer for training large models with memory optimization.

Fully Sharded Data Parallel training implementation in PyTorch.

Advanced techniques for efficient large-scale model training combining different parallelism strategies.

NVIDIA's guide to implementing pipeline parallelism for large model training.

Includes broad discussions of PP schedules.

Detailed explanation of the ring all-reduce algorithm used in distributed training.

Implementation of the Ring Attention mechanism combined with FlashAttention for efficient training.

Tutorial explaining the concepts and implementation of Ring Attention.

DeepSpeed's guide to understanding the trade-offs between ZeRO and 3D parallelism strategies.

Introduces mixed precision training techniques for deep learning models.

Explains the collective communication involved in a 6D parallel mesh.

### Hardware

DeepSeek's report on designing a cluster with 10k PCI GPUs.

Meta's detailed overview of their massive AI infrastructure built with NVIDIA H100 GPUs.

Analysis of large-scale H100 GPU clusters and their implications for AI infrastructure.

CUDA docs for humans.

### Others

Comprehensive handbook covering various aspects of training LLMs.

Detailed documentation of the BLOOM model training process and challenges.

Meta's detailed logbook documenting the training process of the OPT-175B model.

Investigation of the relationship between model size and training overhead.

Investigation of long context training in terms of data and training cost.

A GPU reading group and community.

ML scalability & performance reading group.

How to scale your model.

Standalone ~500 LoC FSDP implementation

Some of Horace He's blog posts.

Easy explanation of FlashAttention.

Large-scale language modeling tutorials with PyTorch.
