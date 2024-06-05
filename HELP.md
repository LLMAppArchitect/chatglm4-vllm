# Engine Arguments

https://docs.vllm.ai/en/latest/models/engine_args.html


```shell

usage: -m vllm.entrypoints.openai.api_server [-h] [--model MODEL]
                                             [--tokenizer TOKENIZER]
                                             [--skip-tokenizer-init]
                                             [--revision REVISION]
                                             [--code-revision CODE_REVISION]
                                             [--tokenizer-revision TOKENIZER_REVISION]
                                             [--tokenizer-mode {auto,slow}]
                                             [--trust-remote-code]
                                             [--download-dir DOWNLOAD_DIR]
                                             [--load-format {auto,pt,safetensors,npcache,dummy,tensorizer}]
                                             [--dtype {auto,half,float16,bfloat16,float,float32}]
                                             [--kv-cache-dtype {auto,fp8,fp8_e5m2,fp8_e4m3}]
                                             [--quantization-param-path QUANTIZATION_PARAM_PATH]
                                             [--max-model-len MAX_MODEL_LEN]
                                             [--guided-decoding-backend {outlines,lm-format-enforcer}]
                                             [--distributed-executor-backend {ray,mp}]
                                             [--worker-use-ray]
                                             [--pipeline-parallel-size PIPELINE_PARALLEL_SIZE]
                                             [--tensor-parallel-size TENSOR_PARALLEL_SIZE]
                                             [--max-parallel-loading-workers MAX_PARALLEL_LOADING_WORKERS]
                                             [--ray-workers-use-nsight]
                                             [--block-size {8,16,32}]
                                             [--enable-prefix-caching]
                                             [--use-v2-block-manager]
                                             [--num-lookahead-slots NUM_LOOKAHEAD_SLOTS]
                                             [--seed SEED]
                                             [--swap-space SWAP_SPACE]
                                             [--gpu-memory-utilization GPU_MEMORY_UTILIZATION]
                                             [--num-gpu-blocks-override NUM_GPU_BLOCKS_OVERRIDE]
                                             [--max-num-batched-tokens MAX_NUM_BATCHED_TOKENS]
                                             [--max-num-seqs MAX_NUM_SEQS]
                                             [--max-logprobs MAX_LOGPROBS]
                                             [--disable-log-stats]
                                             [--quantization {aqlm,awq,deepspeedfp,fp8,marlin,gptq_marlin_24,gptq_marlin,gptq,squeezellm,sparseml,None}]
                                             [--rope-scaling ROPE_SCALING]
                                             [--enforce-eager]
                                             [--max-context-len-to-capture MAX_CONTEXT_LEN_TO_CAPTURE]
                                             [--max-seq-len-to-capture MAX_SEQ_LEN_TO_CAPTURE]
                                             [--disable-custom-all-reduce]
                                             [--tokenizer-pool-size TOKENIZER_POOL_SIZE]
                                             [--tokenizer-pool-type TOKENIZER_POOL_TYPE]
                                             [--tokenizer-pool-extra-config TOKENIZER_POOL_EXTRA_CONFIG]
                                             [--enable-lora]
                                             [--max-loras MAX_LORAS]
                                             [--max-lora-rank MAX_LORA_RANK]
                                             [--lora-extra-vocab-size LORA_EXTRA_VOCAB_SIZE]
                                             [--lora-dtype {auto,float16,bfloat16,float32}]
                                             [--long-lora-scaling-factors LONG_LORA_SCALING_FACTORS]
                                             [--max-cpu-loras MAX_CPU_LORAS]
                                             [--fully-sharded-loras]
                                             [--device {auto,cuda,neuron,cpu}]
                                             [--image-input-type {pixel_values,image_features}]
                                             [--image-token-id IMAGE_TOKEN_ID]
                                             [--image-input-shape IMAGE_INPUT_SHAPE]
                                             [--image-feature-size IMAGE_FEATURE_SIZE]
                                             [--scheduler-delay-factor SCHEDULER_DELAY_FACTOR]
                                             [--enable-chunked-prefill]
                                             [--speculative-model SPECULATIVE_MODEL]
                                             [--num-speculative-tokens NUM_SPECULATIVE_TOKENS]
                                             [--speculative-max-model-len SPECULATIVE_MAX_MODEL_LEN]
                                             [--speculative-disable-by-batch-size SPECULATIVE_DISABLE_BY_BATCH_SIZE]
                                             [--ngram-prompt-lookup-max NGRAM_PROMPT_LOOKUP_MAX]
                                             [--ngram-prompt-lookup-min NGRAM_PROMPT_LOOKUP_MIN]
                                             [--model-loader-extra-config MODEL_LOADER_EXTRA_CONFIG]
                                             [--served-model-name SERVED_MODEL_NAME [SERVED_MODEL_NAME ...]]
```

# Support Models

``` 
(RayWorkerWrapper pid=69963) ERROR 05-25 18:48:06 worker_base.py:145]   File "/home/me/.conda/envs/phi3-vllm/lib/python3.11/site-packages/vllm/model_executor/model_loader/utils.py", line 35, in get_model_architecture
(RayWorkerWrapper pid=69963) ERROR 05-25 18:48:06 worker_base.py:145]     raise ValueError(
(RayWorkerWrapper pid=69963) ERROR 05-25 18:48:06 worker_base.py:145] ValueError: Model architectures ['Phi3SmallForCausalLM'] are not supported for now. 
Supported architectures: ['AquilaModel', 'AquilaForCausalLM', 'BaiChuanForCausalLM', 'BaichuanForCausalLM', 'BloomForCausalLM', 
'ChatGLMModel', 'ChatGLMForConditionalGeneration', 'CohereForCausalLM', 'DbrxForCausalLM', 'DeciLMForCausalLM', 
'DeepseekForCausalLM', 'FalconForCausalLM', 'GemmaForCausalLM', 'GPT2LMHeadModel', 'GPTBigCodeForCausalLM', 
'GPTJForCausalLM', 'GPTNeoXForCausalLM', 'InternLMForCausalLM', 'InternLM2ForCausalLM', 'JAISLMHeadModel', 
'LlamaForCausalLM', 'LlavaForConditionalGeneration', 'LLaMAForCausalLM', 'MistralForCausalLM', 'MixtralForCausalLM',
 'QuantMixtralForCausalLM', 'MptForCausalLM', 'MPTForCausalLM', 'MiniCPMForCausalLM', 'OlmoForCausalLM', 'OPTForCausalLM', 'OrionForCausalLM', 
 'PhiForCausalLM', 'Phi3ForCausalLM', 'QWenLMHeadModel', 'Qwen2ForCausalLM', 
'Qwen2MoeForCausalLM', 'RWForCausalLM', 'StableLMEpochForCausalLM', 'StableLmForCausalLM', 'Starcoder2ForCausalLM', 'XverseForCausalLM']

```