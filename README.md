# chatglm4-vllm


## install


```bash
make install
# or
pip3 install -U ray uvicorn fastapi pydantic vllm
```
        
## run

``` 
make run
```

## kill

``` 
make k
```


## vllm usages

``` 
 usage: api_server.py [-h] [--host HOST] [--port PORT] [--uvicorn-log-level {debug,info,warning,error,critical,trace}] [--allow-credentials] [--allowed-origins ALLOWED_ORIGINS] [--allowed-methods ALLOWED_METHODS]
                     [--allowed-headers ALLOWED_HEADERS] [--api-key API_KEY] [--lora-modules LORA_MODULES [LORA_MODULES ...]] [--chat-template CHAT_TEMPLATE] [--response-role RESPONSE_ROLE] [--ssl-keyfile SSL_KEYFILE]
                     [--ssl-certfile SSL_CERTFILE] [--ssl-ca-certs SSL_CA_CERTS] [--ssl-cert-reqs SSL_CERT_REQS] [--root-path ROOT_PATH] [--middleware MIDDLEWARE] [--model MODEL] [--tokenizer TOKENIZER] [--skip-tokenizer-init]
                     [--revision REVISION] [--code-revision CODE_REVISION] [--tokenizer-revision TOKENIZER_REVISION] [--tokenizer-mode {auto,slow}] [--trust-remote-code] [--download-dir DOWNLOAD_DIR]
                     [--load-format {auto,pt,safetensors,npcache,dummy,tensorizer}] [--dtype {auto,half,float16,bfloat16,float,float32}] [--kv-cache-dtype {auto,fp8,fp8_e5m2,fp8_e4m3}]
                     [--quantization-param-path QUANTIZATION_PARAM_PATH] [--max-model-len MAX_MODEL_LEN] [--guided-decoding-backend {outlines,lm-format-enforcer}] [--distributed-executor-backend {ray,mp}] [--worker-use-ray]
                     [--pipeline-parallel-size PIPELINE_PARALLEL_SIZE] [--tensor-parallel-size TENSOR_PARALLEL_SIZE] [--max-parallel-loading-workers MAX_PARALLEL_LOADING_WORKERS] [--ray-workers-use-nsight] [--block-size {8,16,32}]
                     [--enable-prefix-caching] [--disable-sliding-window] [--use-v2-block-manager] [--num-lookahead-slots NUM_LOOKAHEAD_SLOTS] [--seed SEED] [--swap-space SWAP_SPACE] [--gpu-memory-utilization GPU_MEMORY_UTILIZATION]
                     [--num-gpu-blocks-override NUM_GPU_BLOCKS_OVERRIDE] [--max-num-batched-tokens MAX_NUM_BATCHED_TOKENS] [--max-num-seqs MAX_NUM_SEQS] [--max-logprobs MAX_LOGPROBS] [--disable-log-stats]
                     [--quantization {aqlm,awq,deepspeedfp,fp8,marlin,gptq_marlin_24,gptq_marlin,gptq,squeezellm,sparseml,None}] [--rope-scaling ROPE_SCALING] [--enforce-eager]
                     [--max-context-len-to-capture MAX_CONTEXT_LEN_TO_CAPTURE] [--max-seq-len-to-capture MAX_SEQ_LEN_TO_CAPTURE] [--disable-custom-all-reduce] [--tokenizer-pool-size TOKENIZER_POOL_SIZE]
                     [--tokenizer-pool-type TOKENIZER_POOL_TYPE] [--tokenizer-pool-extra-config TOKENIZER_POOL_EXTRA_CONFIG] [--enable-lora] [--max-loras MAX_LORAS] [--max-lora-rank MAX_LORA_RANK]
                     [--lora-extra-vocab-size LORA_EXTRA_VOCAB_SIZE] [--lora-dtype {auto,float16,bfloat16,float32}] [--long-lora-scaling-factors LONG_LORA_SCALING_FACTORS] [--max-cpu-loras MAX_CPU_LORAS] [--fully-sharded-loras]
                     [--device {auto,cuda,neuron,cpu}] [--image-input-type {pixel_values,image_features}] [--image-token-id IMAGE_TOKEN_ID] [--image-input-shape IMAGE_INPUT_SHAPE] [--image-feature-size IMAGE_FEATURE_SIZE]
                     [--scheduler-delay-factor SCHEDULER_DELAY_FACTOR] [--enable-chunked-prefill] [--speculative-model SPECULATIVE_MODEL] [--num-speculative-tokens NUM_SPECULATIVE_TOKENS]
                     [--speculative-max-model-len SPECULATIVE_MAX_MODEL_LEN] [--speculative-disable-by-batch-size SPECULATIVE_DISABLE_BY_BATCH_SIZE] [--ngram-prompt-lookup-max NGRAM_PROMPT_LOOKUP_MAX]
                     [--ngram-prompt-lookup-min NGRAM_PROMPT_LOOKUP_MIN] [--model-loader-extra-config MODEL_LOADER_EXTRA_CONFIG] [--served-model-name SERVED_MODEL_NAME [SERVED_MODEL_NAME ...]] [--engine-use-ray]
                     [--disable-log-requests] [--max-log-len MAX_LOG_LEN]

```


``` 
curl http://localhost:8082/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "internlm2_5-7b-chat",
    "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Introduce deep learning to me."}
    ]
    }'
```

response:

``` 
{"id":"cmpl-25d39164459342c8a75702c5822ff768","object":"chat.completion","created":1720366850,"model":"internlm2_5-7b-chat","choices":[{"index":0,"message":{"role":"assistant","content":"Deep learning is a subset of machine learning that involves training artificial neural networks to recognize patterns in data. These neural networks are composed of multiple layers of interconnected nodes that are designed to mimic the structure and function of the human brain. By feeding large amounts of data through these networks, deep learning algorithms can learn to recognize complex patterns and make predictions or decisions with high accuracy. Deep learning is used in a variety of applications, including image and speech recognition, natural language processing, and autonomous vehicles."},"logprobs":null,"finish_reason":"stop","stop_reason":92542}],"usage":{"prompt_tokens":28,"total_tokens":127,"completion_tokens":99}}
```


## LLMs

``` 
THUDM/LongWriter-glm4-9b

```


