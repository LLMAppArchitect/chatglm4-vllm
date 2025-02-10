from modelscope import snapshot_download
snapshot_download('tclf90/deepseek-r1-distill-qwen-14b-gptq-int4', cache_dir="model", revision='g128')