import os

import ray
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams

app = FastAPI()

max_model_len = 12800
model_name = "THUDM/glm-4-9b-chat"

stop_token_ids = [151329, 151336, 151338]

# 设置 Ray 环境变量
os.environ['RAY_memory_monitor_refresh_ms'] = '0'
os.environ['RAY_memory_usage_threshold'] = '0.8'
# 指定要使用的CUDA设备
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # 例如，这里设置为使用两个GPU，编号为0和1
# 设置`PYTORCH_CUDA_ALLOC_CONF`环境变量，以避免内存碎片化。
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# Set the memory usage threshold (in bytes) for Ray, For example, 1GB
memory_usage_threshold = 2 * 10 ** 9
# 设置对象存储的内存大小为1GB
object_store_memory = 1 * 10 ** 9
# 初始化Ray ， Start Ray and set the memory usage threshold
ray.init(
    # 设置Ray进程的内存大小
    _memory=memory_usage_threshold,
    # 指定Ray可以使用的GPU数量
    num_gpus=2,
    object_store_memory=object_store_memory,
    _temp_dir="/home/me/ray/temp"
)

# 如果遇见 OOM 现象，建议减少max_model_len，或者增加tp_size
# Create an LLM
llm = LLM(
    model=model_name,
    trust_remote_code=True,
    tensor_parallel_size=2,
    gpu_memory_utilization=0.9,
    swap_space=4,
    enforce_eager=True,
    # GLM-4-9B-Chat-1M 如果遇见 OOM 现象，建议开启下述参数
    enable_chunked_prefill=True,
    max_num_batched_tokens=8192
)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


# 定义输入数据的模型
class InputData(BaseModel):
    prompt: str
    max_tokens: int


@app.post("/v1/chat/completions")
def completions(input_data: InputData):
    print(input_data)

    prompt = [{"role": "user", "content": input_data.prompt}]
    inputs = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

    max_tokens = max_model_len
    if input_data.max_tokens != 0:
        max_tokens = input_data.max_tokens

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=max_tokens,
                                     stop_token_ids=stop_token_ids)
    outputs = llm.generate(prompts=inputs, sampling_params=sampling_params)

    # print(outputs)
    text = outputs[0].outputs[0].text

    print(text)

    return text


if __name__ == '__main__':
    # 启动API服务
    uvicorn.run(app,
                host="0.0.0.0",
                port=8000,
                log_level="info",
                access_log=False,
                timeout_keep_alive=5)
