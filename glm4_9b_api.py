import os

import ray
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams

app = FastAPI()

max_model_len = 12800

# https://huggingface.co/THUDM/glm-4-9b-chat
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

example_prompt = """

### 角色 Role ###
您是一位世界级人工智能专家,程序员,软件架构师,CTO,世界顶级技术畅销书作者，计算机图灵奖获得者，计算机领域大师。

### 任务目标 GOAL ###
现在请您以《【大模型应用开发动手做AI Agent】Plan-and-Solve策略的提出》为标题， 使用逻辑清晰、结构紧凑、简单易懂的专业的技术语言（章节标题要非常吸引读者），写一篇有深度有思考有见解的专业IT领域的技术博客文章。

### 约束条件 CONSTRAINTS ###
- 字数要求：文章字数一定要大于8000字。
- 尽最大努力给出核心概念原理和架构的 Mermaid 流程图(要求：Mermaid 流程节点中不要有括号、逗号等特殊字符)。
- 文章各个段落章节的子目录请具体细化到三级目录。
- 直接开始文章正文部分的撰写。
- 格式要求：文章内容使用markdown格式输出；文章中的数学公式请使用latex格式，latex嵌入文中独立段落使用 $$，段落内使用 $
- 完整性要求：文章内容必须要完整，不能只提供概要性的框架和部分内容，不要只是给出目录。不要只给概要性的框架和部分内容。
- 作者署名：文章末尾需要写上作者署名 “作者：禅与计算机程序设计艺术 / Zen and the Art of Computer Programming”
- 内容要求：文章核心章节内容必须包含如下目录内容(文章结构模板)：
--------------------------------

关键词：

## 1. 背景介绍
### 1.1  问题的由来
### 1.2  研究现状
### 1.3  研究意义
### 1.4  本文结构
## 2. 核心概念与联系
## 3. 核心算法原理 & 具体操作步骤
### 3.1  算法原理概述
### 3.2  算法步骤详解
### 3.3  算法优缺点
### 3.4  算法应用领域
## 4. 数学模型和公式 & 详细讲解 & 举例说明
### 4.1  数学模型构建
### 4.2  公式推导过程
### 4.3  案例分析与讲解
### 4.4  常见问题解答
## 5. 项目实践：代码实例和详细解释说明
### 5.1  开发环境搭建
### 5.2  源代码详细实现
### 5.3  代码解读与分析
### 5.4  运行结果展示
## 6. 实际应用场景
### 6.4  未来应用展望
## 7. 工具和资源推荐
### 7.1  学习资源推荐
### 7.2  开发工具推荐
### 7.3  相关论文推荐
### 7.4  其他资源推荐
## 8. 总结：未来发展趋势与挑战
### 8.1  研究成果总结
### 8.2  未来发展趋势
### 8.3  面临的挑战
### 8.4  研究展望
## 9. 附录：常见问题与解答

--------------------------------

!!!Important:必须要严格遵循上面"约束条件 CONSTRAINTS"中的所有要求撰写这篇文章!!!

### 文章正文内容部分 Content ###
现在，请开始撰写文章正文部分：

# 【大模型应用开发动手做AI Agent】Plan-and-Solve策略的提出

作者：禅与计算机程序设计艺术 / Zen and the Art of Computer Programming 

"""

with open("example_blog.md", 'r') as f:
    example_blog = f.read()

# 定义输入数据的模型
class InputData(BaseModel):
    prompt: str
    max_tokens: int


@app.post("/v1/chat/completions")
def completions(input_data: InputData):
    print(input_data)

    prompt = [
        {"role": "user", "content": example_prompt},
        {"role": "assistant", "content": example_blog},
        {"role": "user", "content": input_data.prompt},
    ]

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



