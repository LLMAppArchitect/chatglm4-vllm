#!/usr/bin/env bash

# 循环执行
for i in {1..5} ; do
     # 打印序号和时间戳
    echo "    WriteBlogOutlineRandomlyWithLLM Request $i at $(date +%Y-%m-%d_%H:%M:%S)    "
    # 执行curl命令
    curl -X GET "http://127.0.0.1:9000/api/ai/WriteBlogOutlineRandomlyWithLLM?model=LongWriter" -H  "Request-Origion:SwaggerBootstrapUi" -H  "accept:*/*"
    # 暂停n秒钟
    sleep 5
done
