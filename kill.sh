#!/usr/bin/env bash
#ps -ef|grep -E 'RayWorkerWrapper|vllm.entrypoints.openai.api_server'|awk '{print $2}'| xargs kill -9
ps -ef|grep -E 'chatglm4-vllm'|awk '{print $2}'| xargs kill -9
