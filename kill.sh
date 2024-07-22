#!/usr/bin/env bash
ps -ef|grep -E 'RayWorkerWrapper|vllm.entrypoints.openai.api_server'|awk '{print $2}'| xargs kill -9