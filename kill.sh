#!/usr/bin/env bash
ps -ef|grep 'glm4_9b_api.py'|awk '{print $2}'| xargs kill -9