#!/usr/bin/env bash
ps -ef|grep 'ray::RayWorkerWrapper.execute_method'|awk '{print $2}'| xargs kill -9