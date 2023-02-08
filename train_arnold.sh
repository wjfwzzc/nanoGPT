#!/bin/bash

if [[ "$ARNOLD_DEVICE_TYPE" == *A100* ]]; then
    IB_HCA=mlx5
else
    IB_HCA=$ARNOLD_RDMA_DEVICE:1
fi

port=$(echo "$ARNOLD_WORKER_0_PORT" | cut -d "," -f 1)

export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=$IB_HCA
export NCCL_SOCKET_IFNAME=eth0

torchrun \
    --rdzv_id=lab.vc.nanoGPT.$ARNOLD_WORKER_0_HOST \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$ARNOLD_WORKER_0_HOST:$port \
    --nnodes=$ARNOLD_WORKER_NUM \
    --nproc_per_node=$ARNOLD_WORKER_GPU \
    train.py $@
