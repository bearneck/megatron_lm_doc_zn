#!/bin/bash

# 配置：运行脚本前设置这些路径
MEGATRON_PATH=${MEGATRON_PATH:-"your_own_megatron_path"} # Megatron-LM 仓库的路径
CONTAINER_IMAGE=${CONTAINER_IMAGE:-"your_own_container_image"} # .sqsh 文件路径或 docker 镜像 URL
OUTPUT_PATH=${OUTPUT_PATH:-"your_own_output_path"} # SLURM 日志的存放路径

# 检查点转换命令
# 注意：更新下方命令中的检查点路径
RUN_CMD="
cd ${MEGATRON_PATH};
git rev-parse HEAD;
export PYTHONPATH=${MEGATRON_PATH}:${PYTHONPATH};
python3 tools/checkpoint/checkpoint_inspector.py \
    convert-torch-dist-to-fsdp-dtensor --swiglu \
    your_own_path_to_input_torch_dist_checkpoint \
    your_own_path_to_output_fsdp_dtensor_checkpoint \
    --param-to-param-group-map-json your_own_path_to_param_to_param_group_map.json"

# SLURM 设置
SLURM_LOGS="${OUTPUT_PATH}/slurm_logs"
mkdir -p ${SLURM_LOGS} || {
    echo "Error: Failed to create SLURM logs directory ${SLURM_LOGS}"
    exit 1
}

# 提交 SLURM 作业
# 注意：根据你的集群配置更新下方的 SBATCH 参数
set +e
sbatch <<EOF
#!/bin/bash

#SBATCH --job-name=your_own_job_name
#SBATCH --partition=your_own_partition
#SBATCH --nodes=your_own_num_nodes
#SBATCH --ntasks-per-node=your_own_tasks_per_node
#SBATCH --gres=gpu:your_own_gpu_per_node
#SBATCH --time=your_own_time
#SBATCH --account=your_own_account
#SBATCH --exclusive
#SBATCH --dependency=singleton

srun --mpi=pmix -l \
    --container-image=${CONTAINER_IMAGE} \
    --container-mounts=your_own_container_mounts \
    --container-workdir=${MEGATRON_PATH} \
    bash -x -c "${RUN_CMD}" 2>&1 | tee ${SLURM_LOGS}/\${SLURM_JOB_ID}.log

EOF
set -e