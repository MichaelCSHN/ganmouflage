#!/bin/bash

# #####################################################################################
# env.sh: 配置 GANmouflage 项目的 Conda 环境
# 目标架构: Python 3.8, PyTorch 1.13.0, CUDA 11.7, PyTorch3D 0.7.3 (源码编译)
# 基于成功配置 "camoGAN_py38pt113cu117" 环境的经验
# #####################################################################################

# --- 配置参数 ---
export ENV_NAME="camoGAN_py38pt113cu117_adapted" # 建议新环境名，您也可以沿用旧名
export PYTHON_VERSION="3.8"
export PYTORCH_VERSION="1.13.0"
export TORCHVISION_VERSION="0.14.0"
export TORCHAUDIO_VERSION="0.13.0"
export CUDA_TOOLKIT_VERSION="11.7" # 对应 pytorch-cuda=11.7
export PYTORCH3D_VERSION="0.7.3"

echo "================================================================================="
echo "环境配置脚本 for ${ENV_NAME}"
echo "Python: ${PYTHON_VERSION}, PyTorch: ${PYTORCH_VERSION}, CUDA: ${CUDA_TOOLKIT_VERSION}, PyTorch3D: ${PYTORCH3D_VERSION}"
echo "================================================================================="

# --- 1. 创建或指定 Conda 环境 ---
conda env inspect $ENV_NAME > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "正在创建 Conda 环境: $ENV_NAME (Python ${PYTHON_VERSION})..."
    conda create -n $ENV_NAME python=$PYTHON_VERSION -y
else
    echo "Conda 环境 $ENV_NAME 已存在。"
fi

echo ""
echo "重要提示: 请手动激活 Conda 环境后，再执行后续的安装命令："
echo "conda activate ${ENV_NAME}"
echo "---------------------------------------------------------------------------------"
echo "以下命令需要在激活的 ${ENV_NAME} 环境中执行："
echo ""

# 将后续命令输出到一个临时文件，方便用户复制粘贴
INSTALL_COMMANDS_FILE="install_commands_for_${ENV_NAME}.sh"

cat > ${INSTALL_COMMANDS_FILE} << EOF
#!/bin/bash
# 请在激活环境 "conda activate ${ENV_NAME}" 后逐条或批量执行以下命令

echo "--- 正在安装核心 PyTorch 架构 ---"
conda install -y pytorch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} pytorch-cuda=${CUDA_TOOLKIT_VERSION} -c pytorch -c nvidia

echo "--- 正在安装编译 PyTorch3D 所需的工具 (如果全新环境) ---"
# 注意: 在您成功的 camoGAN_py38pt113cu117 环境中, gcc/gxx 工具链最终来自 'pkgs/main' 频道。
# 此处优先尝试 conda-forge，如果遇到问题，可能需要审视这些包的实际来源。
conda install -y -c conda-forge libxcrypt sysroot_linux-64 gcc_linux-64 gxx_linux-64 binutils_linux-64 ninja

# --- 关于 PyTorch3D v${PYTORCH3D_VERSION} ---
echo "*********************************************************************************"
echo "重要: PyTorch3D v${PYTORCH3D_VERSION} 在您之前的 camoGAN_py38pt113cu117 环境中是基于特定配置从源码编译成功的。"
echo "如果您正在一个全新的 ${ENV_NAME} 环境中从头开始，您需要重复之前的源码编译步骤："
echo "  1. git clone --branch v${PYTORCH3D_VERSION} https://github.com/facebookresearch/pytorch3d.git; cd pytorch3d"
echo "  2. git clean -fdx"
echo "  3. unset CPLUS_INCLUDE_PATH CPATH"
echo "  4. export CUDA_HOME=\\\$CONDA_PREFIX CUDA_PATH=\\\$CONDA_PREFIX CUDA_TOOLKIT_ROOT_DIR=\\\$CONDA_PREFIX"
echo "  5. export MAX_JOBS=1"
echo "  6. pip install . -vvv"
echo "确保在此环境中 \$CONDA_PREFIX/include/crypt.h 文件存在 (通常由 libxcrypt 提供)。"
echo "如果您是在已成功编译 PyTorch3D 的 camoGAN_py38pt113cu117 环境基础上操作，则此步骤已完成。"
echo "*********************************************************************************"

echo "--- 正在安装 GANmouflage 项目的其他依赖 ---"

echo "安装 fvcore 和 iopath (PyTorch3D 依赖)..."
conda install -y -c fvcore -c iopath -c conda-forge fvcore iopath

echo "安装 pyembree..."
conda install -y -c conda-forge pyembree

echo "安装其他 pip 包..."
pip install open3d pandas opencv-python trimesh matplotlib scikit-learn scikit-image tensorboardX tqdm lpips

echo "安装 Kornia (适配 PyTorch ${PYTORCH_VERSION})..."
# Kornia v0.5.0 与 PyTorch 1.13.0 不兼容。Kornia v0.7.x 通常兼容 PyTorch 1.13+。
pip install "kornia>=0.7.0"

echo "安装 Gradual Warmup LR Scheduler..."
pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git

echo "--- 依赖安装指令结束 ---"
echo "请检查是否有错误，并确保 PyTorch3D v${PYTORCH3D_VERSION} 已正确安装或已准备好按上述说明进行编译。"
EOF

chmod +x ${INSTALL_COMMANDS_FILE}

echo "安装命令已保存到 ./${INSTALL_COMMANDS_FILE}"
echo "请激活环境后，运行 ./${INSTALL_COMMANDS_FILE} 或手动执行其中的命令。"
echo "================================================================================="