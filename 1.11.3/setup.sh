#!/bin/bash

# 获取当前目录
current_dir=$(pwd)

# 询问用户部署的子目录
subdir=${1:-fate}

echo "You set the deployment directory to '${current_dir}/${subdir}':"

# 检查子目录是否存在
if [ -d "${current_dir}/${subdir}" ]; then
    echo "The directory ${current_dir}/${subdir} already exists."
    echo "Has the FATE framework been deployed in this directory? (Y/N)"
    read -r deploy_confirmation
    if [ "$deploy_confirmation" != "Y" ]; then
        echo "Exiting."
        exit 1
    fi
fi

fateDir=${subdir}

# 如果用户确认已部署FATE，跳过下载和解压FATE框架的步骤
if [ "$deploy_confirmation" != "Y" ]; then
    # 设置版本号
    version="1.11.3"

    # 检查文件是否存在
    if [ ! -f "${current_dir}/standalone_fate_install_${version}_release.tar.gz" ]; then
        # 文件不存在，使用wget下载
        echo "Downloading standalone_fate_install_${version}_release.tar.gz..."
        wget "https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/fate/${version}/release/standalone_fate_install_${version}_release.tar.gz" -P "${current_dir}"
    fi

    # 检查下载是否成功
    if [ ! -f "${current_dir}/standalone_fate_install_${version}_release.tar.gz" ]; then
        echo "Download failed. Please check the URL or network connection."
        exit 1
    fi

    # 解压内容到子目录fateDir
    echo "Extracting standalone_fate_install_${version}_release.tar.gz to ${current_dir}/${fateDir}..."
    tmp_dir="tmp_$(date +%Y%m%d%H%M%S)_$RANDOM"
    mkdir -p "${current_dir}/${tmp_dir}"
    tar -zxf "${current_dir}/standalone_fate_install_${version}_release.tar.gz" -C "${current_dir}/${tmp_dir}"
    mv "${current_dir}/${tmp_dir}/standalone_fate_install_${version}_release" "${current_dir}/${fateDir}" && rm -rf "${current_dir}/${tmp_dir}"
fi

# 下载v0.tar.gz并给它一个随机名字tmpName
tmpName=$(cat /dev/urandom | tr -cd 'a-f0-9' | head -c 16).tar.gz
echo "Downloading v0.tar.gz as ${tmpName}..."
wget -O "${current_dir}/${tmpName}" "https://raw.githubusercontent.com/imcjp/FLDPCR4FATE/main/1.11.3/v0.tar.gz"

# 检查下载是否成功
if [ ! -f "${current_dir}/${tmpName}" ]; then
    echo "Download of v0.tar.gz failed. Please check the URL or network connection."
    exit 1
fi

# 解压tmpName到${current_dir}/${fateDir}
echo "Extracting ${tmpName} to ${current_dir}/${fateDir}..."
tar -zxf "${current_dir}/${tmpName}" -C "${current_dir}/${fateDir}"
rm -rf "${current_dir}/${tmpName}"


echo "Deployment of FLDPCR to FATE V 1.11.3 completed successfully."
