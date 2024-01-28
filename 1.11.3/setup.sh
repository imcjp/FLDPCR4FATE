#!/bin/bash

# 获取当前目录
current_dir=$(pwd)

# 询问用户部署的子目录
subdir=""

while [ -z "$input" ]; do
    echo "Please enter the subdirectory name to deploy in ${current_dir}:"
    read subdir
done

# 检查子目录是否存在
if [ -d "${current_dir}/${subdir}" ]; then
    echo "The directory ${current_dir}/${subdir} already exists. Exiting."
    exit 1
else
    # 记录子目录名称为fateDir
    fateDir=${subdir}
    echo "Subdirectory ${fateDir} will be created."
fi

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
mkdir -p "${current_dir}/${fateDir}"
tar -zxvf "${current_dir}/standalone_fate_install_${version}_release.tar.gz" -C "${current_dir}/${fateDir}"
mv "${current_dir}/${fateDir}/standalone_fate_install_${version}_release/*" "${current_dir}/${fateDir}" && rmdir "${current_dir}/${fateDir}/standalone_fate_install_${version}_release"

echo "Deployment completed successfully."