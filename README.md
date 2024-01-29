# FLDPCR4FATE
 这是一个将FLDPCR集成于FATE框架的项目

## 1. 说明

**FLDPCR4FATE**项目是一个将FLDPCR技术集成到FATE框架中的创新性工作，弥补了现有FATE框架不支持差分隐私的问题。通过结合FLDPCR，该项目不仅增强了数据隐私保护，还显著提高了差分隐私下联邦学习的学习精度。
相关技术项目可参加GitHub链接：[FLDPCR](https://github.com/imcjp/FLDPCR) 和 [Opacus-DPCR](https://github.com/imcjp/Opacus-DPCR)。这一技术的集成，为追求数据隐私保护与模型性能之间平衡的研究者和开发者提供了一个重要的工具。


**服务器配置：**

- **数量:** 1
- **配置:** 8 核 / 16GB 内存 / 500GB 硬盘 / GPU(可选）
- **操作系统:** CentOS (Ubuntu)
- **用户:** User: app owner:apps

单机版提供 3 种部署方式，可以根据实际情况选择：

- 从 PyPI 安装 FATE
- 使用 Docker 镜像安装 FATE
- 在主机中安装 FATE (使用已编译的安装包)

## 2. FLDPCR4FATE的部署安装


**注意:** FLDPCR4FATE的安装依赖于具体的FATE版本，可选的版本见本页面中以FATE版本号命名的文件夹。例如，文件夹1.11.3表示在版本号为1.11.3的FATE基础上部署FLDPCR。用户在安装是可以通过 `${version}`自行指定。

### 2.1 部署前环境检查

检查本地端口 8080、9360 和 9380 是否被占用。

```bash
netstat -apln|grep 8080;
netstat -apln|grep 9360;
netstat -apln|grep 9380
```

由于需要安装操作系统依赖包，所以需要 root 权限。您可以使用 root 用户执行后续操作，如果不使用 root 用户，请为要使用的用户分配 sudo 权限：

```bash
echo "{要使用的用户名}  ALL=(ALL) NOPASSWD:ALL" | tee /etc/sudoers.d/{要使用的用户名}
```

### 2.2 FLDPCR4FATE的部署

设置关联的FATE版本，可选的版本见本页面中以FATE版本号命名的文件夹。

```bash
export version=1.11.3
```

设置期望部署的文件夹，若用户指定 `${projName}`，后续将在当前目录的 `${projName}`文件夹下部署FLDPCR4FATE。请注意 `${projName}`文件夹应当在部署前不存在于当前目录。若${projName}不指定或未空，则默认部署到当前目录的fateProj文件夹。

```bash
export projName=fateProj
```

**直接执行如下命令即可完成本项目的部署。**

```bash
curl -s https://raw.githubusercontent.com/imcjp/FLDPCR4FATE/main/${version}/setup.sh | bash -s -- ${projName}
```

### 2.3 安装

进入部署后的目录并使用 `bin/init.sh` 进行安装。

除了自动完成FATE框架所具有的任务，还将进一步完成以下任务：

- 安装 Python 包 opacus-dpcr
- 安装 GPU版的Pytorch（可选）

```bash
cd ${projName};
bash bin/init.sh init
```

### 2.4 启动

```bash
bash bin/init.sh status （可选）
bash bin/init.sh start
bash bin/init.sh status （可选，若顺利，则正在运行）
```

### 2.5 关闭

若需关闭，请使用如下命令

```bash
bash bin/init.sh stop
bash bin/init.sh status （可选，若顺利，则显示关闭）
```


## 3. 测试
### 3.1 加载环境变量

```bash
source bin/init_env.sh
```

### 3.2 上传数据

将本项目提供的MNIST数据集上传至FATE Flow。该MNIST数据集以CSV格式存储，被划分为20份平均。详情参见 'examples/data/mnist_csv/'。

```bash
python examples/pipeline/homo_nn/fldpcr/pipeline-upload-mnist.py
```

### 3.3 启动隐私训练

**执行本项目所提供的FLDPCR学习示例。**

```bash
python examples/pipeline/homo_nn/fldpcr/demo.py --nIter 10 --gpuId 0
```

**demo.py脚步的参数说明如下：**

```bash
usage: demo.py [-h] [--nClient NCLIENT]
               [--dpcrModel {SimpleMech,TwoLevel,BinMech,FDA,BCRG,ABCRG}]
               [--nIter NITER] [--batch_size BATCH_SIZE]
               [--dp_sigma DP_SIGMA] [--dp_delta DP_DELTA]
               [--dp_norm DP_NORM] [--gpuId GPUID] [--k K] [--seed SEED]
```

**Optional arguments:**
- `-h`, `--help`  
  Show this help message and exit.

- `--nClient NCLIENT`  
  Number of clients for federated learning.

- `--dpcrModel {SimpleMech,TwoLevel,BinMech,FDA,BCRG,ABCRG}`  
  Specify the DPCR model to use. Choices are: `SimpleMech` (equivalent to traditional DP, ignore parameter $k$), `TwoLevel`, `BinMech`, `FDA`, `BCRG`, `ABCRG`. For `TwoLevel`, T=k^2; for `BinMech`, $T=2^k$; for `FDA`, `BCRG`, and `ABCRG`, $T=2^k-1$. If the actual number of iterations exceeds T, a new DPCR model will be constructed for training.

- `--nIter NITER`  
  Number of iterations for training.

- `--batch_size BATCH_SIZE`  
  Batch size for training.

- `--dp_sigma DP_SIGMA`  
  Differential privacy noise scale.

- `--dp_delta DP_DELTA`  
  Differential privacy delta parameter.

- `--dp_norm DP_NORM`  
  Maximum clipping norm for differential privacy.

- `--gpuId GPUID`  
  GPU ID to use for training (if available), otherwise `-1` for CPU.

- `--k K`  
  Model size parameter k.

- `--seed SEED`  
  Random seed for reproducibility.
