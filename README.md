# FLDPCR4FATE
 這是一個將FLDPCR集成於FATE框架的項目

## 1. 說明

**FLDPCR4FATE**項目是一個將FLDPCR技術集成到FATE框架中的創新性工作，彌補了現有FATE框架不支持差分隱私的問題。通過結合FLDPCR，該項目不僅增強了數據隱私保護，還顯著提高了差分隱私下聯邦學習的學習精度。
相關技術項目可參加GitHub鏈接：[FLDPCR](https://github.com/imcjp/FLDPCR) 和 [Opacus-DPCR](https://github.com/imcjp/Opacus-DPCR)。這一技術的集成，爲追求數據隱私保護與模型性能之間平衡的研究者和開發者提供了一個重要的工具。


**服務器配置：**

- **數量:** 1
- **配置:** 8 核 / 16GB 內存 / 500GB 硬盤 / GPU(可選）
- **操作系統:** Ubuntu
- **用戶:** User: app owner:apps

## 2. FLDPCR4FATE的部署安裝


**注意:** FLDPCR4FATE的安裝依賴於具體的FATE版本，可選的版本參加見本頁面中以FATE版本號命名的文件夾。例如，文件夾1.11.3表示在版本號爲1.11.3的FATE基礎上部署FLDPCR。用戶在安裝是可以通過 `${version}`自行指定。

### 2.1 部署前環境檢查

檢查本地端口 8080、9360 和 9380 是否被佔用。

```bash
netstat -apln|grep 8080;
netstat -apln|grep 9360;
netstat -apln|grep 9380
```

由於需要安裝操作系統依賴包，所以需要 root 權限。您可以使用 root 用戶執行後續操作，如果不使用 root 用戶，請爲要使用的用戶分配 sudo 權限：

```bash
echo "{要使用的用戶名}  ALL=(ALL) NOPASSWD:ALL" | tee /etc/sudoers.d/{要使用的用戶名}
```

### 2.2 FLDPCR4FATE的部署

設置關聯的FATE版本，可選的版本**參見本頁面中以FATE版本號命名的文件夾**。

```bash
export version=1.11.3
```

設置期望部署的文件夾，若用戶指定 `${projName}`，後續將在當前目錄的 `${projName}`文件夾下部署FLDPCR4FATE。請注意 `${projName}`文件夾應當在部署前不存在於當前目錄。若${projName}不指定或未空，則默認部署到當前目錄的 `fate` 文件夾。

```bash
export projName=fate
```

**直接執行如下命令即可完成本項目的部署。**

```bash
curl -s https://raw.githubusercontent.com/imcjp/FLDPCR4FATE/main/${version}/setup.sh | bash -s -- ${projName}
```

### 2.3 安裝

進入部署後的目錄並使用 `bin/init.sh` 進行安裝。

除了自動完成FATE框架所具有的任務，還將進一步完成以下任務：

- 安裝 Python 包 opacus-dpcr
- 安裝 GPU版的Pytorch（可選）

```bash
cd ${projName};
bash bin/init.sh init
```

- 若在執行 `bash bin/init.sh init` 時選擇不安裝GPU版運行庫，後續也可以通過 `source bin/init_env.sh` 進入FATE環境安裝，安裝命令的參考如下：

```bash
pip uninstall -y torch torchvision # 先卸載已有運行庫
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # 安裝GPU版運行庫
```

### 2.4 啓動

```bash
bash bin/init.sh status （可選）
bash bin/init.sh start
bash bin/init.sh status （可選，若順利，則顯示正在運行）
```

### 2.5 關閉

若需關閉，請使用如下命令

```bash
bash bin/init.sh stop
bash bin/init.sh status （可選，若順利，則顯示關閉）
```


## 3. 測試
### 3.1 加載環境變量

```bash
source bin/init_env.sh
```

### 3.2 上傳數據

將本項目提供的MNIST數據集上傳至FATE Flow。該MNIST數據集以CSV格式存儲，被劃分爲20份平均。詳情參見 'examples/data/mnist_csv/'。

```bash
python examples/pipeline/homo_nn/fldpcr/pipeline-upload-mnist.py
```

### 3.3 啓動隱私訓練

**執行本項目所提供的FLDPCR學習示例。（使用SimpleMech，相當於傳統的差分隱私）**

```bash
python examples/pipeline/homo_nn/fldpcr/demo.py --nIter 40 --gpuId 0
```

若沒有GPU或未安裝GPU運行庫，建議執行以下CPU版本的示例：

```bash
python examples/pipeline/homo_nn/fldpcr/demo.py --nIter 3 --gpuId -1
```

#### 實驗結果對比：

- 對於nIter=40，使用傳統差分隱私的情況（dpcrModel=SimpleMech），命令如下：

```bash
python examples/pipeline/homo_nn/fldpcr/demo.py --nIter 40 --gpuId 0
```

實驗結果如下：
![image](https://github.com/imcjp/FLDPCR4FATE/blob/main/assets/expResults/loss_SimpleMech.png)

可以看出傳統差分隱私學習方法效果並不好，40次訓練下來Loss仍高達**3.41**。

- 對於nIter=40，使用我們提出的BCRG方法（dpcrModel=BCRG，k=12），命令如下：

```bash
python examples/pipeline/homo_nn/fldpcr/demo.py --nIter 40 --gpuId 0 --dpcrModel BCRG --k 12
```

實驗結果如下：
![image](https://github.com/imcjp/FLDPCR4FATE/blob/main/assets/expResults/loss_BCRG.png)

可以看出BCRG已經極大降低了訓練的Loss值，40次訓練下來Loss降到了**0.68**。


- 對於nIter=40，使用BCRG方法的改進版本ABCRG（dpcrModel=ABCRG，k=12），命令如下：

```bash
python examples/pipeline/homo_nn/fldpcr/demo.py --nIter 40 --gpuId 0 --dpcrModel ABCRG --k 12
```

實驗結果如下：
![image](https://github.com/imcjp/FLDPCR4FATE/blob/main/assets/expResults/loss_ABCRG.png)

可以看出ABCRG進一步降低訓練的Loss值，40次訓練下來Loss進一步降到了**0.635**。

**實驗結論：我們提出的方法能夠顯著提升隱私聯邦學習的訓練效果。**

**demo.py腳步的參數說明如下：**

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
  Specify the DPCR model to use. Choices are: `SimpleMech` (equivalent to traditional DP, ignore parameter $k$), `TwoLevel`, `BinMech`, `FDA`, `BCRG`, `ABCRG`. For `TwoLevel`, $T=k^2$; for `BinMech`, $T=2^k$; for `FDA`, `BCRG`, and `ABCRG`, $T=2^k-1$. If the actual number of iterations exceeds T, a new DPCR model will be constructed for training.

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
