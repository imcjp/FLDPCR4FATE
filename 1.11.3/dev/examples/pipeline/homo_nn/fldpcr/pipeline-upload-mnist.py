#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import os

# 获取当前运行脚本的绝对路径
script_path = os.path.abspath(__file__)
# 获取脚本所在的目录
script_dir = os.path.dirname(script_path)
# 将工作目录切换到脚本所在目录
os.chdir(script_dir)

from pipeline.backend.pipeline import PipeLine
from pipeline.utils.tools import load_job_config
import json


def main(namespace=""):
    guest = 300

    pipeline_upload = PipeLine().set_initiator(role="guest", party_id=guest).set_roles(guest=guest)


    with open('data_testsuite.json', 'r') as file:
        data = json.load(file)

    batch_names = ['mnist_train']
    for info in data['dataBatch']:
        if info['batch_name'] in batch_names:
            s = info['id_from']
            t = info['id_end']
            step = info['id_step'] if 'id_step' in info else 1
            for j in range(s, t + 1, step):
                table_name = info['table_name'].format(id=j)
                namespace = info["namespace"].format(id=j)
                file= info['file'].format(id=j)
                print(f'{namespace}.{table_name}: {file}')
                pipeline_upload.add_upload_data(file=file,
                                                table_name=table_name,  # table name
                                                namespace=namespace,  # namespace
                                                head=info["head"], partition=info["partition"],  # data info
                                                id_delimiter=",")

    # upload both data
    pipeline_upload.upload(drop=1)


if __name__ == "__main__":
    main()
