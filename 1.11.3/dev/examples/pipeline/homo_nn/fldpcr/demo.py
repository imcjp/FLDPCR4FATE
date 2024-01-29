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

import argparse

import torch as t
import torch.nn as nn
from pipeline import fate_torch_hook
from pipeline.backend.pipeline import PipeLine
from pipeline.component import Reader, DataTransform, HomoNN, Evaluation
from pipeline.component.nn import TrainerParam, DatasetParam
from pipeline.interface import Data
from pipeline.runtime.entity import JobParameters

# TODO desc:偷梁换柱，把原生的torch替换为FATE内置的pipeline.component.nn.backend.torch
fate_torch_hook(t)
#################################
import yaml

arbiter0 = 100
host0 = 200
guest0 = 300

print('打开FATE board查看进程：')
print('http://localhost:8080')


def main(args):
    arbiter = arbiter0
    nParty = args.nClient
    nGuestParty = 1
    nHostParty = nParty - nGuestParty
    print(f'参与者一共有{nParty}个...')
    guests = [guest0 + i for i in range(nGuestParty)]
    hosts = [host0 + i for i in range(nHostParty)]
    # set_initiator可能是设置发起者
    pipeline = PipeLine().set_initiator(role='guest', party_id=guests[0]).set_roles(guest=guests, host=hosts,
                                                                                    arbiter=arbiter)

    reader_0 = Reader(name="reader_0")
    pt = 1
    for i in range(nGuestParty):
        fname = f"mnist_train_part_{pt}"
        print(fname)
        train_data = {"name": fname, "namespace": "cjp"}
        reader_0.get_party_instance(role='guest', party_id=guests[i]).component_param(table=train_data)
        pt += 1
    for i in range(nHostParty):
        fname = f"mnist_train_part_{pt}"
        if pt == 12:
            print('dd')
        print(fname)
        train_data = {"name": fname, "namespace": "cjp"}
        reader_0.get_party_instance(role='host', party_id=hosts[i]).component_param(table=train_data)
        pt += 1

    data_transform_0 = DataTransform(name='data_transform_0', with_label=True, output_format="dense")

    model = t.nn.Sequential(t.nn.CustModel(module_name='reshape', class_name='ReshapeLayer', s1=28),
                            t.nn.CustModel(module_name='mnist', class_name='CNN'))
    loss = nn.CrossEntropyLoss()
    optimizer = t.optim.SGD(model.parameters(), lr=0.01)

    dpConf={
        'max_norm': args.dp_norm,
        'sigma': args.dp_sigma,
        'delta': args.dp_delta
    }
    if args.dpcrModel == 'SimpleMech':
        algConf={'name':args.dpcrModel, 'T':1}
    else:
        algConf={'name':args.dpcrModel, 'kOrder':args.k}

    nn_component = HomoNN(name=f'fldpcr_{args.dpcrModel}',
                          model=model,
                          loss=loss,
                          optimizer=optimizer,
                          trainer=TrainerParam(trainer_name='fldpcr', epochs=args.nIter,
                                               batch_size=args.batch_size,
                                               algConf=algConf, dpConf=dpConf,
                                               validation_freqs=1, cuda=args.gpuId if args.gpuId != -1 else None,
                                               task_type='multi'),
                          # reshape and set label to long for CrossEntropyLoss
                          dataset=DatasetParam(dataset_name='table', flatten_label=True, label_dtype='long'),
                          torch_seed=args.seed
                          )

    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(nn_component, data=Data(train_data=data_transform_0.output.data))
    pipeline.add_component(Evaluation(name='eval_0', eval_type='multi'), data=Data(data=nn_component.output.data))

    pipeline.compile()
    print(yaml.dump(pipeline._train_conf['component_parameters']))
    jm = JobParameters(task_cores=0)
    pipeline.fit(job_parameters=jm)

def parse_args():
    # 创建解析器
    parser = argparse.ArgumentParser(description='Process parameters for federated learning with differential privacy.')

    # 添加参数
    parser.add_argument('--nClient', type=int, default=5, help='Number of clients for federated learning.')
    parser.add_argument('--dpcrModel', type=str, default='ABCRG',
                        choices=['SimpleMech', 'TwoLevel', 'BinMech', 'FDA', 'BCRG', 'ABCRG'],
                        help='DPCR model. Choices are: SimpleMech (equivalent to traditional DP), TwoLevel, BinMech, FDA, BCRG, ABCRG. '
                             'For TwoLevel, T=k^2; for BinMech, T=2^k; for FDA, BCRG, and ABCRG, T=2^k-1. '
                             'If the actual number of iterations exceeds T, a new DPCR model will be constructed for training.')
    parser.add_argument('--nIter', type=int, default=40, help='Number of iterations.')
    parser.add_argument('--batch_size', type=int, default=30, help='Batch size.')
    parser.add_argument('--dp_sigma', type=float, default=2.19, help='Differential privacy noise scale.')
    parser.add_argument('--dp_delta', type=float, default=3.33e-4, help='Differential privacy delta parameter.')
    parser.add_argument('--dp_norm', type=float, default=1.0, help='Maximum clipping norm for differential privacy.')
    parser.add_argument('--gpuId', type=int, default=-1, help='GPU ID to use (if any), otherwise -1 for CPU.')
    parser.add_argument('--k', type=int, default=1, help='Model size parameter k.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility.')

    # 解析参数
    args = parser.parse_args()

    return args


if __name__ == "__main__":


    # 解析命令行参数
    args = parse_args()

    # 打印参数值，确认是否正确解析
    print(f'nClient: {args.nClient}')
    print(f'dpcrModel: {args.dpcrModel}')
    print(f'nIter: {args.nIter}')
    print(f'batch_size: {args.batch_size}')
    print(f'dp_sigma: {args.dp_sigma}')
    print(f'dp_delta: {args.dp_delta}')
    print(f'dp_norm: {args.dp_norm}')
    print(f'gpuId: {args.gpuId}')
    print(f'k: {args.k}')

    # 根据参数进行后续操作...
    # 例如，可以根据gpuId判断是否使用GPU
    if args.gpuId >= 0:
        print(f'Using GPU with ID: {args.gpuId}')
    else:
        print('Using CPU')

    main(args)
