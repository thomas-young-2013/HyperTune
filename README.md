<meta name="robots" content="noindex">

# Hyper-Tune

**Hyper-Tune: an Efficient Hyper-parameter Tuning at Scale**

## Experimental Environment Installation

Note that in our experiments, the operating system is Ubuntu 18.04.3 LTS.
We use **xgboost==1.3.1** and **torch==1.7.1** (torchvision==0.7.0, CUDA Version 10.1.243).
The configuration space is defined using **ConfigSpace==0.4.18**.
The multi-fidelity surrogate in our method is implemented based on probabilistic random forest in [SMAC3](https://github.com/automl/SMAC3),
which depends on **pyrfr==0.8.0**. (included in requirements.txt)

In our paper, we use Pytorch to train neural networks on 32 RTX 2080Ti GPUs,
and the experiments are conducted on ten machines with 640 AMD EPYC 7702P CPU cores in total (64 cores, 128 threads each).

1. preparations: Python == 3.7
2. install SWIG:
    ```
    apt-get install swig3.0
    ln -s /usr/bin/swig3.0 /usr/bin/swig
    ```
3. install requirements:
    ```
    cat requirements.txt | xargs -n 1 -L 1 pip install
    ```

## Data Preparation

### XGBoost

+ Download 4 datasets (Covertype, Pokerhand, Hepmass, Higgs) for XGBoost tuning experiments:
  + https://archive.ics.uci.edu/ml/datasets/Covertype
  + https://archive.ics.uci.edu/ml/datasets/Poker+Hand
  + https://archive.ics.uci.edu/ml/datasets/HEPMASS
  + https://archive.ics.uci.edu/ml/datasets/HIGGS
+ Preprocess data (including train-test split) with `test/preprocess_data.py`.
+ Put datasets (`.npy` files) under `./datasets/`.

### NAS-Bench-201

+ Follow the instructions of <https://github.com/D-X-Y/NAS-Bench-201>
and download benchmark file of NAS-Bench-201(`NAS-Bench-201-v1_1-096897.pth`).

### ResNet

+ Download `cifar10.zip`(preprocessed) from [Google Drive](https://drive.google.com/file/d/1TVY0nXLHsjqPUXm8TmkOv9qcaRX7mQHH)
or [Baidu-Wangpan (code:t47a)](https://pan.baidu.com/s/1Ie_CKtJaJjddY0oA6I6Sng).
(Note that at present, we only provide downloading from Baidu-Wangpan because Google Drive is not an anonymous service.)
+ Unzip `cifar10.zip` and put it under `./datasets/img_datasets/` (the path should be `./datasets/img_datasets/cifar10/`).

### LSTM

+ We implement LSTM based on <https://github.com/salesforce/awd-lstm-lm> to conduct our experiments.
Please follow the instructions in project readme and use `getdata.sh` to to acquire the Penn Treebank dataset.
+ Put dataset (`.txt` files) under `./test/awd_lstm_lm/data/penn/`


# Documentations

## Project Code Overview

+ `tuner/` : the implemented method and compared baselines.
+ `test/` : the python scripts in the experiments, and useful tools.

## Experiments Design

See `tuner/__init__.py` to get the name of each baseline method. (Keys of `mth_dict`)

Compared methods are listed as follows:

| Method | String of `${method_name}`|
| --- | --- |
| Batch BO | `bo` |
| Successive Halving | `sh` |
| Hyperband | `hyperband` |
| BOHB | `bohb` |
| MFES-HB | `mfeshb` |
| A-Random | `arandom` |
| A-BO | `abo` |
| A-REA | `area` |
| ASHA | `asha` |
| A-BOHB | `abohb_aws` (see the <font color=#FF0000>**Note**</font> below) |
| A-Hyperband | `ahyperband` |
| ours | `tuner` |

<font color=#FF0000>**Note**</font>: To run A-BOHB(`abohb_aws`) implemented in **Autogluon**(<https://github.com/awslabs/autogluon>),
please install the corresponding environment and follow the instructions at the last of this document.

### Exp.1: Compare methods on Nas-Bench-201

Exp settings:
+ n_workers=8, rep=10.
+ `cifar10-valid`: runtime_limit=86400
+ `cifar100`: runtime_limit=172800
+ `ImageNet16-120`: runtime_limit=432000

Compared methods: `bo`, `sh`, `hyperband`, `bohb`, `mfeshb`, `arandom`, `area`, `abo`, `asha`, `ahyperband`, `abohb_aws`(See the last of this document), `tuner`

To conduct the simulation experiment shown in Figure 5, the script is as follows.
Please specify `${dataset_name}`, `${runtime_limit}`, `${method_name}`:

```
python test/nas_benchmarks/benchmark_nasbench201.py --data_path './NAS-Bench-201-v1_1-096897.pth' --dataset ${dataset_name} --runtime_limit ${runtime_limit} --mths ${method_name} --R 27 --n_workers 8 --rep 10
```

### Exp.2: Compare methods on XGBoost

Exp settings:
+ n_workers=8, rep=10.
+ `covtype`(Covertype): runtime_limit=10800
+ `pokerhand`(Pokerhand): runtime_limit=7200
+ `hepmass`(Hepmass): runtime_limit=43200
+ `HIGGS`(Higgs): runtime_limit=43200

Compared methods: `bo`, `sh`, `hyperband`, `bohb`, `mfeshb`, `arandom`, `abo`, `asha`, `ahyperband`, `abohb_aws`(See the last of this document), `tuner`

To conduct the experiment shown in Figure 7, the script is as follows.
Please specify `${dataset_name}`, `${runtime_limit}`, `${method_name}`:

```
python test/benchmark_xgb.py --datasets ${dataset_name} --runtime_limit ${runtime_limit} --mth ${method_name} --R 27 --n_workers 8 --rep 10
```

Please make sure there are enough CPUs on the machine.

### Exp.3: Compare methods on LSTM and ResNet

Exp settings:
+ n_workers=4, rep=10.
+ `penn`(Penn Treebank for LSTM): runtime_limit=172800
+ `cifar10`(for ResNet): runtime_limit=172800

Compared methods: `sh`, `hyperband`, `bohb`, `mfeshb`, `asha`, `ahyperband`, `abohb_aws`(See the last of this document), `tuner`

To conduct the experiment shown in Figure 6(a), the script is as follows:
```
python test/awd_lstm_lm/benchmark_lstm.py --dataset penn --runtime_limit ${runtime_limit} --mth ${method_name} --R 27 --n_workers 4 --rep 10
```

To conduct the experiment shown in Figure 6(b), the script is as follows:
```
python test/resnet/benchmark_resnet.py --dataset cifar10 --runtime_limit ${runtime_limit} --mth ${method_name} --R 27 --n_workers 4 --rep 10
```

Please specify `${runtime_limit}`, `${method_name}`.

### Exp.4: Test robustness of partial evaluations on noised Hartmann

Exp settings:
+ n_workers=8, rep=10.
+ `hartmann`(noised math function): runtime_limit=1080
+ `noise_alpha`: 0, 100, 10000 (corresponding to 0, 40, 4000 in our paper)

Compared methods: `asha`(with different initial resource), `abohb_aws`(See the last of this document), `tuner`

To conduct the simulation experiment shown in Figure 8, the script are as follows.
Please specify `${noise_alpha}`:

+ run `asha` with different initial resource (e.g. `--R 9` means the initial resource is 1/9):
```
python test/math_benchmarks/benchmark_math.py --dataset hartmann --noise_alpha ${noise_alpha} --runtime_limit 1080 --mths asha --R 1 --n_workers 8 --rep 10
python test/math_benchmarks/benchmark_math.py --dataset hartmann --noise_alpha ${noise_alpha} --runtime_limit 1080 --mths asha --R 3 --n_workers 8 --rep 10
python test/math_benchmarks/benchmark_math.py --dataset hartmann --noise_alpha ${noise_alpha} --runtime_limit 1080 --mths asha --R 9 --n_workers 8 --rep 10
python test/math_benchmarks/benchmark_math.py --dataset hartmann --noise_alpha ${noise_alpha} --runtime_limit 1080 --mths asha --R 27 --n_workers 8 --rep 10
```

+ run `tuner`:
```
python test/math_benchmarks/benchmark_math.py --dataset hartmann --noise_alpha ${noise_alpha} --runtime_limit 1080 --mths tuner --R 27 --n_workers 8 --rep 10
```

### Exp.5: Test scalability on workers

Exp settings:
+ rep=10.
+ `n_workers`: 1, 2, 4, 8, 16, 32, 64. (128, 256 for Counting Ones)

Compared method: `tuner`(with different n_workers)

To conduct the experiment shown in Figure 10, the script are as follows.
Please specify `${n_workers}`:

+ Nas-Bench-201 on cifar100: runtime_limit=172800
```
python test/nas_benchmarks/benchmark_nasbench201.py --data_path './NAS-Bench-201-v1_1-096897.pth' --dataset cifar100 --runtime_limit 172800 --mths tuner --R 27 --n_workers ${n_workers} --rep 10
```

+ Counting Ones function on 32+32 dimensions: runtime_limit=5400
```
python test/math_benchmarks/benchmark_math.py --dataset counting-32-32 --runtime_limit 5400 --mths tuner --noise 0 --R 27 --n_workers ${n_workers} --rep 10
```

+ XGBoost on Covertype: runtime_limit=10800
```
python test/benchmark_xgb.py --datasets covtype --runtime_limit 10800 --mth tuner --R 27 --n_workers ${n_workers} --rep 10
```

<font color=#FF0000>**Note**</font>: if you do not have enough CPUs on one machine to run the experiment with n_workers=16 (which requires 16*16 CPUs),
you can run on multiple machines by the following commands:

+ First, start the master node with some local workers (e.g. 8 local workers, need 16 workers in total).
```
python test/benchmark_xgb.py --n_jobs 16 --datasets covtype --runtime_limit 10800 --mth tuner --R 27 --n_workers 16 --max_local_workers 8 --port 13579 --rep 1 --start_id 0
```

+ Then, start the worker nodes with more workers (e.g. 1 worker node with 8 workers). Please specify IP and port of master node.
```
python test/benchmark_xgb_worker.py --n_jobs 16 --parallel async --dataset covtype --R 27 --n_workers 8 --ip ${master_ip} --port 13579
```

+ In this example, experiment is conducted only once. Please specify `--start_id` to run experiment multiple times with different random seeds.

### Exp.6: Ablation study

In ablation study, the compared experimental methods are as follows:

| Method | String of `${method_name}`|
| --- | --- |
| A-Hyperband with bracket selection | `ahyperband_bs` |
| A-BOHB*(our implementation) | `abohb` |
| A-BOHB* with bracket selection | `abohb_bs` |
| ours without bracket selection | `tuner_exp1` |
| delayed ASHA | `asha_delayed` |
| A-Hyperband with delayed ASHA | `ahyperband_delayed` |
| ours with original ASHA | `tuner_exp2` |

To conduct the simulation experiment shown in Figure 9(a), the script are as follows.
```
python test/nas_benchmarks/benchmark_nasbench201.py --data_path './NAS-Bench-201-v1_1-096897.pth' --dataset cifar10-valid --runtime_limit 86400 --mths ahyperband,ahyperband_bs,abohb,abohb_bs,tuner_exp1,tuner --R 27 --n_workers 8 --rep 10
```

To conduct the simulation experiment shown in Figure 9(b), the script are as follows.
```
python test/nas_benchmarks/benchmark_nasbench201.py --data_path './NAS-Bench-201-v1_1-096897.pth' --dataset ImageNet16-120 --runtime_limit 432000 --mths ahyperband,ahyperband_bs,abohb,abohb_bs,tuner_exp1,tuner --R 27 --n_workers 8 --rep 10
```

To conduct the experiment shown in Figure 9(c), the script are as follows.
```
python test/benchmark_xgb.py --datasets covtype --runtime_limit 10800 --mth asha --R 27 --n_workers 8 --rep 10
python test/benchmark_xgb.py --datasets covtype --runtime_limit 10800 --mth asha_delayed --R 27 --n_workers 8 --rep 10
python test/benchmark_xgb.py --datasets covtype --runtime_limit 10800 --mth ahyperband --R 27 --n_workers 8 --rep 10
python test/benchmark_xgb.py --datasets covtype --runtime_limit 10800 --mth ahyperband_delayed --R 27 --n_workers 8 --rep 10
python test/benchmark_xgb.py --datasets covtype --runtime_limit 10800 --mth tuner_exp2 --R 27 --n_workers 8 --rep 10
python test/benchmark_xgb.py --datasets covtype --runtime_limit 10800 --mth tuner --R 27 --n_workers 8 --rep 10
```

To conduct the experiment shown in Figure 9(d), the script are as follows.
```
python test/benchmark_xgb.py --datasets pokerhand --runtime_limit 7200 --mth asha --R 27 --n_workers 8 --rep 10
python test/benchmark_xgb.py --datasets pokerhand --runtime_limit 7200 --mth asha_delayed --R 27 --n_workers 8 --rep 10
python test/benchmark_xgb.py --datasets pokerhand --runtime_limit 7200 --mth ahyperband --R 27 --n_workers 8 --rep 10
python test/benchmark_xgb.py --datasets pokerhand --runtime_limit 7200 --mth ahyperband_delayed --R 27 --n_workers 8 --rep 10
python test/benchmark_xgb.py --datasets pokerhand --runtime_limit 7200 --mth tuner_exp2 --R 27 --n_workers 8 --rep 10
python test/benchmark_xgb.py --datasets pokerhand --runtime_limit 7200 --mth tuner --R 27 --n_workers 8 --rep 10
```

### Special instruction: run A-BOHB with Autogluon

To run the baseline method A-BOHB(`abohb_aws`) in the experiments, please install **Autogluon**(<https://github.com/awslabs/autogluon>).
And we provide scripts in `test/autogluon_abohb/`. Usages are as follows.

<font color=#FF0000>**Note**</font>: **Autogluon** uses `--num_cpus` and `--num_gpus` to infer number of workers. `--n_workers` is just for method naming.
Please set appropriate `--num_cpus` and `--num_gpus` according to your machine to limit the number of workers.

<font color=#FF0000>**Note**</font>: Please specify `--start_id` to run experiment multiple times with different random seeds.

+ Run Nas-Bench-201. Please specify `${runtime_limit}` and `${dataset}`:
```
python test/autogluon_abohb/benchmark_autogluon_abohb_nasbench201.py --R 27 --reduction_factor 3 --brackets 4 --num_cpus 16 --n_workers 8 --timeout ${runtime_limit} --dataset ${dataset} --rep 1 --start_id 0
```

+ Run XGBoost. Please specify `${runtime_limit}` and `${dataset}`:
```
python test/autogluon_abohb/benchmark_autogluon_abohb_xgb.py --R 27 --reduction_factor 3 --brackets 4 --num_cpus 16 --n_workers 8 --n_jobs 16 --timeout ${runtime_limit} --dataset ${dataset} --rep 1 --start_id 0
```

+ Run LSTM:
```
python test/autogluon_abohb/benchmark_autogluon_abohb_lstm.py --R 27 --reduction_factor 3 --brackets 4 --num_gpus 1 --n_workers 4 --timeout 172800 --dataset penn --rep 1 --start_id 0
```

+ Run ResNet:
```
python test/autogluon_abohb/benchmark_autogluon_abohb_resnet.py --R 27 --reduction_factor 3 --brackets 4 --num_gpus 1 --n_workers 4 --timeout 172800 --dataset cifar10 --rep 1 --start_id 0
```

+ Run noised Hartmann. Please specify `${noise_alpha}`:
```
python test/autogluon_abohb/benchmark_autogluon_abohb_math.py --R 27 --reduction_factor 3 --brackets 4 --num_cpus 16 --n_workers 8 --timeout 1080 --dataset hartmann --noise_alpha ${noise_alpha} --rep 1 --start_id 0
```

