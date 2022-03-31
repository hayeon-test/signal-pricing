# Temporal Fusion Transformer For PyTorch on k8s

## Table Of Contents

- [Temporal Fusion Transformer For PyTorch on k8s](#temporal-fusion-transformer-for-pytorch-on-k8s)
  - [Table Of Contents](#table-of-contents)
  - [Model overview](#model-overview)
    - [Model architecture](#model-architecture)
    - [Default configuration](#default-configuration)
    - [Feature support matrix](#feature-support-matrix)
      - [Features](#features)
    - [Mixed precision training](#mixed-precision-training)
      - [Enabling mixed precision](#enabling-mixed-precision)
      - [Enabling TF32](#enabling-tf32)
    - [Glossary](#glossary)
  - [Setup](#setup)
    - [Requirements](#requirements)
  - [Quick Start Guide](#quick-start-guide)
  - [Advanced](#advanced)
    - [Scripts and sample code](#scripts-and-sample-code)
    - [Command-line options](#command-line-options)
    - [Getting the data](#getting-the-data)
      - [Dataset guidelines](#dataset-guidelines)
      - [Multi-dataset](#multi-dataset)
    - [Training process](#training-process)
    - [Inference process (TODO)](#inference-process-todo)
    - [Triton deployment (TODO)](#triton-deployment-todo)

## Model overview

The Temporal Fusion Transformer [TFT](https://arxiv.org/abs/1912.09363) model is a state-of-the-art architecture for interpretable, multi-horizon time-series prediction. The model was first developed and [implemented by Google](https://github.com/google-research/google-research/tree/master/tft) with the collaboration with the University of Oxford.
This implementation differs from the reference implementation by addressing the issue of missing data, which is common in production datasets, by either masking their values in attention matrices or embedding them as a special value in the latent space.
This model enables the prediction of confidence intervals for future values of time series for multiple future timesteps.

This model is trained with mixed precision using Tensor Cores on Volta, Turing, and the NVIDIA Ampere GPU architectures. Therefore, researchers can get results 1.45x faster than training without Tensor Cores while experiencing the benefits of mixed precision training. This model is tested against each NGC monthly container release to ensure consistent accuracy and performance over time.

### Model architecture

The TFT model is a hybrid architecture joining LSTM encoding of time series and interpretability of transformer attention layers. Prediction is based on three types of variables: static (constant for a given time series), known (known in advance for whole history and future), observed (known only for historical data). All these variables come in two flavors: categorical, and continuous. In addition to historical data, we feed the model with historical values of time series. All variables are embedded in high-dimensional space by learning an embedding vector. Categorical variables embeddings are learned in the classical sense of embedding discrete values. The model learns a single vector for each continuous variable, which is then scaled by this variable’s value for further processing. The next step is to filter variables through the Variable Selection Network (VSN), which assigns weights to the inputs in accordance with their relevance to the prediction. Static variables are used as a context for variable selection of other variables and as an initial state of LSTM encoders.
After encoding, variables are passed to multi-head attention layers (decoder), which produce the final prediction. Whole architecture is interwoven with residual connections with gating mechanisms that allow the architecture to adapt to various problems by skipping some parts of it.
For the sake of explainability, heads of self-attention layers share value matrices. This allows interpreting self-attention as an ensemble of models predicting different temporal patterns over the same feature set. The other feature that helps us understand the model is VSN activations, which tells us how relevant the given feature is to the prediction.
![](TFT_architecture.PNG)
_image source: https://arxiv.org/abs/1912.09363_

### Default configuration

The specific configuration of the TFT model depends on the dataset used. Not only is the volume of the model subject to change but so are the data sampling and preprocessing strategies. During preprocessing, data is normalized per feature. For a part of the datasets, we apply scaling per-time-series, which takes into account shifts in distribution between entities (i.e., a factory consumes more electricity than an average house). The model is trained with the quantile loss: <img src="https://render.githubusercontent.com/render/math?math=\Large\sum_{i=1}^N\sum_{q\in\mathcal{Q}}\sum_{t=1}^{t_{max}}\frac{QL(y_it,\hat{y}_i(q,t),q)}{Nt_{max}}">
For quantiles in [0.1, 0.5, 0.9]. The default configurations are tuned for distributed training on DGX-1-32G with mixed precision. We use dynamic loss scaling. Specific values are provided in the table below.

| Dataset                 | Training samples | Validation samples | Test samples | History length | Forecast horizon | Dropout | Hidden size | #Heads | BS     | LR   | Gradient clipping |
| ----------------------- | ---------------- | ------------------ | ------------ | -------------- | ---------------- | ------- | ----------- | ------ | ------ | ---- | ----------------- |
| Alphavantage Stock Data | 450k             | 50k                | 53.5k        | 168            | 24               | 0.1     | 128         | 4      | 8x1024 | 1e-3 | 0.0               |

### Feature support matrix

The following features are supported by this model:

| Feature                   | Yes column |
| ------------------------- | ---------- |
| Distributed data parallel | Yes        |
| PyTorch AMP               | Yes        |

#### Features

[Automatic Mixed Precision](https://pytorch.org/docs/stable/amp.html)
provides an easy way to leverage Tensor Cores’ performance. It allows the execution of parts of a network in lower precision. Refer to [Mixed precision training](#mixed-precision-training) for more information.

[PyTorch
DistributedDataParallel](https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel) - a module
wrapper that enables easy multiprocess distributed data-parallel
training.

### Mixed precision training

Mixed precision is the combined use of different numerical precisions in a
computational method.
[Mixed precision](https://arxiv.org/abs/1710.03740) training offers significant
computational speedup by performing operations in half-precision format while
storing minimal information in single-precision to retain as much information
as possible in critical parts of the network. Since the introduction of [Tensor Cores](https://developer.nvidia.com/tensor-cores) in Volta, and following with
both the Turing and Ampere architectures, significant training speedups are
experienced by switching to
mixed precision -- up to 3x overall speedup on the most arithmetically intense
model architectures. Using mixed precision training previously required two
steps:

1. Porting the model to use the FP16 data type where appropriate.
2. Manually adding loss scaling to preserve small gradient values.

The ability to train deep learning networks with lower precision was introduced
in the Pascal architecture and first supported in [CUDA
8](https://devblogs.nvidia.com/parallelforall/tag/fp16/) in the NVIDIA Deep
Learning SDK.

For information about:

- How to train using mixed precision, refer to the [Mixed Precision
  Training](https://arxiv.org/abs/1710.03740) paper and [Training With Mixed
  Precision](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html)
  documentation.
- Techniques used for mixed precision training, refer to the [Mixed-Precision
  Training of Deep Neural
  Networks](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/)
  blog.
- APEX tools for mixed precision training, refer to the [NVIDIA Apex: Tools for Easy Mixed-Precision Training in
  PyTorch](https://devblogs.nvidia.com/apex-pytorch-easy-mixed-precision-training/)
  .

#### Enabling mixed precision

Mixed precision is enabled in PyTorch by using the Automatic Mixed Precision torch.cuda.amp module, which casts variables to half-precision upon retrieval while storing variables in single-precision format. Furthermore, to preserve small gradient magnitudes in backpropagation, a [loss scaling](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html#lossscaling) step must be included when applying gradients. In PyTorch, loss scaling can be applied automatically by the GradScaler class. All the necessary steps to implement AMP are verbosely described [here](https://pytorch.org/docs/stable/notes/amp_examples.html#amp-examples).

To enable mixed precision for TFT, simply add the `--use_amp` option to the training script.

#### Enabling TF32

TensorFloat-32 (TF32) is the new math mode in [NVIDIA A100](https://www.nvidia.com/en-us/data-center/a100/) GPUs for handling the matrix math, also called tensor operations. TF32 running on Tensor Cores in A100 GPUs can provide up to 10x speedups compared to single-precision floating-point math (FP32) on Volta GPUs.

TF32 Tensor Cores can speed up networks using FP32, typically with no loss of accuracy. It is more robust than FP16 for models which require high dynamic range for weights or activations.

For more information, refer to the [TensorFloat-32 in the A100 GPU Accelerates AI Training, HPC up to 20x](https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/) blog post.

TF32 is supported in the NVIDIA Ampere GPU architecture and is enabled by default.

### Glossary

**Multi horizon prediction**  
Process of estimating values of a time series for multiple future time steps.

**Quantiles**  
Cut points dividing the range of a probability distribution intervals with equal probabilities.

**Time series**  
Series of data points indexed and equally spaced in time.

**Transformer**  
The paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762) introduces a novel architecture called Transformer that uses an attention mechanism and transforms one sequence into another.

## Setup

The following section lists the requirements that you need to meet in order to start training the TFT model.

### Requirements

This repository contains Dockerfile, which extends the PyTorch NGC container and encapsulates some dependencies. Aside from these dependencies, ensure you have the following components:

- 1.19 <= Kubernetes <=1.21
- [Helm 3](https://helm.sh/docs/intro/install/)
- [Free or Enterprise NGC API Key](https://catalog.ngc.nvidia.com/)
- [Free Alpha Vantage API Key](https://www.alphavantage.co/support/#api-key)
- Supported GPUs:
  - [NVIDIA Volta architecture](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/)
  - [NVIDIA Turing architecture](https://www.nvidia.com/en-us/design-visualization/technologies/turing-architecture/)
  - [NVIDIA Ampere architecture](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/)

For more information about how to get started with NGC containers, refer to the following sections from the NVIDIA GPU Cloud Documentation and the Deep Learning Documentation:

- [Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html)
- [Accessing And Pulling From The NGC Container Registry](https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html#accessing_registry)
- Running [PyTorch](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/running.html#running)

For those unable to use the PyTorch NGC container to set up the required environment or create your own container, refer to the versioned [NVIDIA Container Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).

## Quick Start Guide

To train your model using mixed or TF32 precision with Tensor Cores, perform the following steps using the default parameters of the TFT model on any of the benchmark datasets. For the specifics concerning training and inference, refer to the [Advanced](#advanced) section.

1. Build the Docker images
   ```bash
    docker-compose build model-training feature-engineering
   ```
2. Publish the Docker images somwehere...

```bash
docker-compose push model-training feature-engineering
```

3. Launch the service through Helm. This will launch data downloading, processing, training and monitoring. The end goal of this would be an end-to-end example of online learning. Where the model may continue to learn as new data are streamed to the /data source. The example of stock data is contriveed but Alpha Vantage offers a very stable API for example purposes.

```bash
helm install signal-pricing signal-pricing \
  --set imageCredentials.password=<NGC_API_KEY> \
  --set imageCredentials.email=<NGC_USER_EMAIL> \
  --set envVars.ALPHA_VANTAGE_API_KEY=<ALPHA_VANTAGE_API_KEY>
```

Now that you have your model trained and evaluated, you can choose to compare your training results with our [Training accuracy results](#training-accuracy-results). You can also choose to benchmark your performance to [Training performance benchmark](#training-performance-results). Following the steps in these sections will ensure that you achieve the same accuracy and performance results as stated in the [Results](#results) section.

## Advanced

The following sections provide more details about the dataset, running training and inference, and the training results.

### Scripts and sample code

In the root directory, the most important files are:

`train.py`: Entry point for training
`data_utils.py`: File containing the dataset implementation and preprocessing functions
`modeling.py`: Definition of the model
`configuration.py`: Contains configuration classes for various experiments
`test.py`: Entry point testing trained model.
`Dockerfile`: Container definition
`log_helper.py`: Contains helper functions for setting up dllogger
`criterions.py`: Definitions of loss functions

### Command-line options

To view the full list of available options and their descriptions, use the `-h` or `--help` command-line option, for example:
`python train.py --help`.

The following example output is printed when running the model:

```
usage: train.py [-h] --data_path DATA_PATH --dataset {electricity,traffic} [--epochs EPOCHS] [--sample_data SAMPLE_DATA SAMPLE_DATA] [--batch_size BATCH_SIZE] [--lr LR] [--seed SEED] [--use_amp] [--clip_grad CLIP_GRAD]
                [--early_stopping EARLY_STOPPING] [--results RESULTS] [--log_file LOG_FILE] [--distributed_world_size N] [--distributed_rank DISTRIBUTED_RANK] [--local_rank LOCAL_RANK] [--overwrite_config OVERWRITE_CONFIG]

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
  --dataset {electricity,traffic}
  --epochs EPOCHS
  --sample_data SAMPLE_DATA SAMPLE_DATA
  --batch_size BATCH_SIZE
  --lr LR
  --seed SEED
  --use_amp             Enable automatic mixed precision
  --clip_grad CLIP_GRAD
  --early_stopping EARLY_STOPPING
                        Stop training if validation loss does not improve for more than this number of epochs.
  --results RESULTS
  --log_file LOG_FILE
  --distributed_world_size N
                        total number of GPUs across all nodes (default: all visible GPUs)
  --distributed_rank DISTRIBUTED_RANK
                        rank of the current worker
  --local_rank LOCAL_RANK
                        rank of the current worker
  --overwrite_config OVERWRITE_CONFIG
                        JSON string used to overload config

```

### Getting the data

The TFT model was trained on the stock pricing datasets. This repository contains the `get_data.sh` download script, which for electricity and and traffic datasets will automatically download and preprocess the training, validation and test datasets, and produce files that contain scalers. This is part of the Helm deploy as an initContainer.

#### Dataset guidelines

The `data_utils.py` file contains all functions that are used to preprocess the data. Initially the data is loaded to a `pandas.DataFrame` and parsed to the common format which contains the features we will use for training. Then standardized data is cleaned, normalized, encoded and binarized.
This step does the following:
Drop all the columns that are not marked in the configuration file as used for training or preprocessing
Flatten indices in case time series are indexed by more than one column
Split the data into training, validation and test splits
Filter out all the time series shorter than minimal example length
Normalize columns marked as continuous in the configuration file
Encode as integers columns marked as categorical
Save the data in csv and binary formats

#### Multi-dataset

In order to use an alternate dataset, you have to write a function that parses your data to a common format. The format is as follows:
There is at least one id column
There is exactly one time column (that can also be used as a feature column)
Each feature is in a separate column
Each row represents a moment in time for only one time series
Additionally, you must specify a configuration of the network, including a data description. Refer to the example in `configuration.py` file.

### Training process

The `train.py` script is an entry point for a training procedure. Refined recipes can be found in the `scripts` directory.
The model trains for at most `--epochs` epochs. If option `--early_stopping N` is set, then training will end if for N subsequent epochs validation loss hadn’t improved.
The details of the architecture and the dataset configuration are encapsulated by the `--dataset` option. This option chooses one of the configurations stored in the `configuration.py` file. You can enable mixed precision training by providing the `--use_amp` option. The training script supports multi-GPU training with the APEX package. To enable distributed training prepend training command with `python -m torch.distributed.run --nproc_per_node=${NGPU}`. All these are configurable within the `signal-pricing/templates/statefulset.yaml` file.

Example command:

```
python -m torch.distributed.run --nproc_per_node=8 train.py \
        --dataset electricity \
        --data_path /data/processed/electricity_bin \
        --batch_size=1024 \
        --sample 450000 50000 \
        --lr 1e-3 \
        --epochs 25 \
        --early_stopping 5 \
        --seed 1 \
        --use_amp \
        --results /results/TFT_electricity_bs8x1024_lr1e-3/seed_1
```

The model is trained by optimizing quantile loss <img src="https://render.githubusercontent.com/render/math?math=\Large\sum_{i=1}^N\sum_{q\in\mathcal{Q}}\sum_{t=1}^{t_{max}}\frac{QL(y_{it},\hat{y}_i(q,t),q)}{Nt_{max}}">
. After training, the checkpoint with the least validation loss is evaluated on a test split with q-risk metric <img src="https://render.githubusercontent.com/render/math?math=\Large\frac{2\sum_{y\in\Omega}\sum_{t=1}^{t_{max}}QL(y_t,\hat{y}(q,t),q)}{\sum_{y\in\Omega}\sum_{t=1}^{t_{max}}|y_t|}">.
Results are by default stored in the `/results` directory. This can be changed by providing the `--results` option. At the end of the training, the results directory will contain the trained checkpoint which had the lowest validation loss, dllogger logs (in dictionary per line format), and TensorBoard logs.

### Inference process (TODO)

Inference can be run by launching the `inference.py` script. The script requires a trained checkpoint to run. It is crucial to prepare the data in the same way as training data prior to running the inference. Example command:

```
python inference.py \
--checkpoint /results/checkpoint.pt \
--data /data/processed/electricity_bin/test.csv \
--tgt_scalers /data/processed/electricity_bin/tgt_scalers.bin \
--cat_encodings /data/processed/electricity_bin/cat_encodings.bin \
--batch_size 2048 \
--visualize \
--save_predictions \
--joint_visualization \
--results /results
```

In the default setting, it performs the evaluation of the model on a specified dataset and prints q-risk evaluated on this dataset. In order to save the predictions, use the `--save_predictions` option. Predictions will be stored in the directory specified by the `--results` option in the csv format. Option `--joint_visualization` allows us to plot graphs in TensorBoard format, allowing us to inspect the results and compare them to true values. Using `--visualize`, you can save plots for each example in a separate file.

### Triton deployment (TODO)

The [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server) provides a cloud inferencing solution optimized for NVIDIA GPUs. The server provides an inference service via an HTTP or GRPC endpoint, allowing remote clients to request inferencing for any model being managed by the server. More information on how to perform inference using NVIDIA Triton Inference Server can be found in [triton/README.md](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Forecasting/TFT/triton).
