#!/bin/bash
# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

DS='pricing'
DS_PATH=/data/raw
SUFFIX=$(date -d "today" +"%Y%m%d%H%M")
if [ -f ${DS_PATH}/standarized.csv ]
then
    mv ${DS_PATH}/standarized.csv ${DS_PATH}/standarized-${SUFFIX}.csv
fi

if [ -d /data/results ]
then
    mv /data/results /data/results-${SUFFIX}
fi

if [ -d /data/processed ]
then
    mv /data/processed /data/processed-${SUFFIX}
fi

mkdir -p ${DS_PATH}
mkdir -p /data/processed

python get_data.py

python -c "from data_utils import standarize_${DS} as standarize; standarize(\"${DS_PATH}\")"
python -c "from data_utils import preprocess; \
            from configuration import ${DS^}Config as Config; \
            preprocess(\"${DS_PATH}/standarized.csv\", \"/data/processed/${DS}_bin\", Config())"
