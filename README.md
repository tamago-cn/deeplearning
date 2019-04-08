# deeplearning
deeplearning project

## 使用豆瓣源
```shell
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << EOF
[global]
timeout = 60
index-url = https://pypi.doubanio.com/simple
EOF
```

## 安装依赖

pip install tensorflow

```python
# python2需添加以下3句以获得python3的特性
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

# 调整tensorflow日志等级, '1' 所有，'2' warning，'3' error
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]= '2'

import tensorflow as tf

```
