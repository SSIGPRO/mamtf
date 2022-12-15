##########################################################################
# Copyright [2022] [Luciano Prono]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################

#!/bin/bash 

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
echo $tf_include_dir
SCRIPT_DIR=$(dirname $0)
cd $SCRIPT_DIR

echo "COMPILING GPU KERNEL"

nvcc -std=c++14 -O3 -c -o mam_op.cu.o mam_op.cu.cc \
  ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr

echo "COMPILING CPU KERNEL AND REGISTERING TF OPERATIONS"

g++ -std=c++14 -Ofast -shared \
  -o mam_op.so mam_op.cc \
  mam_op.cu.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -D GOOGLE_CUDA=1