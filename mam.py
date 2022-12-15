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

import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

import os

file_dir = os.path.dirname(os.path.realpath(__file__))
_mam_module = tf.load_op_library(os.path.join(file_dir, "mam_op.so"))


# Gradients

@ops.RegisterGradient("BetaMam")
def _beta_mam_grad(op, grad, grad2, grad3):
    x = op.inputs[0]
    w = op.inputs[1]
    beta = op.inputs[2]
    argmax = op.outputs[1]
    argmin = op.outputs[2]
    yg = grad
    
    xgrad, wgrad = _mam_module.beta_mam_grad(x, w, beta, yg, argmax, argmin)
    betagrad = 0
    
    return [xgrad, wgrad, betagrad] 


# Redefine mam functions
    
def mam(x, w, arg_out=False):
    y, argmax, argmin = _mam_module.mam(x, w)
    if arg_out:
        return y, argmax, argmin
    else:
        return y
    
def beta_mam(x, w, beta, arg_out=False):
    y, argmax, argmin = _mam_module.beta_mam(x, w, beta)
    if arg_out:
        return y, argmax, argmin
    else:
        return y