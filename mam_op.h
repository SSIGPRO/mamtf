/******************************************************************************
* Copyright [2022] [Luciano Prono]
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
******************************************************************************/

#ifndef MAM_OP_H_
#define MAM_OP_H_

#include <unsupported/Eigen/CXX11/Tensor>

template <typename Device, typename T>
struct MamOpFunctor
{
    void operator()(const Device& d,
                    int batchsize,
                    int sizein,
                    int sizeout,
                    const T* x,
                    const T* w,
                    T* y,
                    int* argmax,
                    int* argmin);
};

template <typename Device>
struct BetaMamOpFunctor
{
    void operator()(const Device& d,
                    int batchsize,
                    int sizein,
                    int sizeout,
                    const float* x,
                    const float* w,
                    const float* beta,
                    float* y,
                    int* argmax,
                    int* argmin);
};

template <typename Device>
struct BetaMamGradOpFunctor
{
    void operator()(const Device& d,
                    int batchsize,
                    int sizein,
                    int sizeout,
                    const float* x,
                    const float* w,
                    const float* beta,
                    const float* yg,
                    const int* argmax,
                    const int* argmin,
                    float* xg,
                    float* wg);
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename T>
struct MamOpFunctor<Eigen::GpuDevice, T>
{
    void operator()(const Eigen::GpuDevice& d,
                    int batchsize,
                    int sizein,
                    int sizeout,
                    const T* x,
                    const T* w,
                    T* y,
                    int *argmax,
                    int *argmin);
};

template <>
struct BetaMamOpFunctor<Eigen::GpuDevice>
{
    void operator()(const Eigen::GpuDevice& d,
                    int batchsize,
                    int sizein,
                    int sizeout,
                    const float* x,
                    const float* w,
                    const float* beta,
                    float* y,
                    int *argmax,
                    int *argmin);
};

template <>
struct BetaMamGradOpFunctor<Eigen::GpuDevice>
{
    void operator()(const Eigen::GpuDevice& d,
                    int batchsize,
                    int sizein,
                    int sizeout,
                    const float* x,
                    const float* w,
                    const float* beta,
                    const float* yg,
                    const int* argmax,
                    const int* argmin,
                    float* xg,
                    float* wg);
};

#endif

#endif //MAM_OP_H_