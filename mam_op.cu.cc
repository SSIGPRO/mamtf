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

#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "sam_op.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

#include <limits>

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

/*****************************************************************************/
/*****************************************************************************/
/* Mam operation *************************************************************/
/*****************************************************************************/
/*****************************************************************************/

// Define the CUDA kernel.
template <typename T>
__global__ void MamOpCudaKernel(const int batchsize,
                                const int sizein,
                                const int sizeout,
                                const T* x,
                                const T* w,
                                T* y,
                                int* argmax,
                                int* argmin)
{
    int batch = blockIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    
    if(batch < batchsize and col < sizeout)
    {
        const T* xp = x + batch*sizein;
        const T* wp = w + col;
        T* yp = y + batch*sizeout + col;
        int* argmaxp = argmax + batch*sizeout + col;
        int* argminp = argmin + batch*sizeout + col;
                   
        T max = -std::numeric_limits<T>::max();
        T min = std::numeric_limits<T>::max();
        int maxindex = 0;
        int minindex = 0;
                
        for(int i = 0; i < sizein; ++i) //foreach row
        {
            T temp = *wp * *(xp++);
            wp += sizeout;
                    
            if(temp > max) //search for max
            {
                maxindex = i;
                max = temp;
            }
            if(temp < min) //search for min
            {
                minindex = i;
                min = temp;
            }
        }
        *yp = max + min;
        *argmaxp = maxindex;
        *argminp = minindex;
    }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
void MamOpFunctor<GPUDevice, T>::operator()(
    const GPUDevice& d,
    int batchsize,
    int sizein,
    int sizeout,
    const T* x,
    const T* w,
    T* y,
    int* argmax,
    int* argmin) 
{
    // Launch the cuda kernel.
    //
    // See core/util/gpu_kernel_helper.h for example of computing
    // block count and thread_per_block count.    
    int block_x = (sizeout - 1)/1024 + 1;
    int block_y = batchsize;
    int threads = (block_x > 1) ? 1024 : sizeout;
    MamOpCudaKernel<T>
        <<<dim3(block_x, block_y, 1), threads, 0, d.stream()>>>(
        batchsize,
        sizein,
        sizeout,
        x,
        w,
        y,
        argmax,
        argmin
    );
}

// Explicitly instantiate functors for the types of OpKernels registered.
template struct MamOpFunctor<GPUDevice, float>;
template struct MamOpFunctor<GPUDevice, int32>;

/*****************************************************************************/
/*****************************************************************************/
/* BetaMam operation *********************************************************/
/*****************************************************************************/
/*****************************************************************************/

// Define the CUDA kernel.
__global__ void BetaMamOpCudaKernel(const int batchsize,
                                    const int sizein,
                                    const int sizeout,
                                    const float* x,
                                    const float* w,
                                    const float* beta,
                                    float* y,
                                    int* argmax,
                                    int* argmin)
{
    int batch = blockIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    
    float shrink = 1.0 - *beta;
    
    if(batch < batchsize and col < sizeout)
    {
        const float* xp = x + batch*sizein;
        const float* wp = w + col;
        float* yp = y + batch*sizeout + col;
        int* argmaxp = argmax + batch*sizeout + col;
        int* argminp = argmin + batch*sizeout + col;
                   
        float acc = 0;
        float max = -std::numeric_limits<float>::max();
        float min = std::numeric_limits<float>::max();
        int maxindex = 0;
        int minindex = 0;
                
        for(int i = 0; i < sizein; ++i) //foreach row
        {
            float temp = *wp * *(xp++);
            wp += sizeout;    
            if(temp > max) //search for max
            {
                maxindex = i;
                max = temp;
            }
            if(temp < min) //search for min
            {
                minindex = i;
                min = temp;
            }
            acc += temp;
        }
        acc = acc * shrink;
        acc += *beta * (max+min);
        
        *yp = acc;
        *argmaxp = maxindex;
        *argminp = minindex;
    }
}

// Define the GPU implementation that launches the CUDA kernel.
void BetaMamOpFunctor<GPUDevice>::operator()(
    const GPUDevice& d,
    int batchsize,
    int sizein,
    int sizeout,
    const float* x,
    const float* w,
    const float* beta,
    float* y,
    int* argmax,
    int* argmin) 
{
    // Launch the cuda kernel.
    //
    // See core/util/gpu_kernel_helper.h for example of computing
    // block count and thread_per_block count.    
    int block_x = (sizeout - 1)/1024 + 1;
    int block_y = batchsize;
    int threads = (block_x > 1) ? 1024 : sizeout;
    
    BetaMamOpCudaKernel
        <<<dim3(block_x, block_y, 1), threads, 0, d.stream()>>>(
        batchsize,
        sizein,
        sizeout,
        x,
        w,
        beta,
        y,
        argmax,
        argmin
    );
}

// Explicitly instantiate functors for the types of OpKernels registered.
template struct BetaMamOpFunctor<GPUDevice>;


/*****************************************************************************/
/*****************************************************************************/
/* BetaMam gradient operation ************************************************/
/*****************************************************************************/
/*****************************************************************************/

// Define the CUDA kernel.
__global__ void BetaMamXGradOpCudaKernel(const int batchsize,
                                         const int sizein,
                                         const int sizeout,
                                         const float* w,
                                         const float* beta,
                                         const float* yg,
                                         const int* argmax,
                                         const int* argmin,
                                         float* xg)
{    
    int batch = blockIdx.y;
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    
    // evaluate xg
    if(batch < batchsize && row < sizein)
    {
        float shrink = 1.0 - *beta;
        
        const float* wp = w + sizeout*row;
        const float* ygp = yg + sizeout*batch;
        const int* argmaxp = argmax + sizeout*batch;
        const int* argminp = argmin + sizeout*batch;
        float* xgp = xg + sizein*batch + row;
        
        float acc = 0;
        float accmaxmin = 0;
        for(int j = 0; j < sizeout; ++j) //foreach column
        {
            float temp = *(wp++) * *(ygp++);
            acc += temp;
            
            if(*argmaxp == row || *argminp == row)
            {
                accmaxmin += temp;
            }
            
            argmaxp++;
            argminp++;
        }
        acc = acc * shrink;
        acc += *beta * accmaxmin;
        *xgp = acc;               
    }   
}

__global__ void BetaMamWGradOpCudaKernel(const int batchsize,
                                         const int sizein,
                                         const int sizeout,
                                         const float* x,
                                         const float* beta,
                                         const float* yg,
                                         const int* argmax,
                                         const int* argmin,
                                         float* wg)
{   
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    
    if(col < sizeout && row < sizein)
    {
        float shrink = 1.0 - *beta;
        
        const float* xp = x + row;
        const float* ygp = yg + col;
        const int* argmaxp = argmax + col;
        const int* argminp = argmin + col;
        float *wgp = wg + sizeout*row + col;
            
        float acc = 0;
        float accmaxmin = 0;
        
        for(int b = 0; b < batchsize; ++b) //foreach batch element
        {
            float temp = *xp * *ygp;
            xp += sizein;
            ygp += sizeout;
            acc += temp;
            
            if(*argmaxp == row || *argminp == row)
            {
                accmaxmin += temp;
            }
            
            argmaxp += sizeout;
            argminp += sizeout;
        }
        acc = acc * shrink;
        acc += *beta * accmaxmin;
        *wgp = acc;
    }
}

// Define the GPU implementation that launches the CUDA kernel.
void BetaMamGradOpFunctor<GPUDevice>::operator()(
    const GPUDevice& d,
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
    float* wg) 
{
    // Launch the cuda kernel.
    //
    // See core/util/gpu_kernel_helper.h for example of computing
    // block count and thread_per_block count.    
    
    int block_x, block_y;
    int thread_x, thread_y;
    
    block_x = (sizein - 1)/1024 + 1;
    block_y = batchsize;
    thread_x = (block_x > 1) ? 1024 : sizein;
    
    BetaMamXGradOpCudaKernel
        <<<dim3(block_x, block_y, 1), thread_x, 0, d.stream()>>>(
        batchsize,
        sizein,
        sizeout,
        w,
        beta,
        yg,
        argmax,
        argmin,
        xg
    );
    
    block_x = (sizeout - 1)/32 + 1;
    block_y = (sizein - 1)/32 + 1;
    thread_x = (block_x > 1) ? 32 : sizeout;
    thread_y = (block_y > 1) ? 32 : sizein;
    
    BetaMamWGradOpCudaKernel
        <<<dim3(block_x, block_y, 1),
           dim3(thread_x, thread_y, 1),
           0,
           d.stream()>>>(
        batchsize,
        sizein,
        sizeout,
        x,
        beta,
        yg,
        argmax,
        argmin,
        wg
    );
}

// Explicitly instantiate functors for the types of OpKernels registered.
template struct BetaMamGradOpFunctor<GPUDevice>;

#endif  // GOOGLE_CUDA