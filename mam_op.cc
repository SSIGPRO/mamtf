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

#include "mam_op.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#include <math.h>
#include <limits>

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

/*****************************************************************************/
/*****************************************************************************/
/* MAM operation *************************************************************/
/*****************************************************************************/
/*****************************************************************************/

REGISTER_OP("Mam")
    .Attr("T: numbertype")
    .Input("x: T")
    .Input("w: T")
    .Output("y: T")
    .Output("argmax: int32")
    .Output("argmin: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c)
                {
                    // Check if w columns match with x rows
                    ::tensorflow::shape_inference::DimensionHandle internal;
                    TF_RETURN_IF_ERROR(c->Merge(c->Dim(c->input(0), 1),
                                                c->Dim(c->input(1), 0),
                                                &internal));
                    
                    // Make output shape
                    c->set_output(0, c->MakeShape({c->Dim(c->input(0), 0),
                                                   c->Dim(c->input(1), 1)}));
                    c->set_output(1, c->MakeShape({c->Dim(c->input(0), 0),
                                                   c->Dim(c->input(1), 1)}));
                    c->set_output(2, c->MakeShape({c->Dim(c->input(0), 0),
                                                   c->Dim(c->input(1), 1)}));
                    return Status::OK();
                });

// CPU specialization of actual computation.
template <typename T>
struct MamOpFunctor<CPUDevice, T>
{
    void operator()(const CPUDevice& d,
                    int batchsize,
                    int sizein,
                    int sizeout,
                    const T* x,
                    const T* w,
                    T* y,
                    int* argmax,
                    int* argmin)
    {       
        printf("Launching CPU kernel\n");
        const T* xp = x;
        T* yp = y;
        int* argmaxp = argmax;
        int* argminp = argmin;
            
        for(int b = 0; b < batchsize; ++b) //foreach batch
        {        
            const T* wp = w;
            for(int j = 0; j < sizeout; ++j) //foreach column
            {
                const T* wpp = wp++;
                const T* xpp = xp;
                
                float max = -std::numeric_limits<T>::max();
                float min = std::numeric_limits<T>::max();
                int maxindex = 0;
                int minindex = 0;
                
                for(int i = 0; i < sizein; ++i) //foreach row
                {
                    T temp = *wpp * *(xpp++);
                    wpp += sizeout;
                    
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
                
                *(yp++) = max + min;
                *(argmaxp++) = maxindex;
                *(argminp++) = minindex;
            }
            xp += sizein;
        }
    }
};

// OpKernel definition.
template <typename Device, typename T>
class MamOp : public OpKernel
{
    public:
        explicit MamOp(OpKernelConstruction* context) : OpKernel(context){}
        
        void Compute(OpKernelContext* context) override
        {
            // Grab the input tensor
            const Tensor& x = context->input(0);
            const Tensor& w = context->input(1);
                        
            // Create output tensors
            Tensor* y = NULL;
            OP_REQUIRES_OK(context,
                           context->allocate_output(0,
                                                    TensorShape{x.dim_size(0),
                                                                w.dim_size(1)},
                                                    &y));
            Tensor* argmax = NULL;
            OP_REQUIRES_OK(
                context,
                context->allocate_output(1,
                                         TensorShape{x.dim_size(0), 
                                                     w.dim_size(1)}, 
                                         &argmax)
            );
            Tensor* argmin = NULL;
            OP_REQUIRES_OK(
                context,
                context->allocate_output(2,
                                         TensorShape{x.dim_size(0), 
                                                     w.dim_size(1)}, 
                                         &argmin)
            );
            
            // Extract the size of the tensors
            const int batchsize = x.dim_size(0); // batch size
            const int sizein = w.dim_size(0); // input size
            const int sizeout = w.dim_size(1); // output size
            
            // Do the computation
            OP_REQUIRES(
                context,
                x.dim_size(1) == w.dim_size(0),
                errors::InvalidArgument(
                    "Input size 1 is not equal to parameter matrix size 0."
                )
            );
            OP_REQUIRES(context,
                        x.NumElements() <= tensorflow::kint32max,
                        errors::InvalidArgument(
                            "Too many elements in tensor."
                        ));
            OP_REQUIRES(context,
                        w.NumElements() <= tensorflow::kint32max,
                        errors::InvalidArgument(
                            "Too many elements in tensor."
                        ));
            
            MamOpFunctor<Device, T>()(
                context->eigen_device<Device>(),
                batchsize,
                sizein,
                sizeout,
                x.flat<T>().data(),
                w.flat<T>().data(),
                y->flat<T>().data(),
                argmax->flat<int32>().data(),
                argmin->flat<int32>().data());
        }
};

// Register the CPU kernels.
#define REGISTER_CPU_MAM(T)                                                   \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("Mam").Device(DEVICE_CPU).TypeConstraint<T>("T"),                  \
      MamOp<CPUDevice, T>);
REGISTER_CPU_MAM(float);
REGISTER_CPU_MAM(int32);

// Register the GPU kernels.
// Declare explicit instantiations in sam_op.cu.cc.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU_MAM(T)                                                   \
  extern template struct MamOpFunctor<GPUDevice, T>;                          \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("Mam").Device(DEVICE_GPU).TypeConstraint<T>("T"),                  \
      MamOp<GPUDevice, T>);
REGISTER_GPU_MAM(float);
//REGISTER_GPU_MAM(int32);
#endif  // GOOGLE_CUDA


/*****************************************************************************/
/*****************************************************************************/
/* Beta MAM operation ********************************************************/
/*****************************************************************************/
/*****************************************************************************/


REGISTER_OP("BetaMam")
    .Input("x: float")
    .Input("w: float")
    .Input("beta: float")
    .Output("y: float")
    .Output("argmax: int32")
    .Output("argmin: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c)
                {
                    // Check if w columns match with x rows
                    ::tensorflow::shape_inference::DimensionHandle internal;
                    TF_RETURN_IF_ERROR(c->Merge(c->Dim(c->input(0), 1),
                                                c->Dim(c->input(1), 0),
                                                &internal));
                    
                    // Make output shape
                    c->set_output(0, c->MakeShape({c->Dim(c->input(0), 0),
                                                   c->Dim(c->input(1), 1)}));
                    c->set_output(1, c->MakeShape({c->Dim(c->input(0), 0),
                                                   c->Dim(c->input(1), 1)}));
                    c->set_output(2, c->MakeShape({c->Dim(c->input(0), 0),
                                                   c->Dim(c->input(1), 1)}));
                    return Status::OK();
                });

// CPU specialization of actual computation.
template <>
struct BetaMamOpFunctor<CPUDevice>
{
    void operator()(const CPUDevice& d,
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
        const float* xp = x;
        float* yp = y;
        int* argmaxp = argmax;
        int* argminp = argmin;
        
        float shrink = 1.0 - *beta;
            
        for(int b = 0; b < batchsize; ++b) //foreach batch
        {        
            const float* wp = w;
            for(int j = 0; j < sizeout; ++j) //foreach column
            {
                const float* wpp = wp++;
                const float* xpp = xp;
                
                float acc = 0;
                float max = -std::numeric_limits<float>::max();
                float min = std::numeric_limits<float>::max();
                int maxindex = 0;
                int minindex = 0;
                
                for(int i = 0; i < sizein; ++i) //foreach row
                {
                    float temp = *wpp * *(xpp++);
                    wpp += sizeout;
                    
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
                acc += *beta * (max + min);
                
                *(yp++) = acc;
                *(argmaxp++) = maxindex;
                *(argminp++) = minindex;
            }
            xp += sizein;
        }
    }
};

// OpKernel definition.
template <typename Device>
class BetaMamOp : public OpKernel
{
    public:
        explicit BetaMamOp(OpKernelConstruction* context) : OpKernel(context){}
        
        void Compute(OpKernelContext* context) override
        {
            // Grab the input tensor
            const Tensor& x = context->input(0);
            const Tensor& w = context->input(1);
            const Tensor& beta = context->input(2);
                        
            // Create output tensors
            Tensor* y = NULL;
            OP_REQUIRES_OK(context,
                           context->allocate_output(0,
                                                    TensorShape{x.dim_size(0),
                                                                w.dim_size(1)},
                                                    &y));
            Tensor* argmax = NULL;
            OP_REQUIRES_OK(
                context,
                context->allocate_output(1,
                                         TensorShape{x.dim_size(0), 
                                                     w.dim_size(1)}, 
                                         &argmax)
            );
            Tensor* argmin = NULL;
            OP_REQUIRES_OK(
                context,
                context->allocate_output(2,
                                         TensorShape{x.dim_size(0), 
                                                     w.dim_size(1)}, 
                                         &argmin)
            );
            
            // Extract the size of the tensors
            const int batchsize = x.dim_size(0); // batch size
            const int sizein = w.dim_size(0); // input size
            const int sizeout = w.dim_size(1); // output size
            
            // Do the computation
            OP_REQUIRES(
                context,
                x.dim_size(1) == w.dim_size(0),
                errors::InvalidArgument(
                    "Input size 1 is not equal to parameter matrix size 0."
                )
            );
            OP_REQUIRES(context,
                        x.NumElements() <= tensorflow::kint32max,
                        errors::InvalidArgument(
                            "Too many elements in tensor."
                        ));
            OP_REQUIRES(context,
                        w.NumElements() <= tensorflow::kint32max,
                        errors::InvalidArgument(
                            "Too many elements in tensor."
                        ));
            
            BetaMamOpFunctor<Device>()(
                context->eigen_device<Device>(),
                batchsize,
                sizein,
                sizeout,
                x.flat<float>().data(),
                w.flat<float>().data(),
                beta.flat<float>().data(),
                y->flat<float>().data(),
                argmax->flat<int32>().data(),
                argmin->flat<int32>().data());
        }
};

// Register the CPU kernel
REGISTER_KERNEL_BUILDER(Name("BetaMam").Device(DEVICE_CPU),
                        BetaMamOp<CPUDevice>);

// Register the GPU kernels.
REGISTER_KERNEL_BUILDER(Name("BetaMam").Device(DEVICE_GPU),
                        BetaMamOp<GPUDevice>);



/*****************************************************************************/
/*****************************************************************************/
/* Beta MAM gradient operation ***********************************************/
/*****************************************************************************/
/*****************************************************************************/


REGISTER_OP("BetaMamGrad")
    .Input("x: float")
    .Input("w: float")
    .Input("beta: float")
    .Input("yg: float")
    .Input("argmax: int32")
    .Input("argmin: int32")
    .Output("xg: float")
    .Output("wg: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c)
                {
                    // Check if w columns match with x rows
                    ::tensorflow::shape_inference::DimensionHandle internal;
                    TF_RETURN_IF_ERROR(c->Merge(c->Dim(c->input(0), 1),
                                                c->Dim(c->input(1), 0),
                                                &internal));
                    TF_RETURN_IF_ERROR(c->Merge(c->Dim(c->input(1), 1),
                                                c->Dim(c->input(3), 1),
                                                &internal));
                    TF_RETURN_IF_ERROR(c->Merge(c->Dim(c->input(0), 0),
                                                c->Dim(c->input(3), 0),
                                                &internal));
                    
                    // Make output shape
                    c->set_output(0, c->MakeShape({c->Dim(c->input(0), 0),
                                                   c->Dim(c->input(0), 1)}));
                    c->set_output(1, c->MakeShape({c->Dim(c->input(1), 0),
                                                   c->Dim(c->input(1), 1)}));
                    return Status::OK();
                });


// CPU specialization of actual computation.
template <>
struct BetaMamGradOpFunctor<CPUDevice>
{
    void operator()(const CPUDevice& d,
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
        float shrink = 1.0 - *beta;
            
        // evaluate xg
        const float* ygp = yg;
        const int* argmaxp = argmax;
        const int* argminp = argmin;
        float* xgp = xg;
        
        for(int b = 0; b < batchsize; ++b) //foreach batch element
        {
            const float* wp = w;
            float* xgpp = xgp;

            for(int i = 0; i < sizein; ++i) //foreach row
            {
                const float* ygpp = ygp;
                const int* argmaxpp = argmaxp;
                const int* argminpp = argminp;
                float acc = 0;
                float accmaxmin = 0;
                for(int j = 0; j < sizeout; ++j) //foreach column
                {
                    float temp = *(wp++) * *(ygpp++);
                    if(*argmaxpp == i || *argminpp == i)
                    {
                        accmaxmin += temp;
                    }
                    argmaxpp++;
                    argminpp++;
                    acc += temp;
                }
                acc = acc * shrink;
                acc += *beta * accmaxmin;
                *xgpp = acc;
                
                xgpp++;                
            }
            ygp += sizeout;
            argmaxp += sizeout;
            argminp += sizeout;
            xgp += sizein;
        }
                    
        // evaluate wg
        ygp = yg;
        argmaxp = argmax;
        argminp = argmin;
        float *wgp = wg;
        
        for(int j = 0; j < sizeout; ++j) //foreach column
        {
            const float* xp = x;
            float *wgpp = wgp;
            
            for(int i = 0; i < sizein; ++i) //foreach row
            {
                const float *xpp = xp;
                const float* ygpp = ygp;
                const int* argmaxpp = argmaxp;
                const int* argminpp = argminp;
                
                float acc = 0;
                float accmaxmin = 0;
                
                for(int b = 0; b < batchsize; ++b) //foreach batch element
                {
                    float temp = *xpp * *ygpp;
                    xpp += sizein;
                    ygpp += sizeout;
                    acc += temp;
                    if(*argmaxpp == i || *argminpp == i)
                    {
                        accmaxmin += temp;
                    }
                    argmaxpp += sizeout;
                    argminpp += sizeout;
                }
                acc = acc * shrink;
                acc += *beta * accmaxmin;
                *wgpp = acc;
                
                xp++;
                wgpp += sizeout;
            }
           ygp++;
           argmaxp++;
           argminp++;
           wgp++;
        }
    }
};

// OpKernel definition.
template <typename Device>
class BetaMamGradOp : public OpKernel
{
    public:
        explicit BetaMamGradOp(OpKernelConstruction* context)
            : OpKernel(context){}
        
        void Compute(OpKernelContext* context) override
        {
            // Grab the input tensor
            const Tensor& x = context->input(0);
            const Tensor& w = context->input(1);
            const Tensor& beta = context->input(2);
            const Tensor& yg = context->input(3);
            const Tensor& argmax = context->input(4);
            const Tensor& argmin = context->input(5);
                        
            // Create output tensors
            Tensor* xg = NULL;
            OP_REQUIRES_OK(context,
                           context->allocate_output(0,
                                                    TensorShape{x.dim_size(0),
                                                                x.dim_size(1)},
                                                    &xg));
            Tensor* wg = NULL;
            OP_REQUIRES_OK(
                context,
                context->allocate_output(1,
                                         TensorShape{w.dim_size(0), 
                                                     w.dim_size(1)}, 
                                         &wg)
            );
            
            // Extract the size of the tensors
            const int batchsize = x.dim_size(0); // batch size
            const int sizein = w.dim_size(0); // input size
            const int sizeout = w.dim_size(1); // output size
            
            // Do the computation
            OP_REQUIRES(
                context,
                x.dim_size(1) == w.dim_size(0),
                errors::InvalidArgument(
                    "Input size 1 is not equal to parameter matrix size 0."
                )
            );
            OP_REQUIRES(
                context,
                w.dim_size(1) == yg.dim_size(1),
                errors::InvalidArgument(
                    "Parameter matrix size 1 is not equal with gradient size 1."
                )
            );
            OP_REQUIRES(
                context,
                x.dim_size(0) == yg.dim_size(0),
                errors::InvalidArgument(
                    "Input size 0 is not equal to gradient size 0."
                )
            );
            OP_REQUIRES(context,
                        x.NumElements() <= tensorflow::kint32max,
                        errors::InvalidArgument(
                            "Too many elements in tensor."
                        ));
            OP_REQUIRES(context,
                        w.NumElements() <= tensorflow::kint32max,
                        errors::InvalidArgument(
                            "Too many elements in tensor."
                        ));
            
            BetaMamGradOpFunctor<Device>()(
                context->eigen_device<Device>(),
                batchsize,
                sizein,
                sizeout,
                x.flat<float>().data(),
                w.flat<float>().data(),
                beta.flat<float>().data(),
                yg.flat<float>().data(),
                argmax.flat<int>().data(),
                argmin.flat<int>().data(),
                xg->flat<float>().data(),
                wg->flat<float>().data());
        }
};

// Register the CPU kernel
REGISTER_KERNEL_BUILDER(Name("BetaMamGrad").Device(DEVICE_CPU),
                        BetaMamGradOp<CPUDevice>);

// Register the GPU kernels.
REGISTER_KERNEL_BUILDER(Name("BetaMamGrad").Device(DEVICE_GPU),
                        BetaMamGradOp<GPUDevice>);