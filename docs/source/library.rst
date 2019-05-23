
.. toctree::
   :maxdepth: 4 
   :caption: Contents:

======
rocFFT
======

Introduction
------------

The rocFFT library is an implementation of discrete Fast Fourier Transforms (FFT) written in HiP for GPU devices.
The code is open and hosted here: https://github.com/ROCmSoftwarePlatform/rocFFT

The rocFFT library:

* Provides a fast and accurate platform for calculating discrete FFTs.
* Supports single and double precision floating point formats.
* Supports 1D, 2D, and 3D transforms.
* Supports computation of transforms in batches.
* Supports real and complex FFTs.
* Supports lengths that are any combination of powers of 2, 3, 5.

FFT Computation
---------------

The FFT is an implementation of the Discrete Fourier Transform (DFT) that makes use of symmetries in the DFT definition to
reduce the mathematical complexity from :math:`O(N^2)` to :math:`O(N \log N)` when the sequence length, *N*, is
the product of small prime factors.

What is computed by the library? Here are the formulas:

For a 1D complex DFT:

:math:`{\tilde{x}}_j = {{1}\over{scale}}\sum_{k=0}^{n-1}x_k\exp\left({\pm i}{{2\pi jk}\over{n}}\right)\hbox{ for } j=0,1,\ldots,n-1`

where, :math:`x_k` are the complex data to be transformed, :math:`\tilde{x}_j` are the transformed data, and the sign :math:`\pm`
determines the direction of the transform: :math:`-` for forward and :math:`+` for backward. Note that you must provide the scaling
factor.  By default, the scale is set to 1 for the transforms.

For a 2D complex DFT:

:math:`{\tilde{x}}_{jk} = {{1}\over{scale}}\sum_{q=0}^{m-1}\sum_{r=0}^{n-1}x_{rq}\exp\left({\pm i} {{2\pi jr}\over{n}}\right)\exp\left({\pm i}{{2\pi kq}\over{m}}\right)`

for :math:`j=0,1,\ldots,n-1\hbox{ and } k=0,1,\ldots,m-1`, where, :math:`x_{rq}` are the complex data to be transformed,
:math:`\tilde{x}_{jk}` are the transformed data, and the sign :math:`\pm` determines the direction of the transform.  By default, the
scale is set to 1 for the transforms.

For a 3D complex DFT:

:math:`\tilde{x}_{jkl} = {{1}\over{scale}}\sum_{s=0}^{p-1}\sum_{q=0}^{m-1}\sum_{r=0}^{n-1}x_{rqs}\exp\left({\pm i} {{2\pi jr}\over{n}}\right)\exp\left({\pm i}{{2\pi kq}\over{m}}\right)\exp\left({\pm i}{{2\pi ls}\over{p}}\right)`

for :math:`j=0,1,\ldots,n-1\hbox{ and } k=0,1,\ldots,m-1\hbox{ and } l=0,1,\ldots,p-1`, where :math:`x_{rqs}` are the complex data to
be transformed, :math:`\tilde{x}_{jkl}` are the transformed data, and the sign :math:`\pm` determines the direction of the transform.
By default, the scale is set to 1 for the transforms.

Library Setup and Cleanup
-------------------------

At the beginning of the program, before any of the library api is called, the function :cpp:func:`rocfft_setup` has to be called. Similarly,
the function :cpp:func:`rocfft_cleanup` has to be called at the end of the program. These apis ensure resources are properly allocated and freed. 

Workflow
--------

In order to compute an FFT transform with rocFFT, a plan has to be created first. A plan is a handle to an internal data structure that
holds the details about the transform that the user wishes to compute. After the plan is created, it can be executed (a separate api call) 
with the specified data buffers. The execution step can be repeated any number of times with the same plan on different input/output buffers
as needed. And when the plan is no longer needed, it gets destroyed.

To do a transform,

#. Initialize the library by calling :cpp:func:`rocfft_setup()`
#. Create a plan, for each distinct type of FFT needed:

   * To create a plan, do either of the following

     * If the plan specification is simple, call :cpp:func:`rocfft_plan_create` and specify the value of the fundamental parameters
     * If the plan has more details, first a plan description is created with :cpp:func:`rocfft_plan_description_create`, and additional apis such
       as :cpp:func:`rocfft_plan_description_set_data_layout` are called to specify plan details. And then, :cpp:func:`rocfft_plan_create` is called
       with the description handle passed to it along with other details.

#. Execute the plan

   * The execution api :cpp:func:`rocfft_execute` is used to do the actual computation on the data buffers specified
   * For specifying scratch/work buffers, and other parameters or to get back information regarding execution, an execution info object needs to be created
     using :cpp:func:`rocfft_execution_info_create` and passed to the execution api
   * Execution api can be called repeatedly as needed for different data, with the same plan

#. Destroy the plan
#. Terminate the library by calling :cpp:func:`rocfft_cleanup()`


Example
-------

.. code-block:: c

   #include <iostream>
   #include <vector>
   #include "hip/hip_runtime_api.h"
   #include "hip/hip_vector_types.h"
   #include "rocfft.h"
   
   int main()
   {
           // rocFFT gpu compute
           // ========================================
  
           rocfft_setup();

           size_t N = 16;
           size_t Nbytes = N * sizeof(float2);
   
           // Create HIP device buffer
           float2 *x;
           hipMalloc(&x, Nbytes);
   
           // Initialize data
           std::vector<float2> cx(N);
           for (size_t i = 0; i < N; i++)
           {
                   cx[i].x = 1;
                   cx[i].y = -1;
           }
   
           //  Copy data to device
           hipMemcpy(x, cx.data(), Nbytes, hipMemcpyHostToDevice);
   
           // Create rocFFT plan
           rocfft_plan plan = NULL;
           size_t length = N;
           rocfft_plan_create(&plan, rocfft_placement_inplace,
                rocfft_transform_type_complex_forward, rocfft_precision_single,
                1, &length, 1, NULL);
   
           // Execute plan
           rocfft_execute(plan, (void**) &x, NULL, NULL);
   
           // Wait for execution to finish
           hipDeviceSynchronize();
   
           // Destroy plan
           rocfft_plan_destroy(plan);
   
           // Copy result back to host
           std::vector<float2> y(N);
           hipMemcpy(y.data(), x, Nbytes, hipMemcpyDeviceToHost);
   
           // Print results
           for (size_t i = 0; i < N; i++)
           {
                   std::cout << y[i].x << ", " << y[i].y << std::endl;
           }
   
           // Free device buffer
           hipFree(x);
   
           rocfft_cleanup();

           return 0;
   }

Plans
-----

A plan is the collection of (almost) all the parameters needed to specify an FFT computation. A rocFFT plan includes the
following information:

* Type of transform (complex or real)
* Dimension of the transform (1D, 2D or 3D)
* Length or extent of data in each dimension
* Number of datasets that are transformed (batch size)
* Floating-point precision of the data
* Scaling factor for the transformed data
* In-place or not in-place transform
* Format (array type) of the input/output buffer
* Layout of data in the input/output buffer 

The rocFFT plan does not include the following parameters:

* The handles to the input and output data buffers.
* The handle to a temporary scratch buffer (if needed).
* Other information to control execution on the device.

These parameters are specified when the plan is executed.

Data
----

The input/output buffers that hold the data for the transform must be allocated, initialized and specified to the library by the
user. For larger transforms, scratch/work buffers may be needed. Because the library tries to minimize its own allocation of
memory regions on the device, it expects the user to manage work buffers. The size of the buffer needed can be queried using
:cpp:func:`rocfft_plan_get_work_buffer_size` and after their allocation can be passed to the library by
:cpp:func:`rocfft_execution_info_set_work_buffer`. The samples in the source repository show how to use these.

Transform and Array types 
-------------------------

There are two main types of FFT transforms in the library:

* Complex FFT - Transformation of complex data(could be forward or backward); the library supports the following two
  array types to store complex numbers:

  #. Planar format - where the real and imaginary components are kept in 2 separate arrays:

     * Buffer1: ``RRRRR...`` 
     * Buffer2: ``IIIII...``
  #. Interleaved format - where the real and imaginary components are stored as contiguous pairs in the same array: 

     * Buffer: ``RIRIRIRIRIRI...``
  
* Real FFT - Transformation of real data. For transforms involving real data, there are two possibilities:

  * Real data being subject to forward FFT transform that results in complex data (Hermitian).
  * Complex data (Hermitian) being subject to backward FFT transform that results in real data.

The library provides enums to specify transform and array types.

Batches
-------

The efficiency of the library is improved by utilizing transforms in batches. Sending as much data as possible in a single
transform call leverages the parallel compute capabilities of devices (GPU devices in particular), and minimizes the penalty
of control transfer. It is best to think of a device as a high-throughput, high-latency device. Using a networking analogy as
an example, this approach is similar to having a massively high-bandwidth pipe with very high ping response times. If the client
is ready to send data to the device for compute, it should be sent in as few API calls as possible and this can be done by batching.
rocFFT plans have a parameter `number_of_transforms` (this value is also referred to as batch size in various places in the document)
in :cpp:func:`rocfft_plan_create` to describe the number of transforms being requested. All 1D, 2D, and 3D transforms can be batched.


Strides and Distances
---------------------

Strides and distances enable users to specify custom layout of data using :cpp:func:`rocfft_plan_description_set_data_layout`

For 1D data, if strides[0] = strideX = 1, successive elements in the first dimension (dimension index 0) are stored
contiguously in memory. If strideX is a value greater than 1, gaps in memory exist between each element of the vector.
For multi-dimensional cases; if strides[1] = strideY = LenX for 2D data and strides[2] = strideZ = LenX*LenY for 3D data,
no gaps exist in memory between each element, and all vectors are stored tightly packed in memory. Here, LenX, LenY, and LenZ denote the
transform lengths lengths[0], lengths[1], and lengths[2], respectively, which are used to set up the plan.

Distance is stride that exists between corresponding elements of successive FFT data instances (primitives) in a batch. Distance is measured in units of the memory type;
complex data measures in complex units, and real data measures in real units. For tightly packed data, the distance between FFT primitives is the size of the FFT primitive,
such that dist=LenX for 1D data, dist=LenX*LenY for 2D data, and dist=LenX*LenY*LenZ for 3D data. It is possible to set the distance of a plan to be less than the size
of the FFT vector; typically 1 when doing column (strided) access on packed data. When computing a batch of 1D FFT vectors, if distance == 1, and strideX == length(vector),
it means data for each logical FFT is read along columns (in this case along the batch). You must verify that the distance and strides are valid, such that each logical
FFT instance is not overlapping with any other; if not valid, undefined results may occur. A simple example would be to perform a 1D length 4096 on each row of an array of
1024 rows x 4096 columns of values stored in a column-major array, such as a FORTRAN program might provide. (This would be equivalent to a C or C++ program that has an
array of 4096 rows x 1024 columns stored in a row-major manner, on which you want to perform a 1-D length 4096 transform on each column.) In this case, specify the
strides as [1024] and distance as 1.

Result placement
----------------

The API supports both in-place and not in-place transforms. With in-place transforms, only input buffers are provided to the
execution API, and the resulting data is written in the same buffer, overwriting the input data. With not in-place transforms, distinct
output buffers are provided, and the results are written into the output buffer. In this case, input buffer is only read and so data in it is
preserved after the execution.

Real FFTs
---------

.. toctree::
   :maxdepth: 2

   real

