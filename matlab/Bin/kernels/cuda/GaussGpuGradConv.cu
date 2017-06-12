// Author : B. Charlier (2017)

#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <mex.h>


#define UseCudaOnDoubles USE_DOUBLE_PRECISION

///////////////////////////////////////
////////// GRAD CONV //////////////////
///////////////////////////////////////


template < typename TYPE, int DIMPOINT, int DIMVECT >
__global__ void GaussGpuGradConvOnDevice(TYPE ooSigma2,
        TYPE *alpha, TYPE *x, TYPE *beta, TYPE *gamma,
        int nx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // the following line does not work with nvcc 3.0 (it is a bug; it works with anterior and posterior versions)
    // extern __shared__ TYPE SharedData[];  // shared data will contain x and alpha data for the block
    // here is the bug fix (see http://forums.nvidia.com/index.php?showtopic=166905)
    extern __shared__ char SharedData_char[];
    TYPE* const SharedData = reinterpret_cast<TYPE*>(SharedData_char);
    // end of bug fix

    TYPE xi[DIMPOINT], alphai[DIMVECT], betai[DIMVECT], ximxj[DIMPOINT], gammai[DIMPOINT];
    if(i<nx)  // we will compute gammai only if i is in the range
    {
        // load xi, alphai, betai from device global memory
        for(int k=0; k<DIMPOINT; k++)
            xi[k] = x[i*DIMPOINT+k];
        for(int k=0; k<DIMVECT; k++)
            alphai[k] = alpha[i*DIMVECT+k];
        for(int k=0; k<DIMVECT; k++)
            betai[k] = beta[i*DIMVECT+k];
        for(int k=0; k<DIMPOINT; k++)
            gammai[k] = 0.0f;
    }

    for(int jstart = 0, tile = 0; jstart < nx; jstart += blockDim.x, tile++)
    {
        int j = tile * blockDim.x + threadIdx.x;
        if(j<nx) // we load xj, alphaj and betaj from device global memory only if j<nx
        {
            int inc = DIMPOINT + 2 * DIMVECT;
            for(int k=0; k<DIMPOINT; k++)
                SharedData[threadIdx.x*inc+k] = x[j*DIMPOINT+k];
            for(int k=0; k<DIMVECT; k++)
                SharedData[threadIdx.x*inc+DIMPOINT+k] = alpha[j*DIMVECT+k];
            for(int k=0; k<DIMVECT; k++)
                SharedData[threadIdx.x*inc+DIMPOINT+DIMVECT+k] = beta[j*DIMVECT+k];
        }
        __syncthreads();
        if(i<nx) // we compute gammai only if i is in the range
        {
            TYPE *xj, *alphaj, *betaj;
            xj = SharedData;
            alphaj = SharedData + DIMPOINT;
            betaj = SharedData + DIMPOINT + DIMVECT;
            int inc = DIMPOINT + 2 * DIMVECT;
            for(int jrel = 0; jrel < blockDim.x && jrel<nx-jstart; jrel++, xj+=inc, alphaj+=inc, betaj+=inc)
            {
                TYPE r2 = 0.0f, sga = 0.0f;
                for(int k=0; k<DIMPOINT; k++)
                {
                    ximxj[k] =  xi[k]-xj[k];
                    r2 += ximxj[k]*ximxj[k];
                }
                for(int k=0; k<DIMVECT; k++)
                    sga += betaj[k]*alphai[k] + betai[k]*alphaj[k];
                TYPE s = exp(-r2*ooSigma2) * (-ooSigma2*2.0f*sga);
                for(int k=0; k<DIMPOINT; k++)
                    gammai[k] += s * ximxj[k];
            }
        }
        __syncthreads();
    }

    // Save the result in global memory.
    if(i<nx)
        for(int k=0; k<DIMPOINT; k++)
            gamma[i*DIMPOINT+k] = gammai[k];
}

////////////////////////////////////////////////////////////////////////////

extern "C" int GaussGpuGradConv_float(float ooSigma2,
        float* alpha_h, float* x_h, float* beta_h, float* gamma_h,
        int dimPoint, int dimVect, int nx)
{

    // Data on the device.
    float* x_d;
    float* alpha_d;
    float* gamma_d;
    float* beta_d;

    // Allocate arrays on device.
    cudaMalloc((void**)&x_d, sizeof(float)*(nx*dimPoint));
    cudaMalloc((void**)&alpha_d, sizeof(float)*(nx*dimVect));
    cudaMalloc((void**)&beta_d, sizeof(float)*(nx*dimVect));
    cudaMalloc((void**)&gamma_d, sizeof(float)*(nx*dimPoint));

    // Send data from host to device.
    cudaMemcpy(x_d, x_h, sizeof(float)*(nx*dimPoint), cudaMemcpyHostToDevice);
    cudaMemcpy(alpha_d, alpha_h, sizeof(float)*(nx*dimVect), cudaMemcpyHostToDevice);
    cudaMemcpy(beta_d, beta_h, sizeof(float)*(nx*dimVect), cudaMemcpyHostToDevice);

    // compute on device.
    dim3 blockSize;
    blockSize.x = CUDA_BLOCK_SIZE; // number of threads in each block
    dim3 gridSize;
    gridSize.x =  nx / blockSize.x + (nx%blockSize.x==0 ? 0 : 1);

    if(dimPoint==1 && dimVect==1)
        GaussGpuGradConvOnDevice<float,1,1><<<gridSize,blockSize,blockSize.x*(2*dimPoint+dimVect)*sizeof(float)>>>
        (ooSigma2, alpha_d, x_d, beta_d, gamma_d, nx);
    else if(dimPoint==2 && dimVect==1)
        GaussGpuGradConvOnDevice<float,2,1><<<gridSize,blockSize,blockSize.x*(2*dimPoint+dimVect)*sizeof(float)>>>
        (ooSigma2, alpha_d, x_d, beta_d, gamma_d, nx);
    else if(dimPoint==2 && dimVect==2)
        GaussGpuGradConvOnDevice<float,2,2><<<gridSize,blockSize,blockSize.x*(2*dimPoint+dimVect)*sizeof(float)>>>
        (ooSigma2, alpha_d, x_d, beta_d, gamma_d, nx);
    else if(dimPoint==3 && dimVect==1)
        GaussGpuGradConvOnDevice<float,3,1><<<gridSize,blockSize,blockSize.x*(2*dimPoint+dimVect)*sizeof(float)>>>
        (ooSigma2, alpha_d, x_d, beta_d, gamma_d, nx);
    else if(dimPoint==3 && dimVect==3)
        GaussGpuGradConvOnDevice<float,3,3><<<gridSize,blockSize,blockSize.x*(2*dimPoint+dimVect)*sizeof(float)>>>
        (ooSigma2, alpha_d, x_d, beta_d, gamma_d, nx);
    else
    {
        printf("error: dimensions of Gauss kernel not implemented in cuda");
    cudaFree(x_d);
    cudaFree(alpha_d);
    cudaFree(beta_d);
    cudaFree(gamma_d);
        return(-1);
    }

    // block until the device has completed
    cudaThreadSynchronize();

    // Send data from device to host.
    cudaMemcpy(gamma_h, gamma_d, sizeof(float)*(nx*dimPoint),cudaMemcpyDeviceToHost);

    // Free memory.
    cudaFree(x_d);
    cudaFree(alpha_d);
    cudaFree(beta_d);
    cudaFree(gamma_d);

    return 0;
}


////////////////////////////////////////////////////////////////////////////

#if UseCudaOnDoubles  
extern "C" int GaussGpuGradConv_double(double ooSigma2,
        double* alpha_h, double* x_h, double* beta_h, double* gamma_h,
        int dimPoint, int dimVect, int nx)
{

    // Data on the device.
    double* x_d;
    double* alpha_d;
    double* gamma_d;
    double* beta_d;

    // Allocate arrays on device.
    cudaMalloc((void**)&x_d, sizeof(double)*(nx*dimPoint));
    cudaMalloc((void**)&alpha_d, sizeof(double)*(nx*dimVect));
    cudaMalloc((void**)&beta_d, sizeof(double)*(nx*dimVect));
    cudaMalloc((void**)&gamma_d, sizeof(double)*(nx*dimPoint));

    // Send data from host to device.
    cudaMemcpy(x_d, x_h, sizeof(double)*(nx*dimPoint), cudaMemcpyHostToDevice);
    cudaMemcpy(alpha_d, alpha_h, sizeof(double)*(nx*dimVect), cudaMemcpyHostToDevice);
    cudaMemcpy(beta_d, beta_h, sizeof(double)*(nx*dimVect), cudaMemcpyHostToDevice);

    // compute on device.
    dim3 blockSize;
    blockSize.x = CUDA_BLOCK_SIZE; // number of threads in each block
    dim3 gridSize;
    gridSize.x =  nx / blockSize.x + (nx%blockSize.x==0 ? 0 : 1);

    if(dimPoint==1 && dimVect==1)
        GaussGpuGradConvOnDevice<double,1,1><<<gridSize,blockSize,blockSize.x*(2*dimPoint+dimVect)*sizeof(double)>>>
        (ooSigma2, alpha_d, x_d, beta_d, gamma_d, nx);
    else if(dimPoint==2 && dimVect==1)
        GaussGpuGradConvOnDevice<double,2,1><<<gridSize,blockSize,blockSize.x*(2*dimPoint+dimVect)*sizeof(double)>>>
        (ooSigma2, alpha_d, x_d, beta_d, gamma_d, nx);
    else if(dimPoint==2 && dimVect==2)
        GaussGpuGradConvOnDevice<double,2,2><<<gridSize,blockSize,blockSize.x*(2*dimPoint+dimVect)*sizeof(double)>>>
        (ooSigma2, alpha_d, x_d, beta_d, gamma_d, nx);
    else if(dimPoint==3 && dimVect==1)
        GaussGpuGradConvOnDevice<double,3,1><<<gridSize,blockSize,blockSize.x*(2*dimPoint+dimVect)*sizeof(double)>>>
        (ooSigma2, alpha_d, x_d, beta_d, gamma_d, nx);
    else if(dimPoint==3 && dimVect==3)
        GaussGpuGradConvOnDevice<double,3,3><<<gridSize,blockSize,blockSize.x*(2*dimPoint+dimVect)*sizeof(double)>>>
        (ooSigma2, alpha_d, x_d, beta_d, gamma_d, nx);
    else
    {
        printf("error: dimensions of Gauss kernel not implemented in cuda");
    cudaFree(x_d);
    cudaFree(alpha_d);
    cudaFree(beta_d);
    cudaFree(gamma_d);
        return(-1);
    }

    // block until the device has completed
    cudaThreadSynchronize();

    // Send data from device to host.
    cudaMemcpy(gamma_h, gamma_d, sizeof(double)*(nx*dimPoint),cudaMemcpyDeviceToHost);

    // Free memory.
    cudaFree(x_d);
    cudaFree(alpha_d);
    cudaFree(beta_d);
    cudaFree(gamma_d);

    return 0;
}
#endif




void ExitFcn(void)
{
  cudaDeviceReset();
}


//////////////////////////////////////////////////////////////////
///////////////// MEX ENTRY POINT ////////////////////////////////
//////////////////////////////////////////////////////////////////

 
 /* the gateway function */
 void mexFunction( int nlhs, mxArray *plhs[],
                   int nrhs, const mxArray *prhs[])
 //plhs: double *gamma
 //prhs: double *alpha, double *x, double *beta, double sigma
 
 { 
   // register an exit function to prevent crash at matlab exit or recompiling
   mexAtExit(ExitFcn);

   /*  check for proper number of arguments */
   if(nrhs != 4) 
     mexErrMsgTxt("4 inputs required.");
   if(nlhs < 1 | nlhs > 1) 
     mexErrMsgTxt("One output required.");
 
   //////////////////////////////////////////////////////////////
   // Input arguments
   //////////////////////////////////////////////////////////////
   
   int argu = -1;
 
   //------ the first input argument: alpha---------------//
   argu++;
   /*  create a pointer to the input vectors wts */
   double *alpha = mxGetPr(prhs[argu]);
   /*  get the dimensions of the input weights */
   int dimvect = mxGetM(prhs[argu]);
   int nx = mxGetN(prhs[argu]); //ncols
   
   //----- the second input argument: x--------------//
   argu++;
   /*  create a pointer to the input vectors srcs */
   double *x = mxGetPr(prhs[argu]);
   /*  input sources */
   int dimpoint = mxGetM(prhs[argu]); //mrows
    /* check to make sure the second dimension is nx */
   if( mxGetN(prhs[argu])!=nx )
     mexErrMsgTxt("Input x must have same number of columns as alpha.");

   //------ the third input argument: beta---------------//
   argu++;
   /*  create a pointer to the input vectors wts */
   double *beta = mxGetPr(prhs[argu]);
   /* check to make sure the first dimension is dimvect */
   if( mxGetM(prhs[argu])!=dimvect )
     mexErrMsgTxt("Input beta must have same number of rows as alpha.");
   /* check to make sure the second dimension is nx */
   if( mxGetN(prhs[argu])!=nx )
     mexErrMsgTxt("Input beta must have same number of columns as alpha.");

   //----- the fourth input argument: sigma-------------//
   argu++;
   /* check to make sure the input argument is a scalar */
   if( !mxIsDouble(prhs[argu]) || mxIsComplex(prhs[argu]) ||
       mxGetN(prhs[argu])*mxGetM(prhs[argu])!=1 ) {
     mexErrMsgTxt("Input sigma must be a scalar.");
   }
   /*  get the scalar input sigma */
   double sigma = mxGetScalar(prhs[argu]);
   if (sigma <= 0.0)
 	  mexErrMsgTxt("Input sigma must be a positive number.");
   double oosigma2 = 1.0f/(sigma*sigma);
 
   //////////////////////////////////////////////////////////////
   // Output arguments
   //////////////////////////////////////////////////////////////
   /*  set the output pointer to the output result(vector) */
   plhs[0] = mxCreateDoubleMatrix(dimpoint,nx,mxREAL);
   
   /*  create a C pointer to a copy of the output result(vector)*/
   double *gamma = mxGetPr(plhs[0]);

   
#if UseCudaOnDoubles
   GaussGpuGradConv_double(oosigma2,alpha,x,beta,gamma,dimpoint,dimvect,nx); 
#else
   // convert to float
   float *alpha_f = new float[nx*dimvect];
   float *x_f = new float[nx*dimpoint];
   float *beta_f = new float[nx*dimvect];
   float *gamma_f = new float[nx*dimpoint];
   for(int i=0; i<nx*dimvect; i++)
     alpha_f[i] = alpha[i];
   for(int i=0; i<nx*dimpoint; i++)
     x_f[i] = x[i];
   for(int i=0; i<nx*dimvect; i++)
     beta_f[i] = beta[i];
   
   // function calls;
   GaussGpuGradConv_float(oosigma2,alpha_f,x_f,beta_f,gamma_f,dimpoint,dimvect,nx);
 
   for(int i=0; i<nx*dimpoint; i++)
       gamma[i] = gamma_f[i];

   delete [] alpha_f;
   delete [] x_f;
   delete [] beta_f;
   delete [] gamma_f;
#endif

   return;
   
 }

