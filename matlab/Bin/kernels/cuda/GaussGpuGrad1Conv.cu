// Author : B. Charlier (2017)

#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <mex.h>


#define UseCudaOnDoubles USE_DOUBLE_PRECISION

///////////////////////////////////////
///// GRAD CONV ///////////////////////
///////////////////////////////////////


	template < typename TYPE, int DIMPOINT, int DIMVECT >
__global__ void GaussGpuGrad1ConvOnDevice(TYPE ooSigma2,
        TYPE *alpha, TYPE *x, TYPE *y, TYPE *beta, TYPE *gamma,
        int nx, int ny)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // the following line does not work with nvcc 3.0 (it is a bug; it works with anterior and posterior versions)
    // extern __shared__ TYPE SharedData[];  // shared data will contain x and alpha data for the block
    // here is the bug fix (see http://forums.nvidia.com/index.php?showtopic=166905)
    extern __shared__ char SharedData_char[];
    TYPE* const SharedData = reinterpret_cast<TYPE*>(SharedData_char);
    // end of bug fix

    TYPE xi[DIMPOINT], alphai[DIMVECT], xmy[DIMPOINT], gammai[DIMPOINT];
    if(i<nx)  // we will compute gammai only if i is in the range
    {
        // load xi and alphai from device global memory
        for(int k=0; k<DIMPOINT; k++)
            xi[k] = x[i*DIMPOINT+k];
        for(int k=0; k<DIMVECT; k++)
            alphai[k] = alpha[i*DIMVECT+k];
        for(int k=0; k<DIMPOINT; k++)
            gammai[k] = 0.0f;
    }

    for(int jstart = 0, tile = 0; jstart < ny; jstart += blockDim.x, tile++)
    {
        int j = tile * blockDim.x + threadIdx.x;
        if(j<ny) // we load yj and betaj from device global memory only if j<ny
        {
            int inc = DIMPOINT + DIMVECT;
            for(int k=0; k<DIMPOINT; k++)
                SharedData[threadIdx.x*inc+k] = y[j*DIMPOINT+k];
            for(int k=0; k<DIMVECT; k++)
                SharedData[threadIdx.x*inc+DIMPOINT+k] = beta[j*DIMVECT+k];
        }
        __syncthreads();
        if(i<nx) // we compute gammai only if i is in the range
        {
            TYPE *yj, *betaj;
            yj = SharedData;
            betaj = SharedData + DIMPOINT;
            int inc = DIMPOINT + DIMVECT;
            for(int jrel = 0; jrel < blockDim.x && jrel<ny-jstart; jrel++, yj+=inc, betaj+=inc)
            {
                TYPE r2 = 0.0f, sga = 0.0f;
                for(int k=0; k<DIMPOINT; k++)
                {
                    xmy[k] =  xi[k]-yj[k];
                    r2 += xmy[k]*xmy[k];
                }
                for(int k=0; k<DIMVECT; k++)
                    sga += betaj[k]*alphai[k];
                TYPE s =  (-ooSigma2*2.0f*sga) * exp(-r2*ooSigma2);
                for(int k=0; k<DIMPOINT; k++)
                    gammai[k] += s * xmy[k];
            }
        }
        __syncthreads();
    }

    // Save the result in global memory.
    if(i<nx)
        for(int k=0; k<DIMPOINT; k++)
            gamma[i*DIMPOINT+k] = gammai[k];
}

//////////////////////////////////////////////////////////////

extern "C" int GaussGpuGrad1Conv_float(float ooSigma2,
                                       float* alpha_h, float* x_h, float* y_h, float* beta_h, float* gamma_h,
                                       int dimPoint, int dimVect, int nx, int ny)
{

    // Data on the device.
    float* x_d;
    float* y_d;
    float* alpha_d;
    float* gamma_d;
    float* beta_d;

    // Allocate arrays on device.
    cudaMalloc((void**)&x_d, sizeof(float)*(nx*dimPoint));
    cudaMalloc((void**)&y_d, sizeof(float)*(ny*dimPoint));
    cudaMalloc((void**)&alpha_d, sizeof(float)*(nx*dimVect));
    cudaMalloc((void**)&beta_d, sizeof(float)*(ny*dimVect));
    cudaMalloc((void**)&gamma_d, sizeof(float)*(nx*dimPoint));

    // Send data from host to device.
    cudaMemcpy(x_d, x_h, sizeof(float)*(nx*dimPoint), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y_h, sizeof(float)*(ny*dimPoint), cudaMemcpyHostToDevice);
    cudaMemcpy(alpha_d, alpha_h, sizeof(float)*(nx*dimVect), cudaMemcpyHostToDevice);
    cudaMemcpy(beta_d, beta_h, sizeof(float)*(ny*dimVect), cudaMemcpyHostToDevice);

    // compute on device.
    dim3 blockSize;
    blockSize.x = CUDA_BLOCK_SIZE; // number of threads in each block
    dim3 gridSize;
    gridSize.x =  nx / blockSize.x + (nx%blockSize.x==0 ? 0 : 1);

    if(dimPoint==1 && dimVect==1)
        GaussGpuGrad1ConvOnDevice<float,1,1><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect)*sizeof(float)>>>
        (ooSigma2, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==2 && dimVect==1)
        GaussGpuGrad1ConvOnDevice<float,2,1><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect)*sizeof(float)>>>
        (ooSigma2, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==2 && dimVect==2)
        GaussGpuGrad1ConvOnDevice<float,2,2><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect)*sizeof(float)>>>
        (ooSigma2, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==3 && dimVect==1)
        GaussGpuGrad1ConvOnDevice<float,3,1><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect)*sizeof(float)>>>
        (ooSigma2, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==3 && dimVect==3)
        GaussGpuGrad1ConvOnDevice<float,3,3><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect)*sizeof(float)>>>
        (ooSigma2, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
    else
    {
        printf("error: dimensions of Gauss kernel not implemented in cuda");
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(alpha_d);
    cudaFree(gamma_d);
    cudaFree(beta_d);
        return(-1);
    }

    // block until the device has completed
    cudaThreadSynchronize();

    // Send data from device to host.
    cudaMemcpy(gamma_h, gamma_d, sizeof(float)*(nx*dimPoint),cudaMemcpyDeviceToHost);

    // Free memory.
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(alpha_d);
    cudaFree(gamma_d);
    cudaFree(beta_d);

    return 0;
}


//////////////////////////////////////////////////////////////

#if UseCudaOnDoubles  
extern "C" int GaussGpuGrad1Conv_double(double ooSigma2,
                                       double* alpha_h, double* x_h, double* y_h, double* beta_h, double* gamma_h,
                                       int dimPoint, int dimVect, int nx, int ny)
{

    // Data on the device.
    double* x_d;
    double* y_d;
    double* alpha_d;
    double* gamma_d;
    double* beta_d;

    // Allocate arrays on device.
    cudaMalloc((void**)&x_d, sizeof(double)*(nx*dimPoint));
    cudaMalloc((void**)&y_d, sizeof(double)*(ny*dimPoint));
    cudaMalloc((void**)&alpha_d, sizeof(double)*(nx*dimVect));
    cudaMalloc((void**)&beta_d, sizeof(double)*(ny*dimVect));
    cudaMalloc((void**)&gamma_d, sizeof(double)*(nx*dimPoint));

    // Send data from host to device.
    cudaMemcpy(x_d, x_h, sizeof(double)*(nx*dimPoint), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y_h, sizeof(double)*(ny*dimPoint), cudaMemcpyHostToDevice);
    cudaMemcpy(alpha_d, alpha_h, sizeof(double)*(nx*dimVect), cudaMemcpyHostToDevice);
    cudaMemcpy(beta_d, beta_h, sizeof(double)*(ny*dimVect), cudaMemcpyHostToDevice);

    // compute on device.
    dim3 blockSize;
    blockSize.x = CUDA_BLOCK_SIZE; // number of threads in each block
    dim3 gridSize;
    gridSize.x =  nx / blockSize.x + (nx%blockSize.x==0 ? 0 : 1);

    if(dimPoint==1 && dimVect==1)
        GaussGpuGrad1ConvOnDevice<double,1,1><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect)*sizeof(double)>>>
        (ooSigma2, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==2 && dimVect==1)
        GaussGpuGrad1ConvOnDevice<double,2,1><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect)*sizeof(double)>>>
        (ooSigma2, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==2 && dimVect==2)
        GaussGpuGrad1ConvOnDevice<double,2,2><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect)*sizeof(double)>>>
        (ooSigma2, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==3 && dimVect==1)
        GaussGpuGrad1ConvOnDevice<double,3,1><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect)*sizeof(double)>>>
        (ooSigma2, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
    else if(dimPoint==3 && dimVect==3)
        GaussGpuGrad1ConvOnDevice<double,3,3><<<gridSize,blockSize,blockSize.x*(dimPoint+dimVect)*sizeof(double)>>>
        (ooSigma2, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny);
    else
    {
        printf("error: dimensions of Gauss kernel not implemented in cuda");
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(alpha_d);
    cudaFree(gamma_d);
    cudaFree(beta_d);
        return(-1);
    }

    // block until the device has completed
    cudaThreadSynchronize();

    // Send data from device to host.
    cudaMemcpy(gamma_h, gamma_d, sizeof(double)*(nx*dimPoint),cudaMemcpyDeviceToHost);

    // Free memory.
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(alpha_d);
    cudaFree(gamma_d);
    cudaFree(beta_d);

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
 //prhs: double *alpha, double *x, double *y, double *beta, double sigma
 
 { 
   // register an exit function to prevent crash at matlab exit or recompiling
   mexAtExit(ExitFcn);

   /*  check for proper number of arguments */
   if(nrhs != 5) 
     mexErrMsgTxt("5 inputs required.");
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
   /* check to make sure the number of columns is nx */
   if( mxGetN(prhs[argu])!=nx ) {
     mexErrMsgTxt("Input x must have same number of columns as alpha.");
   }
  
   //----- the third input argument: y--------------//
   argu++;
   /*  create a pointer to the input vectors tgts */
   double *y = mxGetPr(prhs[argu]);
   /*  input sources */
   int ny = mxGetN(prhs[argu]); //ncols
   /* check to make sure the number of rows is dimpoint */
   if( mxGetM(prhs[argu])!=dimpoint )
     mexErrMsgTxt("Input y must have same number of rows as x.");
 
   //------ the fourth input argument: beta---------------//
   argu++;
   /*  create a pointer to the input vectors wts */
   double *beta = mxGetPr(prhs[argu]);
   /* check to make sure the number of rows is dimvect */
   if( mxGetM(prhs[argu])!=dimvect )
     mexErrMsgTxt("Input y must have same number of rows as alpha.");
   /* check to make sure the number of columns is ny */
   if( mxGetN(prhs[argu])!=ny )
     mexErrMsgTxt("Input beta must have same number of columns as y.");

   //----- the fifth input argument: sigma-------------//
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
   GaussGpuGrad1Conv_double(oosigma2,alpha,x,y,beta,gamma,dimpoint,dimvect,nx,ny);  
#else
   // convert to float
   float *alpha_f = new float[nx*dimvect];
   float *x_f = new float[nx*dimpoint];
   float *y_f = new float[ny*dimpoint];
   float *beta_f = new float[ny*dimvect];
   float *gamma_f = new float[nx*dimpoint];
   for(int i=0; i<nx*dimvect; i++)
     alpha_f[i] = alpha[i];
   for(int i=0; i<nx*dimpoint; i++)
     x_f[i] = x[i];
   for(int i=0; i<ny*dimpoint; i++)
     y_f[i] = y[i];
   for(int i=0; i<ny*dimvect; i++)
     beta_f[i] = beta[i];
   
   // function calls;
   GaussGpuGrad1Conv_float(oosigma2,alpha_f,x_f,y_f,beta_f,gamma_f,dimpoint,dimvect,nx,ny);
 
   for(int i=0; i<nx*dimpoint; i++)
       gamma[i] = gamma_f[i];

   delete [] alpha_f;
   delete [] x_f;
   delete [] y_f;
   delete [] beta_f;
   delete [] gamma_f;
#endif

   return;
   
 }



