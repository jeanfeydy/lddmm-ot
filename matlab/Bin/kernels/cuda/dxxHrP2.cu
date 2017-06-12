// Author : B. Charlier (2017)


#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <mex.h>


#define UseCudaOnDoubles USE_DOUBLE_PRECISION

///////////////////////////////////////
///// GRAD CONV ///////////////////////
///////////////////////////////////////

	template < typename TYPE, int DIMPOINT>
__global__ void GaussGpuGrad1ConvOnDevice (TYPE ooSigma2,TYPE ooSigma4, TYPE *alpha, TYPE *x, TYPE *beta, TYPE *gamma, int nx)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	// the following line does not work with nvcc 3.0 (it is a bug; it works with anterior and posterior versions)
	// extern __shared__ TYPE SharedData[];  // shared data will contain x and alpha data for the block
	// here is the bug fix (see http://forums.nvidia.com/index.php?showtopic=166905)
	extern __shared__ char SharedData_char[];
	TYPE* const SharedData = reinterpret_cast<TYPE*>(SharedData_char);
	// end of bug fix

	TYPE xi[DIMPOINT], xmx[DIMPOINT], alphai[DIMPOINT], betai[DIMPOINT], gammai[DIMPOINT];
	if(i<nx)  // we will compute gammai only if i is in the range
	{
		// load xi and alphai from device global memory
		for(int k=0; k<DIMPOINT; k++)
			xi[k] = x[i*DIMPOINT+k];
		for(int k=0; k<DIMPOINT; k++)
			alphai[k] = alpha[i*DIMPOINT+k];
		for(int k=0; k<DIMPOINT; k++)
			betai[k] = beta[i*DIMPOINT+k];
		for(int k=0; k<DIMPOINT; k++)
			gammai[k] = 0.0f;
	}

	for(int jstart = 0, tile = 0; jstart < nx; jstart += blockDim.x, tile++)
	{
		int j = tile * blockDim.x + threadIdx.x;
		if(j<nx) // we load xj and betaj from device global memory only if j<nx
		{
			int inc = DIMPOINT + DIMPOINT+ DIMPOINT;
			for(int k=0; k<DIMPOINT; k++)
				SharedData[threadIdx.x*inc+k] = x[j*DIMPOINT+k];
			for(int k=0; k<DIMPOINT; k++)
				SharedData[threadIdx.x*inc+DIMPOINT+k] = beta[j*DIMPOINT+k];
			for(int k=0; k<DIMPOINT; k++)
				SharedData[threadIdx.x*inc+DIMPOINT+DIMPOINT+k] = alpha[j*DIMPOINT+k];
		}

		__syncthreads();

		if(i<nx) // we compute gammai only if i is in the range
		{
			TYPE *xj, *betaj, *alphaj;
			xj = SharedData;
			betaj = SharedData + DIMPOINT;
			alphaj = SharedData + DIMPOINT + DIMPOINT;
			int inc = DIMPOINT + DIMPOINT + DIMPOINT;
			for(int jrel = 0; jrel < blockDim.x && jrel<nx-jstart; jrel++, xj+=inc, betaj+=inc, alphaj+=inc)
			{
				/*TYPE r2 = 0.0f, sga = 0.0f;*/
				TYPE r2 = 0.0f;
				for(int k=0; k<DIMPOINT; k++)
				{
					xmx[k] =  xi[k]-xj[k];
					r2 += xmx[k]*xmx[k];
				}
				TYPE ps = exp(-r2*ooSigma2);
				TYPE s =  4.0f * ooSigma4 * ps ;
				TYPE sdiag = - 2.0f * ooSigma2 * ps;

				TYPE t1,t2;
				for(int k=0; k<DIMPOINT; k++)
				{
					t1 = s *  xmx[k];
					for(int l=0; l<DIMPOINT; l++)
					{
						t2 =  (t1 *  xmx[l]  + (k==l ? 1 : 0) * sdiag) * (alphai[l] - alphaj[l]);

						for(int m=0; m<DIMPOINT; m++)
						{
							gammai[k] +=   t2 *  betaj[m] * betai[m]; 

							// equivalent without t1 and t2 variables

							// diag and off diag term separated
							/*gammai[k] +=    (s *  xmx[k] *  xmx[l] + (k==l ? 1 : 0) * sdiag ) * betaj[m] * alphai[l] * betai[m]; */
							/*gammai[k] +=  - (s *  xmx[k] *  xmx[l] + (k==l ? 1 : 0) * sdiag ) * betaj[m] * alphaj[l] * betai[m]; */

							// diag and off diag terms together
							/* gammai[k] +=    (s *  xmx[k] *  xmx[l] + (k==l ? 1 : 0) * sdiag ) * betaj[m] * (alphai[l] - alphaj[l]) * betai[m]; */
						}

					}
				}
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

	extern "C" int GaussGpuGrad1Conv_float(float ooSigma2, float ooSigma4,
			float* alpha_h, float* x_h, float* beta_h, float* gamma_h,
			int dimPoint, int nx)
	{

		// Data on the device.
		float* x_d;
		float* alpha_d;
		float* gamma_d;
		float* beta_d;

		// Allocate arrays on device.
		cudaMalloc((void**)&x_d, sizeof(float)*(nx*dimPoint));
		cudaMalloc((void**)&alpha_d, sizeof(float)*(nx*dimPoint));
		cudaMalloc((void**)&beta_d, sizeof(float)*(nx*dimPoint));
		cudaMalloc((void**)&gamma_d, sizeof(float)*(nx*dimPoint));

		// Send data from host to device.
		cudaMemcpy(x_d, x_h, sizeof(float)*(nx*dimPoint), cudaMemcpyHostToDevice);
		cudaMemcpy(alpha_d, alpha_h, sizeof(float)*(nx*dimPoint), cudaMemcpyHostToDevice);
		cudaMemcpy(beta_d, beta_h, sizeof(float)*(nx*dimPoint), cudaMemcpyHostToDevice);

		// compute on device.
		dim3 blockSize;
		blockSize.x = CUDA_BLOCK_SIZE; // number of threads in each block
		dim3 gridSize;
		gridSize.x =  nx / blockSize.x + (nx%blockSize.x==0 ? 0 : 1);

		if(dimPoint==1)
			GaussGpuGrad1ConvOnDevice<float,1><<<gridSize,blockSize,blockSize.x*(dimPoint+dimPoint+dimPoint)*sizeof(float)>>>
				(ooSigma2,ooSigma4, alpha_d, x_d, beta_d, gamma_d, nx);
		else if(dimPoint==2)
			GaussGpuGrad1ConvOnDevice<float,2><<<gridSize,blockSize,blockSize.x*(dimPoint+dimPoint+dimPoint)*sizeof(float)>>>
				(ooSigma2,ooSigma4, alpha_d, x_d, beta_d, gamma_d, nx);
		else if(dimPoint==3)
			GaussGpuGrad1ConvOnDevice<float,3><<<gridSize,blockSize,blockSize.x*(dimPoint+dimPoint+dimPoint)*sizeof(float)>>>
				(ooSigma2,ooSigma4, alpha_d, x_d, beta_d, gamma_d, nx);
		else
		{
			printf("error: dimensions of Gauss kernel not implemented in cuda");
			cudaFree(x_d);
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
		cudaFree(alpha_d);
		cudaFree(gamma_d);
		cudaFree(beta_d);

		return 0;
	}


	//////////////////////////////////////////////////////////////

#if UseCudaOnDoubles  
	extern "C" int GaussGpuGrad1Conv_double(double ooSigma2,double ooSigma4,
			double* alpha_h, double* x_h, double* beta_h, double* gamma_h,
			int dimPoint, int nx)
	{

		// Data on the device.
		double* x_d;
		double* alpha_d;
		double* gamma_d;
		double* beta_d;

		// Allocate arrays on device.
		cudaMalloc((void**)&x_d, sizeof(double)*(nx*dimPoint));
		cudaMalloc((void**)&alpha_d, sizeof(double)*(nx*dimPoint));
		cudaMalloc((void**)&beta_d, sizeof(double)*(nx*dimPoint));
		cudaMalloc((void**)&gamma_d, sizeof(double)*(nx*dimPoint));

		// Send data from host to device.
		cudaMemcpy(x_d, x_h, sizeof(double)*(nx*dimPoint), cudaMemcpyHostToDevice);
		cudaMemcpy(alpha_d, alpha_h, sizeof(double)*(nx*dimPoint), cudaMemcpyHostToDevice);
		cudaMemcpy(beta_d, beta_h, sizeof(double)*(nx*dimPoint), cudaMemcpyHostToDevice);

		// compute on device.
		dim3 blockSize;
		blockSize.x = CUDA_BLOCK_SIZE; // number of threads in each block
		dim3 gridSize;
		gridSize.x =  nx / blockSize.x + (nx%blockSize.x==0 ? 0 : 1);

		if(dimPoint==1)
			GaussGpuGrad1ConvOnDevice<double,1><<<gridSize,blockSize,blockSize.x*(dimPoint+dimPoint+dimPoint)*sizeof(double)>>>
				(ooSigma2,ooSigma4, alpha_d, x_d, beta_d, gamma_d, nx);
		else if(dimPoint==2)
			GaussGpuGrad1ConvOnDevice<double,2><<<gridSize,blockSize,blockSize.x*(dimPoint+dimPoint+dimPoint)*sizeof(double)>>>
				(ooSigma2,ooSigma4, alpha_d, x_d, beta_d, gamma_d, nx);
		else if(dimPoint==3)
			GaussGpuGrad1ConvOnDevice<double,3><<<gridSize,blockSize,blockSize.x*(dimPoint+dimPoint+dimPoint)*sizeof(double)>>>
				(ooSigma2,ooSigma4, alpha_d, x_d, beta_d, gamma_d, nx);
		else
		{
			printf("error: dimensions of Gauss kernel not implemented in cuda");
			cudaFree(x_d);
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
		int dimpoint = mxGetM(prhs[argu]);
		int nx = mxGetN(prhs[argu]); //ncols

		//----- the second input argument: x--------------//
		argu++;
		/*  create a pointer to the input vectors srcs */
		double *x = mxGetPr(prhs[argu]);
		/* check to make sure the number of columns is nx */
		if( mxGetN(prhs[argu])!=nx) {
			mexErrMsgTxt("Input x must have same number of columns as alpha.");
		if( mxGetM(prhs[argu])!=dimpoint)
			mexErrMsgTxt("Input x must have same number of rows as alpha.");
		}

		//------ the third input argument: beta---------------//
		argu++;
		/*  create a pointer to the input vectors wts */
		double *beta = mxGetPr(prhs[argu]);
		/* check to make sure the number of rows is dimvect */
		if( mxGetM(prhs[argu])!=dimpoint )
			mexErrMsgTxt("Input beta must have same number of rows as alpha.");
		/* check to make sure the number of columns is nx */
		if( mxGetN(prhs[argu])!=nx )
			mexErrMsgTxt("Input beta must have same number of columns as x.");

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
		double oosigma4 = oosigma2*oosigma2;

		//////////////////////////////////////////////////////////////
		// Output arguments
		//////////////////////////////////////////////////////////////
		/*  set the output pointer to the output result(vector) */
		plhs[0] = mxCreateDoubleMatrix(dimpoint,nx,mxREAL);

		/*  create a C pointer to a copy of the output result(vector)*/
		double *gamma = mxGetPr(plhs[0]);

#if UseCudaOnDoubles
		GaussGpuGrad1Conv_double(oosigma2,oosigma4,alpha,x,beta,gamma,dimpoint,nx);  
#else
		// convert to float
		float *alpha_f = new float[nx*dimpoint];
		float *x_f = new float[nx*dimpoint];
		float *beta_f = new float[nx*dimpoint];
		float *gamma_f = new float[nx*dimpoint];
		for(int i=0; i<nx*dimpoint; i++)
			alpha_f[i] = alpha[i];
		for(int i=0; i<nx*dimpoint; i++)
			x_f[i] = x[i];
		for(int i=0; i<nx*dimpoint; i++)
			beta_f[i] = beta[i];

		// function calls;
		GaussGpuGrad1Conv_float(oosigma2,oosigma4,alpha_f,x_f,beta_f,gamma_f,dimpoint,nx);

		for(int i=0; i<nx*dimpoint; i++)
			gamma[i] = gamma_f[i];

		delete [] alpha_f;
		delete [] x_f;
		delete [] beta_f;
		delete [] gamma_f;
#endif

		return;

	}
