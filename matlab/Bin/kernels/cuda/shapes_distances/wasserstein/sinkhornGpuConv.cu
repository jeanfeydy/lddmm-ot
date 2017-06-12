// Author : B. Charlier (2017)

#define UseCudaOnDoubles USE_DOUBLE_PRECISION

#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <mex.h>
#include "sinkhornGpuUtils.cx"



template< typename TYPE >
int SinkhornGpuStep(TYPE epsilon,TYPE lambda, TYPE weightGeom,TYPE weightGrass,
		TYPE* alpha_h, TYPE* x_h, TYPE* y_h, TYPE* beta_h, TYPE* mu_h, TYPE* nu_h, TYPE* gammax_h, TYPE* gammay_h, TYPE* gammaWd,
		int dimPoint, int dimVect, int nx, int ny, int max_iter)
{

	// Data on the device.
	TYPE* alpha_d;
	TYPE* x_d;
	TYPE* y_d;
	TYPE* beta_d;
	TYPE* mu_d;
	TYPE* nu_d;
	TYPE* gammax_d;
	TYPE* gammay_d;

	// Allocate arrays on device.
	cudaMalloc((void**)&alpha_d, sizeof(TYPE)*(nx*dimVect));
	cudaMalloc((void**)&x_d, sizeof(TYPE)*(nx*dimPoint));
	cudaMalloc((void**)&y_d, sizeof(TYPE)*(ny*dimPoint));
	cudaMalloc((void**)&beta_d, sizeof(TYPE)*(ny*dimVect));
	cudaMalloc((void**)&mu_d, sizeof(TYPE)*(nx));
	cudaMalloc((void**)&nu_d, sizeof(TYPE)*(ny));
	cudaMalloc((void**)&gammax_d, sizeof(TYPE)*(nx*dimVect));
	cudaMalloc((void**)&gammay_d, sizeof(TYPE)*(ny*dimVect));

	// Send data from host to device.
	cudaMemcpy(x_d, x_h, sizeof(TYPE)*(nx*dimPoint), cudaMemcpyHostToDevice);
	cudaMemcpy(y_d, y_h, sizeof(TYPE)*(ny*dimPoint), cudaMemcpyHostToDevice);
	cudaMemcpy(mu_d, mu_h, sizeof(TYPE)*(nx), cudaMemcpyHostToDevice);
	cudaMemcpy(nu_d, nu_h, sizeof(TYPE)*(ny), cudaMemcpyHostToDevice);

	cudaMemcpy(alpha_d, alpha_h, sizeof(TYPE)*(nx*dimVect), cudaMemcpyHostToDevice);
	cudaMemcpy(beta_d, beta_h, sizeof(TYPE)*(ny*dimVect), cudaMemcpyHostToDevice);

	// compute on device.
	dim3 blockSizex;
	blockSizex.x = CUDA_BLOCK_SIZE; // number of threads in each block
	dim3 gridSizex;
	gridSizex.x =  nx / blockSizex.x + (nx%blockSizex.x==0 ? 0 : 1);

	dim3 blockSizey;
	blockSizey.x = CUDA_BLOCK_SIZE; // number of threads in each block
	dim3 gridSizey;
	gridSizey.x =  ny / blockSizey.x + (ny%blockSizey.x==0 ? 0 : 1);

	for (int iter = 0; iter<max_iter; iter++)
	{


		if(dimPoint==2 && dimVect==1)
			SinkhornGpuGradConvOnDevice<TYPE,2,1><<<gridSizex,blockSizex,blockSizex.x*(dimPoint+dimVect)*sizeof(TYPE)>>>
				(epsilon,lambda,weightGeom,weightGrass, alpha_d, x_d, y_d, beta_d, mu_d, gammax_d, nx, ny);
		else if(dimPoint==4 && dimVect==1)
			SinkhornGpuGradConvOnDevice<TYPE,4,1><<<gridSizex,blockSizex,blockSizex.x*(dimPoint+dimVect)*sizeof(TYPE)>>>
				(epsilon,lambda,weightGeom,weightGrass, alpha_d, x_d, y_d, beta_d, mu_d, gammax_d, nx, ny);
		else if(dimPoint==6 && dimVect==1)
			SinkhornGpuGradConvOnDevice<TYPE,6,1><<<gridSizex,blockSizex,blockSizex.x*(dimPoint+dimVect)*sizeof(TYPE)>>>
				(epsilon,lambda,weightGeom,weightGrass, alpha_d, x_d, y_d, beta_d, mu_d, gammax_d, nx, ny);
		else
		{
			printf("error: dimensions of Sinkhorn kernel not implemented in cuda");
			cudaFree(alpha_d);
			cudaFree(x_d);
			cudaFree(y_d);
			cudaFree(beta_d);
			cudaFree(mu_d);
			cudaFree(nu_d);
			cudaFree(gammax_d);
			cudaFree(gammay_d);
			return(-1);
		}

		// update u
		cudaMemcpy(alpha_d,gammax_d, sizeof(TYPE)*(nx*dimVect), cudaMemcpyDeviceToDevice);

		if(dimPoint==2 && dimVect==1)
			SinkhornGpuGradConvOnDevice<TYPE,2,1><<<gridSizey,blockSizey,blockSizey.x*(dimPoint+dimVect)*sizeof(TYPE)>>>
				(epsilon,lambda,weightGeom,weightGrass, beta_d, y_d, x_d, alpha_d, nu_d, gammay_d, ny, nx);
		else if(dimPoint==4 && dimVect==1)
			SinkhornGpuGradConvOnDevice<TYPE,4,1><<<gridSizey,blockSizey,blockSizey.x*(dimPoint+dimVect)*sizeof(TYPE)>>>
				(epsilon,lambda,weightGeom,weightGrass, beta_d, y_d, x_d, alpha_d, nu_d, gammay_d, ny, nx);
		else if(dimPoint==6 && dimVect==1)
			SinkhornGpuGradConvOnDevice<TYPE,6,1><<<gridSizey,blockSizey,blockSizey.x*(dimPoint+dimVect)*sizeof(TYPE)>>>
				(epsilon,lambda,weightGeom,weightGrass, beta_d, y_d, x_d, alpha_d, nu_d, gammay_d, ny, nx);
		else
		{
			printf("error: dimensions of SinkhornGpuGradConvOnDevice kernel not implemented in cuda");
			cudaFree(alpha_d);
			cudaFree(x_d);
			cudaFree(y_d);
			cudaFree(beta_d);
			cudaFree(mu_d);
			cudaFree(nu_d);
			cudaFree(gammax_d);
			cudaFree(gammay_d);
			return(-1);
		}

		// update v
		cudaMemcpy(beta_d, gammay_d,sizeof(TYPE)*(ny*dimVect), cudaMemcpyDeviceToDevice);

	}

	// block until the device has completed
	cudaThreadSynchronize();

	// Send data from device to host.
	cudaMemcpy(gammax_h,alpha_d,  sizeof(TYPE)*(nx*dimVect),cudaMemcpyDeviceToHost);
	cudaMemcpy(gammay_h,beta_d,  sizeof(TYPE)*(ny*dimVect),cudaMemcpyDeviceToHost);

	/*-------------------------------------------------------------------------------*/

	// Compute  dual energy values
	TYPE dotpumu = 0.0f, dotpvnu = 0.0f, totalmass = 0.0f;
	for (int i=0;i<nx;i++)
		dotpumu += gammax_h[i] * mu_h[i];

	for (int j=0;j<ny;j++)
		dotpvnu += gammay_h[j] * nu_h[j];

	if(dimPoint==2 && dimVect==1)
		WdualOnDevice<TYPE,2,1><<<gridSizex,blockSizex,blockSizex.x*(dimPoint+dimVect)*sizeof(TYPE)>>>
			(epsilon,weightGeom,weightGrass,alpha_d, x_d, y_d, beta_d, gammax_d, nx, ny);
	else if(dimPoint==4 && dimVect==1)
		WdualOnDevice<TYPE,4,1><<<gridSizex,blockSizex,blockSizex.x*(dimPoint+dimVect)*sizeof(TYPE)>>>
			(epsilon,weightGeom,weightGrass,alpha_d, x_d, y_d, beta_d, gammax_d, nx, ny);
	else if(dimPoint==6 && dimVect==1)
		WdualOnDevice<TYPE,6,1><<<gridSizex,blockSizex,blockSizex.x*(dimPoint+dimVect)*sizeof(TYPE)>>>
			(epsilon,weightGeom,weightGrass,alpha_d, x_d, y_d, beta_d, gammax_d, nx, ny);
	else
	{
		printf("error: dimensions of Wdual kernel not implemented in cuda");
		cudaFree(alpha_d);
		cudaFree(x_d);
		cudaFree(y_d);
		cudaFree(beta_d);
		cudaFree(mu_d);
		cudaFree(nu_d);
		cudaFree(gammax_d);
		cudaFree(gammay_d);
		return(-1);
	}

	// block until the device has completed
	cudaThreadSynchronize();

	// Send data from device to host.
	TYPE *Gammax_h = new TYPE[nx];
	cudaMemcpy(Gammax_h,gammax_d,  sizeof(TYPE)*(nx*dimVect),cudaMemcpyDeviceToHost);
	for (int i=0;i<nx;i++)
	{
		totalmass += Gammax_h[i]; 
		/*printf("%g\n", Gammax_h[i]);*/
	}
	// compute the energy
	*gammaWd  = totalmass + dotpvnu + dotpumu ;

	/*-------------------------------------------------------------------------------*/

	// Free memory.
	cudaFree(alpha_d);
	cudaFree(x_d);
	cudaFree(y_d);
	cudaFree(mu_d);
	cudaFree(nu_d);
	cudaFree(beta_d);
	cudaFree(gammax_d);
	cudaFree(gammay_d);

	return 0;
}

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
	//prhs: double *alpha, double *x, double *y, double *beta, double epsilon

{ 
	// register an exit function to prevent crash at matlab exit or recompiling
	mexAtExit(ExitFcn);

	/*  check for proper number of arguments */
	if(nrhs != 10) 
		mexErrMsgTxt("10 inputs required.");
	if(nlhs < 3 | nlhs > 3) 
		mexErrMsgTxt("Three outputs required.");

	//////////////////////////////////////////////////////////////
	// Input arguments
	//////////////////////////////////////////////////////////////

	int argu = -1;

	//------ argument: alpha---------------//
	argu++;
	/*  create a pointer to the input vectors wts */
	double *alpha = mxGetPr(prhs[argu]);
	/*  get the dimensions of the input weights */
	int dimvect = mxGetM(prhs[argu]);
	int nx = mxGetN(prhs[argu]); //ncols

	//----- argument: x--------------//
	argu++;
	/*  create a pointer to the input vectors srcs */
	double *x = mxGetPr(prhs[argu]);
	/*  input sources */
	int dimpoint = mxGetM(prhs[argu]); //mrows
	/* check to make sure the second dimension is nx */
	if( mxGetN(prhs[argu])!=nx )
		mexErrMsgTxt("Input x must have same number of columns as alpha.");

	//----- argument: y--------------//
	argu++;
	/*  create a pointer to the input vectors srcs */
	double *y = mxGetPr(prhs[argu]);
	/*  input sources */
	int ny = mxGetN(prhs[argu]); //ncols
	/* check to make sure the second dimension is nx */
	if( mxGetM(prhs[argu])!=dimpoint )
		mexErrMsgTxt("Input y must have same number of rows as x.");

	//------ argument: beta---------------//
	argu++;
	/*  create a pointer to the input vectors wts */
	double *beta = mxGetPr(prhs[argu]);
	/* check to make sure the first dimension is dimvect */
	if( mxGetM(prhs[argu])!=dimvect )
		mexErrMsgTxt("Input beta must have same number of rows as alpha.");
	/* check to make sure the second dimension is nx */
	if( mxGetN(prhs[argu])!=ny )
		mexErrMsgTxt("Input beta must have same number of columns as y.");

	//----- argument: epsilon-------------//
	argu++;
	/* check to make sure the input argument is a scalar */
	if( !mxIsDouble(prhs[argu]) || mxIsComplex(prhs[argu]) ||
			mxGetN(prhs[argu])*mxGetM(prhs[argu])!=1 ) {
		mexErrMsgTxt("Input epsilon must be a scalar.");
	}
	/*  get the scalar input epsilon */
	double epsilon = mxGetScalar(prhs[argu]);
	if (epsilon <= 0.0)
		mexErrMsgTxt("Input epsilon must be a positive number.");

	//----- argument: lambda-------------//
	argu++;
	/* check to make sure the input argument is a scalar */
	if( !mxIsDouble(prhs[argu]) || mxIsComplex(prhs[argu]) ||
			mxGetN(prhs[argu])*mxGetM(prhs[argu])!=1 ) {
		mexErrMsgTxt("Input lambda  must be a scalar.");
	}
	/*  get the scalar input lambda  */
	double lambda  = mxGetScalar(prhs[argu]);
	if (lambda  <= 0.0)
		mexErrMsgTxt("Input lambda  must be a positive number.");

	//------ argument: mu---------------//
	argu++;
	/*  create a pointer to the input vectors wts */
	double *mu = mxGetPr(prhs[argu]);
	/* check to make sure the first dimension is dimvect */
	if( mxGetM(prhs[argu])!=1 )
		mexErrMsgTxt("Input mu must have one row.");
	/* check to make sure the second dimension is nx */
	if( mxGetN(prhs[argu])!=nx )
		mexErrMsgTxt("Input mu must have same number of columns as x.");

	//------ argument: nu---------------//
	argu++;
	/*  create a pointer to the input vectors wts */
	double *nu = mxGetPr(prhs[argu]);
	/* check to make sure the first dimension is dimvect */
	if( mxGetM(prhs[argu])!=1 )
		mexErrMsgTxt("Input nu must have one row.");
	/* check to make sure the second dimension is nx */
	if( mxGetN(prhs[argu])!=ny )
		mexErrMsgTxt("Input nu must have same number of columns as y.");

	//----- argument: weight -------------//
	argu++;
	/* check to make sure the input argument is a scalar */
	if( mxGetN(prhs[argu])*mxGetM(prhs[argu])!=2 ) {
		mexErrMsgTxt("Input weight must be a 2 vector.");
	}
	/* Get the values*/
	double *weightG = mxGetPr(prhs[argu]);

	//----- argument: max_iter -------------//
	argu++;
	/* check to make sure the input argument is a scalar */
	if( mxGetN(prhs[argu])*mxGetM(prhs[argu])!=1 ) {
		mexErrMsgTxt("Input max_iter must be an integer.");
	}
	if (!mxIsInt32(prhs[argu])){
		mexErrMsgTxt("Input max_iter must be an integer. Use int32() to cast.");
	}
	/* Get the values*/
	int max_iter = (int)mxGetScalar(prhs[argu]);

	//////////////////////////////////////////////////////////////
	// Output arguments
	//////////////////////////////////////////////////////////////

	/*  set the output pointer to the output result(vector) */
	plhs[0] = mxCreateDoubleMatrix(dimvect,nx,mxREAL);
	/*  create a C pointer to a copy of the output result(vector)*/
	double *gammax = mxGetPr(plhs[0]);


	/*  set the output pointer to the output result(vector) */
	plhs[1] = mxCreateDoubleMatrix(dimvect,ny,mxREAL);
	/*  create a C pointer to a copy of the output result(vector)*/
	double *gammay = mxGetPr(plhs[1]);


	/*  set the output pointer to the output result(vector) */
	plhs[2] = mxCreateDoubleMatrix(1,1,mxREAL);
	/*  create a C pointer to a copy of the output result(vector)*/
	double *gammaWd = mxGetPr(plhs[2]);

#if UseCudaOnDoubles
	SinkhornGpuStep<double>(epsilon,lambda,weightG[0],weightG[1],alpha,x,y,beta,mu,nu,gammax,gammay,gammaWd,dimpoint,dimvect,nx,ny,max_iter); 
#else
	// convert to float
	float *alpha_f = new float[nx*dimvect];
	float *x_f = new float[nx*dimpoint];
	float *y_f = new float[ny*dimpoint];
	float *beta_f = new float[ny*dimvect];
	float *mu_f = new float[nx];
	float *nu_f = new float[ny];
	float *gammax_f = new float[nx];
	float *gammay_f = new float[ny];
	float gammaWd_f;

	for(int i=0; i<nx*dimvect; i++)
		alpha_f[i] = alpha[i];
	for(int i=0; i<nx*dimpoint; i++)
		x_f[i] = x[i];
	for(int i=0; i<ny*dimpoint; i++)
		y_f[i] = y[i];
	for(int i=0; i<ny*dimvect; i++)
		beta_f[i] = beta[i];
	for(int i=0; i<nx; i++)
		mu_f[i] = mu[i];
	for(int i=0; i<ny; i++)
		nu_f[i] = nu[i];

	// function calls;
	SinkhornGpuStep<float>(epsilon,lambda,weightG[0],weightG[1],alpha_f,x_f,y_f,beta_f,mu_f,nu_f,gammax_f,gammay_f,&gammaWd_f,dimpoint,dimvect,nx,ny,max_iter);

	for(int i=0; i<nx; i++)
		gammax[i] = gammax_f[i];

	for(int i=0; i<nx; i++)
		gammay[i] = gammay_f[i];

	*gammaWd = double(gammaWd_f);

	delete [] alpha_f;
	delete [] x_f;
	delete [] y_f;
	delete [] beta_f;
	delete [] mu_f;
	delete [] nu_f;
	delete [] gammax_f;
	delete [] gammay_f;
#endif

	return;

}
