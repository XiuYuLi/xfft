#include<Windows.h>
#include<stdio.h>
#include<stdlib.h>
#include"xfft.h"
#include<cufft.h>
#include<vector_types.h>
#pragma comment( lib, "winmm.lib" )
#pragma comment( lib, "cuda.lib" )
#pragma comment( lib, "cufft.lib" )
#pragma comment( lib, "xfft.lib" )

int main()
{
	if(xfftInit()!=xfftSuccess){
		printf( "error: xfft init failed!\n" );
		exit(0);
	}
	int nx=8192;
	int ny=8192;
	xfftOp Op;
	//xfftCreateOp3d( &Op, 0, nx, ny, nz, 1 )
	if(xfftCreateOp2d( &Op, 0, nx, ny, 1 )!=xfftSuccess){
		printf( "error : Op create failed!\n" );
		xfftExit();
		return 0;
	}
	CUdeviceptr d_a, d_b, d_c;
	cuMemAlloc(&d_a,nx*ny*sizeof(float2));
	cuMemAlloc(&d_b,nx*ny*sizeof(float2));	
	float2* p=new float2[nx*ny];
	for( int y=0; y<ny; ++y )
	{
		for( int x=0; x<nx; ++x ){
			p[y*nx+x].x=(x*y+0.079155f)/(nx*ny);
			p[y*nx+x].y=(x*y+0.097317f)/(nx*ny);
		}
	}
	cuMemcpyHtoD( d_a, p, nx*ny*sizeof(float2));
	long start=timeGetTime();
	for( int i=0; i<500; ++i ){
		xfftExec( Op, &d_c, d_a, d_b, 0 );
	}
	cuCtxSynchronize();
	long end=timeGetTime();
	printf( "%d\n", end-start );
	cuMemFree(d_a);
	cuMemFree(d_b);	

	cufftHandle plan;
	CUdeviceptr d_C;
	//cufftPlan3d( &plan, nz, ny, nx, ... )
	cufftPlan2d( &plan, ny, nx, CUFFT_C2C );
	cuMemAlloc(&d_C, nx*ny*sizeof(cufftComplex));
	cuMemcpyHtoD( d_C, p, nx*ny*sizeof(cufftComplex));
	start=timeGetTime();
	for( int i=0; i<500; ++i ){
		cufftExecC2C( plan, (cufftComplex*)d_C, (cufftComplex*)d_C, CUFFT_FORWARD );
	}
	cuCtxSynchronize();
	end=timeGetTime();
	printf( "%d\n", end-start );

	FILE* fp=fopen( "result.txt", "wt" );
	for( int i=0; i<nx*ny; ++i ){
		fprintf( fp, "{%f, %f}, {%f, %f}\n", q[i].x, q[i].y, p[i].x, p[i].y );
	}
	fclose(fp);

	delete[] p;
	cuMemFree(d_C);
	cufftDestroy(plan);	
	xfftDestroy( Op );
	xfftExit();
	printf( "finished!\n" );
}
