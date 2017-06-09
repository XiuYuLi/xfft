#include"hfft.h"

__global__ void d_hfft64x( float2* d_o, const float2* __restrict__ d_i, const float2* __restrict__ d_RF, int bat )
{																		
	extern __shared__ float smem[];										
	float2 c[8], RF[7], temp;	
	unsigned int slice=blockIdx.x*blockDim.y+threadIdx.y;
	if(slice>=bat) return;
	d_i+=((slice<<6)+threadIdx.x);
	d_o+=((slice<<6)+threadIdx.x);
	float* spx=&smem[72*threadIdx.y+threadIdx.x];						
	float* spy=&smem[72*threadIdx.y+9*threadIdx.x];
	RF[0]=d_RF[threadIdx.x];
	mLOAD8(c,d_i,8,)
	mCALRF8(RF)
	mFFT8(c,)												
	mHMRF8(c,RF)									
	mPERMUTE(8,spx,spy,c,9,1,0)											
	mFFT8(c,)
	mISTORE8(d_o,c,8,)								
}
__global__ void d_hifft64x( float2* d_o, const float2* __restrict__ d_i, const float2* __restrict__ d_RF, int bat )
{																		
	extern __shared__ float smem[];										
	float2 c[8], RF[7], temp;	
	unsigned int slice=blockIdx.x*blockDim.y+threadIdx.y;
	if(slice>=bat) return;
	d_i+=((slice<<6)+threadIdx.x);
	d_o+=((slice<<6)+threadIdx.x);
	float* spx=&smem[72*threadIdx.y+threadIdx.x];						
	float* spy=&smem[72*threadIdx.y+9*threadIdx.x];
	RF[0]=d_RF[threadIdx.x];
	RF[0].y=-RF[0].y;
	mLOAD8(c,d_i,8,)
	mCALRF8(RF)
	mFFT8(c,i)												
	mHMRF8(c,RF)									
	mPERMUTE(8,spx,spy,c,9,1,1)											
	mFFT8(c,i)
	mISTORE8(d_o,c,8,)								
}
