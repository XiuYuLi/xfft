﻿#include"hfft.h"

#if SM==37
#define NUM_CTA_8192 2
#else
#define NUM_CTA_8192 1
#endif

__global__ void __launch_bounds__(512,NUM_CTA_8192) d_hfft8192x( float2* d_o, const float2* __restrict__ d_i, const float2* __restrict__ d_RF )
{
	__shared__ float smem[16*641];
	float2 c[16], RF[15], temp;
	d_i+=((blockIdx.x<<13)+threadIdx.x);
	d_o+=((blockIdx.x<<13)+threadIdx.x);
	unsigned int xlane=threadIdx.x&63;
	unsigned int xslot=threadIdx.x>>6;
	unsigned int ylane=threadIdx.x&15;
	unsigned int yslot=threadIdx.x>>4;
	RF[0]=d_RF[threadIdx.x];
	float* spx=&smem[80*xslot+xlane];
	float* spy=&smem[641*xslot+xlane];
	float* spz=&smem[641*(yslot>>3)+80*(yslot&7)+ylane];
	float* spw=&smem[641*ylane+80*(yslot&7)+17*(yslot>>3)];
	mLOAD16(c,d_i,512,)
	mCALRF16(RF)	
	mFFT16(c,)	
	mHMRF16(c,RF)	
	RF[0]=d_RF[xlane<<4];
	mPERMUTE_S16_L8x2(spx,spy,c,641,5128,80,0xf)	
	mCALRF8(RF)
	mFFT8(&c[0],)
	mFFT8(&c[8],)	
	mHMRF8(&c[0],RF)
	mHMRF8(&c[8],RF)	
	RF[0]=d_RF[ylane<<7];	
	mISTORE8x2(spy,c,5128,80,.x)	__syncthreads();
	mLOAD4x4(c,spz,2564,16,.x)	__syncthreads();
	mISTORE8x2(spy,c,5128,80,.y)	__syncthreads();
	mLOAD4x4(c,spz,2564,16,.y)	__syncthreads();	
	mCALRF4(RF)
	mFFT4(&c[ 0],)
	mFFT4(&c[ 4],)
	mFFT4(&c[ 8],)
	mFFT4(&c[12],)
	mHMRF4(&c[ 0],RF)
	mHMRF4(&c[ 4],RF)
	mHMRF4(&c[ 8],RF)
	mHMRF4(&c[12],RF)
	mIPERMUTE4x4(spz,spw,c,2564,17,1,0x7)
	mFFT16(c,)
	mISTORE16(d_o,c,512,)
}
__global__ void __launch_bounds__(512,NUM_CTA_8192) d_hifft8192x( float2* d_o, const float2* __restrict__ d_i, const float2* __restrict__ d_RF )
{
	__shared__ float smem[16*641];
	float2 c[16], RF[15], temp;
	d_i+=((blockIdx.x<<13)+threadIdx.x);
	d_o+=((blockIdx.x<<13)+threadIdx.x);
	unsigned int xlane=threadIdx.x&63;
	unsigned int xslot=threadIdx.x>>6;
	unsigned int ylane=threadIdx.x&15;
	unsigned int yslot=threadIdx.x>>4;
	float* spx=&smem[80*xslot+xlane];
	float* spy=&smem[641*xslot+xlane];
	float* spz=&smem[641*(yslot>>3)+80*(yslot&7)+ylane];
	float* spw=&smem[641*ylane+80*(yslot&7)+17*(yslot>>3)];
	RF[0]=d_RF[threadIdx.x];	
	RF[0].y=-RF[0].y;
	mLOAD16(c,d_i,512,)
	mCALRF16(RF)
	mFFT16(c,i)	
	mHMRF16(c,RF)
	RF[0]=d_RF[xlane<<4];
	RF[0].y=-RF[0].y;
	mPERMUTE_S16_L8x2(spx,spy,c,641,5128,80,0xf)
	mCALRF8(RF)
	mFFT8(&c[0],i)
	mFFT8(&c[8],i)
	mHMRF8(&c[0],RF)
	mHMRF8(&c[8],RF)	
	RF[0]=d_RF[ylane<<7];
	RF[0].y=-RF[0].y;
	mISTORE8x2(spy,c,5128,80,.x)	__syncthreads();
	mLOAD4x4(c,spz,2564,16,.x)	__syncthreads();
	mISTORE8x2(spy,c,5128,80,.y)	__syncthreads();
	mLOAD4x4(c,spz,2564,16,.y)	__syncthreads();
	mCALRF4(RF)
	mFFT4(&c[ 0],i)
	mFFT4(&c[ 4],i)
	mFFT4(&c[ 8],i)
	mFFT4(&c[12],i)
	mHMRF4(&c[ 0],RF)
	mHMRF4(&c[ 4],RF)
	mHMRF4(&c[ 8],RF)
	mHMRF4(&c[12],RF)
	mIPERMUTE4x4(spz,spw,c,2564,17,1,0x7)
	mFFT16(c,i)
	mISTORE16(d_o,c,512,)
}
