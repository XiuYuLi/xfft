#include"vfft.h"
#include"hfft.h"

#define CALRF8 mCALRF8
#define iCALRF8(RF){					\
	RF[1].x=RF[0].x* RF[0].x+RF[0].y*RF[0].y;	\
	RF[1].y=RF[0].x*-RF[0].y-RF[0].y*RF[0].x;	\
	RF[2].x=RF[0].x* RF[1].x-RF[0].y*RF[1].y;	\
	RF[2].y=RF[0].x* RF[1].y+RF[0].y*RF[1].x;	\
	RF[3].x=RF[1].x* RF[1].x-RF[1].y*RF[1].y;	\
	RF[3].y=RF[1].x* RF[1].y+RF[1].y*RF[1].x;	\
	RF[4].x=RF[1].x* RF[2].x-RF[1].y*RF[2].y;	\
	RF[4].y=RF[1].x* RF[2].y+RF[1].y*RF[2].x;	\
	RF[5].x=RF[2].x* RF[2].x-RF[2].y*RF[2].y;	\
	RF[5].y=RF[2].x* RF[2].y+RF[2].y*RF[2].x;	\
	RF[6].x=RF[2].x* RF[3].x-RF[2].y*RF[3].y;	\
	RF[6].y=RF[2].x* RF[3].y+RF[2].y*RF[3].x;	\
}

#define CUDA_VFFT64(LB,dir,e,bdx,op) CUDA_VFFT_DECL(64,LB,dir,e,PRF)			\
{											\
	__shared__ float smem[8*8*bdx];							\
	float2 c[8], temp;								\
	size_t p=((blockIdx.y<<6)+threadIdx.y)*(1<<e)+blockIdx.x*bdx+threadIdx.x;	\
	float* sst=&smem[8*bdx*threadIdx.y+threadIdx.x];				\
	float* sld=&smem[  bdx*threadIdx.y+threadIdx.x];				\
	d_i+=p; d_o+=p;									\
	mLOAD8(c,d_i,8*(1<<e),)								\
	mFFT8(c,dir)									\
	mVMRF8(&d_RF[threadIdx.y<<3],dir,op)						\
	mPERMUTE(8,sst,sld,c,bdx,8*bdx,7)						\
	mFFT8(c,dir)									\
	mISTORE8(d_o,c,8*(1<<e),)							\
}
#define CUDA_UFFT64X(LB,dir,n,e,bdx,op) CUDA_UFFTX_DECL(64,LB,dir,n,e)			\
{                                                                                       \
	__shared__ float smem[8*8*bdx];							\
	float2 c[8], RF[7], temp;							\
	unsigned int slot=blockIdx.x/((1<<e)/bdx);					\
	unsigned int bidx=blockIdx.x&(((1<<e)/bdx)-1);					\
	size_t p=blockIdx.y*n*64*(1<<e)+slot*(1<<e)+threadIdx.y*n*(1<<e)+bidx*bdx+threadIdx.x;	\
	d_i+=p;	d_o+=p;										\
	slot+=threadIdx.y*n;									\
	RF[0]=__fld##op##2(&d_RF[4+(slot<<3)]);							\
	mLOAD8(c,d_i,8*n*(1<<e),)								\
	dir##CALRF8(RF)										\
	mFFT8(c,dir)										\
	mHMRF8(c,RF)										\
	RF[0]=__fld##op##2(&d_RF[4+((slot&(n-1))<<6)]);						\
	float* sst=&smem[8*bdx*threadIdx.y+threadIdx.x];					\
	float* sld=&smem[  bdx*threadIdx.y+threadIdx.x];					\
	mPERMUTE(8,sst,sld,c,bdx,8*bdx,7)							\
	dir##CALRF8(RF)										\
	mFFT8(c,dir)										\
	mHMRF8(c,RF)										\
	mISTORE8(d_o,c,8*n*(1<<e),)								\
}
#define CUDA_VFFT64X(LB,dir,n,e,bdx,op) CUDA_VFFTX_DECL(64,LB,dir,n,e,PRF)			\
{												\
	__shared__ float smem[8*8*32];								\
	float2 c[8], temp;									\
	unsigned int slot=blockIdx.x/((1<<e)/bdx);						\
	unsigned int bidx=blockIdx.x&(((1<<e)/bdx)-1);						\
	size_t p=blockIdx.y*n*64*(1<<e)+bidx*bdx+threadIdx.x;					\
	d_i+=(p+slot*64*(1<<e)+threadIdx.y*(1<<e));						\
	d_o+=(p+slot*(1<<e)+threadIdx.y*n*(1<<e));						\
	mLOAD8(c,d_i,8*(1<<e),)									\
	mFFT8(c,dir)										\
	mVMRF8(&d_RF[8*n*threadIdx.y],dir,op)							\
	float* sst=&smem[8*bdx*threadIdx.y+threadIdx.x];					\
	float* sld=&smem[  bdx*threadIdx.y+threadIdx.x];					\
	mPERMUTE(8,sst,sld,c,bdx,8*bdx,7)							\
	mFFT8(c,dir)										\
	mISTORE8(d_o,c,8*n*(1<<e),)								\
}

#if SM==37
#define NUM_CTA_V64 8
#else
#define NUM_CTA_V64 4
#endif

CUDA_VFFT64(,,1, 2,g)
CUDA_VFFT64(,,2, 4,g)
CUDA_VFFT64(,,3, 8,g)
CUDA_VFFT64(,,4,16,g)
CUDA_VFFT64(LB(256,NUM_CTA_V64),, 5,32,u)
CUDA_VFFT64(LB(256,NUM_CTA_V64),, 6,32,u)
CUDA_VFFT64(LB(256,NUM_CTA_V64),, 7,32,u)
CUDA_VFFT64(LB(256,NUM_CTA_V64),, 8,32,u)
CUDA_VFFT64(LB(256,NUM_CTA_V64),, 9,32,u)
CUDA_VFFT64(LB(256,NUM_CTA_V64),,10,32,u)
CUDA_VFFT64(LB(256,NUM_CTA_V64),,11,32,u)
CUDA_VFFT64(LB(256,NUM_CTA_V64),,12,32,u)
CUDA_VFFT64(LB(256,NUM_CTA_V64),,13,32,u)
CUDA_VFFT64(LB(256,NUM_CTA_V64),,14,32,u)
CUDA_VFFT64(LB(256,NUM_CTA_V64),,15,32,u)
CUDA_VFFT64(LB(256,NUM_CTA_V64),,16,32,u)

CUDA_VFFT64(,i,1, 2,g)
CUDA_VFFT64(,i,2, 4,g)
CUDA_VFFT64(,i,3, 8,g)
CUDA_VFFT64(,i,4,16,g)
CUDA_VFFT64(LB(256,NUM_CTA_V64),i, 5,32,u)
CUDA_VFFT64(LB(256,NUM_CTA_V64),i, 6,32,u)
CUDA_VFFT64(LB(256,NUM_CTA_V64),i, 7,32,u)
CUDA_VFFT64(LB(256,NUM_CTA_V64),i, 8,32,u)
CUDA_VFFT64(LB(256,NUM_CTA_V64),i, 9,32,u)
CUDA_VFFT64(LB(256,NUM_CTA_V64),i,10,32,u)
CUDA_VFFT64(LB(256,NUM_CTA_V64),i,11,32,u)
CUDA_VFFT64(LB(256,NUM_CTA_V64),i,12,32,u)
CUDA_VFFT64(LB(256,NUM_CTA_V64),i,13,32,u)
CUDA_VFFT64(LB(256,NUM_CTA_V64),i,14,32,u)
CUDA_VFFT64(LB(256,NUM_CTA_V64),i,15,32,u)
CUDA_VFFT64(LB(256,NUM_CTA_V64),i,16,32,u)

CUDA_UFFT64X(,,32,1, 2,g)
CUDA_UFFT64X(,,32,2, 4,g)
CUDA_UFFT64X(,,32,3, 8,g)
CUDA_UFFT64X(,,32,4,16,g)
CUDA_UFFT64X(LB(256,NUM_CTA_V64),,32, 5,32,u)
CUDA_UFFT64X(LB(256,NUM_CTA_V64),,32, 6,32,u)
CUDA_UFFT64X(LB(256,NUM_CTA_V64),,32, 7,32,u)
CUDA_UFFT64X(LB(256,NUM_CTA_V64),,32, 8,32,u)
CUDA_UFFT64X(LB(256,NUM_CTA_V64),,32, 9,32,u)
CUDA_UFFT64X(LB(256,NUM_CTA_V64),,32,10,32,u)
CUDA_UFFT64X(LB(256,NUM_CTA_V64),,32,11,32,u)
CUDA_UFFT64X(LB(256,NUM_CTA_V64),,32,12,32,u)
CUDA_UFFT64X(LB(256,NUM_CTA_V64),,32,13,32,u)
CUDA_UFFT64X(LB(256,NUM_CTA_V64),,32,14,32,u)
CUDA_UFFT64X(LB(256,NUM_CTA_V64),,32,15,32,u)
CUDA_UFFT64X(LB(256,NUM_CTA_V64),,32,16,32,u)

CUDA_UFFT64X(,i,32,1, 2,g)
CUDA_UFFT64X(,i,32,2, 4,g)
CUDA_UFFT64X(,i,32,3, 8,g)
CUDA_UFFT64X(,i,32,4,16,g)
CUDA_UFFT64X(LB(256,NUM_CTA_V64),i,32, 5,32,u)
CUDA_UFFT64X(LB(256,NUM_CTA_V64),i,32, 6,32,u)
CUDA_UFFT64X(LB(256,NUM_CTA_V64),i,32, 7,32,u)
CUDA_UFFT64X(LB(256,NUM_CTA_V64),i,32, 8,32,u)
CUDA_UFFT64X(LB(256,NUM_CTA_V64),i,32, 9,32,u)
CUDA_UFFT64X(LB(256,NUM_CTA_V64),i,32,10,32,u)
CUDA_UFFT64X(LB(256,NUM_CTA_V64),i,32,11,32,u)
CUDA_UFFT64X(LB(256,NUM_CTA_V64),i,32,12,32,u)
CUDA_UFFT64X(LB(256,NUM_CTA_V64),i,32,13,32,u)
CUDA_UFFT64X(LB(256,NUM_CTA_V64),i,32,14,32,u)
CUDA_UFFT64X(LB(256,NUM_CTA_V64),i,32,15,32,u)
CUDA_UFFT64X(LB(256,NUM_CTA_V64),i,32,16,32,u)

CUDA_UFFT64X(,,64,1, 2,g)
CUDA_UFFT64X(,,64,2, 4,g)
CUDA_UFFT64X(,,64,3, 8,g)
CUDA_UFFT64X(,,64,4,16,g)
CUDA_UFFT64X(LB(256,NUM_CTA_V64),,64, 5,32,u)
CUDA_UFFT64X(LB(256,NUM_CTA_V64),,64, 6,32,u)
CUDA_UFFT64X(LB(256,NUM_CTA_V64),,64, 7,32,u)
CUDA_UFFT64X(LB(256,NUM_CTA_V64),,64, 8,32,u)
CUDA_UFFT64X(LB(256,NUM_CTA_V64),,64, 9,32,u)
CUDA_UFFT64X(LB(256,NUM_CTA_V64),,64,10,32,u)
CUDA_UFFT64X(LB(256,NUM_CTA_V64),,64,11,32,u)
CUDA_UFFT64X(LB(256,NUM_CTA_V64),,64,12,32,u)
CUDA_UFFT64X(LB(256,NUM_CTA_V64),,64,13,32,u)
CUDA_UFFT64X(LB(256,NUM_CTA_V64),,64,14,32,u)
CUDA_UFFT64X(LB(256,NUM_CTA_V64),,64,15,32,u)
CUDA_UFFT64X(LB(256,NUM_CTA_V64),,64,16,32,u)

CUDA_UFFT64X(,i,64,1, 2,g)
CUDA_UFFT64X(,i,64,2, 4,g)
CUDA_UFFT64X(,i,64,3, 8,g)
CUDA_UFFT64X(,i,64,4,16,g)
CUDA_UFFT64X(LB(256,NUM_CTA_V64),i,64, 5,32,u)
CUDA_UFFT64X(LB(256,NUM_CTA_V64),i,64, 6,32,u)
CUDA_UFFT64X(LB(256,NUM_CTA_V64),i,64, 7,32,u)
CUDA_UFFT64X(LB(256,NUM_CTA_V64),i,64, 8,32,u)
CUDA_UFFT64X(LB(256,NUM_CTA_V64),i,64, 9,32,u)
CUDA_UFFT64X(LB(256,NUM_CTA_V64),i,64,10,32,u)
CUDA_UFFT64X(LB(256,NUM_CTA_V64),i,64,11,32,u)
CUDA_UFFT64X(LB(256,NUM_CTA_V64),i,64,12,32,u)
CUDA_UFFT64X(LB(256,NUM_CTA_V64),i,64,13,32,u)
CUDA_UFFT64X(LB(256,NUM_CTA_V64),i,64,14,32,u)
CUDA_UFFT64X(LB(256,NUM_CTA_V64),i,64,15,32,u)
CUDA_UFFT64X(LB(256,NUM_CTA_V64),i,64,16,32,u)

CUDA_VFFT64X(,,64,1, 2,g)
CUDA_VFFT64X(,,64,2, 4,g)
CUDA_VFFT64X(,,64,3, 8,g)
CUDA_VFFT64X(,,64,4,16,g)
CUDA_VFFT64X(LB(256,NUM_CTA_V64),,64, 5,32,u)
CUDA_VFFT64X(LB(256,NUM_CTA_V64),,64, 6,32,u)
CUDA_VFFT64X(LB(256,NUM_CTA_V64),,64, 7,32,u)
CUDA_VFFT64X(LB(256,NUM_CTA_V64),,64, 8,32,u)
CUDA_VFFT64X(LB(256,NUM_CTA_V64),,64, 9,32,u)
CUDA_VFFT64X(LB(256,NUM_CTA_V64),,64,10,32,u)
CUDA_VFFT64X(LB(256,NUM_CTA_V64),,64,11,32,u)
CUDA_VFFT64X(LB(256,NUM_CTA_V64),,64,12,32,u)
CUDA_VFFT64X(LB(256,NUM_CTA_V64),,64,13,32,u)
CUDA_VFFT64X(LB(256,NUM_CTA_V64),,64,14,32,u)
CUDA_VFFT64X(LB(256,NUM_CTA_V64),,64,15,32,u)
CUDA_VFFT64X(LB(256,NUM_CTA_V64),,64,16,32,u)

CUDA_VFFT64X(,i,64,1, 2,g)
CUDA_VFFT64X(,i,64,2, 4,g)
CUDA_VFFT64X(,i,64,3, 8,g)
CUDA_VFFT64X(,i,64,4,16,g)
CUDA_VFFT64X(LB(256,NUM_CTA_V64),i,64, 5,32,u)
CUDA_VFFT64X(LB(256,NUM_CTA_V64),i,64, 6,32,u)
CUDA_VFFT64X(LB(256,NUM_CTA_V64),i,64, 7,32,u)
CUDA_VFFT64X(LB(256,NUM_CTA_V64),i,64, 8,32,u)
CUDA_VFFT64X(LB(256,NUM_CTA_V64),i,64, 9,32,u)
CUDA_VFFT64X(LB(256,NUM_CTA_V64),i,64,10,32,u)
CUDA_VFFT64X(LB(256,NUM_CTA_V64),i,64,11,32,u)
CUDA_VFFT64X(LB(256,NUM_CTA_V64),i,64,12,32,u)
CUDA_VFFT64X(LB(256,NUM_CTA_V64),i,64,13,32,u)
CUDA_VFFT64X(LB(256,NUM_CTA_V64),i,64,14,32,u)
CUDA_VFFT64X(LB(256,NUM_CTA_V64),i,64,15,32,u)
CUDA_VFFT64X(LB(256,NUM_CTA_V64),i,64,16,32,u)

CUDA_VFFT64X(,,128,1, 2,g)
CUDA_VFFT64X(,,128,2, 4,g)
CUDA_VFFT64X(,,128,3, 8,g)
CUDA_VFFT64X(,,128,4,16,g)
CUDA_VFFT64X(LB(256,NUM_CTA_V64),,128, 5,32,u)
CUDA_VFFT64X(LB(256,NUM_CTA_V64),,128, 6,32,u)
CUDA_VFFT64X(LB(256,NUM_CTA_V64),,128, 7,32,u)
CUDA_VFFT64X(LB(256,NUM_CTA_V64),,128, 8,32,u)
CUDA_VFFT64X(LB(256,NUM_CTA_V64),,128, 9,32,u)
CUDA_VFFT64X(LB(256,NUM_CTA_V64),,128,10,32,u)
CUDA_VFFT64X(LB(256,NUM_CTA_V64),,128,11,32,u)
CUDA_VFFT64X(LB(256,NUM_CTA_V64),,128,12,32,u)
CUDA_VFFT64X(LB(256,NUM_CTA_V64),,128,13,32,u)
CUDA_VFFT64X(LB(256,NUM_CTA_V64),,128,14,32,u)
CUDA_VFFT64X(LB(256,NUM_CTA_V64),,128,15,32,u)
CUDA_VFFT64X(LB(256,NUM_CTA_V64),,128,16,32,u)

CUDA_VFFT64X(,i,128,1, 2,g)
CUDA_VFFT64X(,i,128,2, 4,g)
CUDA_VFFT64X(,i,128,3, 8,g)
CUDA_VFFT64X(,i,128,4,16,g)
CUDA_VFFT64X(LB(256,NUM_CTA_V64),i,128, 5,32,u)
CUDA_VFFT64X(LB(256,NUM_CTA_V64),i,128, 6,32,u)
CUDA_VFFT64X(LB(256,NUM_CTA_V64),i,128, 7,32,u)
CUDA_VFFT64X(LB(256,NUM_CTA_V64),i,128, 8,32,u)
CUDA_VFFT64X(LB(256,NUM_CTA_V64),i,128, 9,32,u)
CUDA_VFFT64X(LB(256,NUM_CTA_V64),i,128,10,32,u)
CUDA_VFFT64X(LB(256,NUM_CTA_V64),i,128,11,32,u)
CUDA_VFFT64X(LB(256,NUM_CTA_V64),i,128,12,32,u)
CUDA_VFFT64X(LB(256,NUM_CTA_V64),i,128,13,32,u)
CUDA_VFFT64X(LB(256,NUM_CTA_V64),i,128,14,32,u)
CUDA_VFFT64X(LB(256,NUM_CTA_V64),i,128,15,32,u)
CUDA_VFFT64X(LB(256,NUM_CTA_V64),i,128,16,32,u)
