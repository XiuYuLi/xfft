#include"vfft.h"

#define CUDA_VFFT32(LB,dir,e,bdx,op) CUDA_VFFT_DECL(32,LB,dir,e,PRF)            \
{                                                                               \
    __shared__ float smem[4*8*bdx];                                             \
    float2 c[8], temp;                                                          \
    size_t p=blockIdx.y*32*(1<<e)+threadIdx.y*(1<<e)+blockIdx.x*bdx+threadIdx.x;\
    d_i+=p;	d_o+=p;                                                             \
    float* sst=&smem[8*bdx*threadIdx.y+threadIdx.x];                            \
    float* sld=&smem[  bdx*threadIdx.y+threadIdx.x];                            \
    mLOAD8(c,d_i,4*(1<<e),)                                                     \
    mFFT8(c,dir)                                                                \
    mVMRF8(&d_RF[threadIdx.y<<3],dir,op)                                        \
    mPERMUTE_S8_L4x2(sst,sld,c,bdx,4*bdx,8*bdx,7)                               \
    mFFT4(&c[0],dir)                                                            \
    mFFT4(&c[4],dir)                                                            \
    mISTORE4x2(d_o,c,4*(1<<e),8*(1<<e),)                                        \
}
#define CUDA_UFFT32X(LB,dir,n,e,bdx,op) CUDA_UFFTX_DECL(32,LB,dir,n,e)                    \
{                                                                                         \
    __shared__ float smem[4*8*bdx];                                                       \
    float2 c[8], temp;                                                                    \
    unsigned int slot=blockIdx.x/((1<<e)/bdx);                                            \
    unsigned int bidx=blockIdx.x&((1<<e)/bdx-1);                                          \
    size_t p=blockIdx.y*n*32*(1<<e)+slot*(1<<e)+threadIdx.y*n*(1<<e)+bidx*bdx+threadIdx.x;\
    d_i+=p; d_o+=p;                                                                       \
    slot+=threadIdx.y*n;                                                                  \
    float* sst=&smem[8*bdx*threadIdx.y+threadIdx.x];                                      \
    float* sld=&smem[  bdx*threadIdx.y+threadIdx.x];                                      \
    mLOAD8(c,d_i,4*n*(1<<e),)                                                             \
    mFFT8(c,dir)                                                                          \
    mVMRF8(&d_RF[slot<<3],dir,op)                                                         \
    mPERMUTE_S8_L4x2(sst,sld,c,bdx,4*bdx,8*bdx,7)                                         \
    mFFT4(&c[0],dir)                                                                      \
    mFFT4(&c[4],dir)                                                                      \
    mVMRF4x2(&d_RF[n*32+((slot&(n-1))<<2)],dir,op)                                        \
    mISTORE4x2(d_o,c,4*n*(1<<e),8*n*(1<<e),)                                              \
}
#define CUDA_VFFT32X(LB,dir,n,e,bdx,op) CUDA_VFFTX_DECL(32,LB,dir,n,e,PRF)\
{                                                                         \
    __shared__ float smem[4*8*bdx];                                       \
    float2 c[8], temp;                                                    \
    unsigned int slot=blockIdx.x/((1<<e)/bdx);                            \
    unsigned int bidx=blockIdx.x&((1<<e)/bdx-1);                          \
    size_t p=blockIdx.y*32*n*(1<<e)+bidx*bdx+threadIdx.x;                 \
    d_i+=(p+slot*32*(1<<e)+threadIdx.y*(1<<e));		                      \
    d_o+=(p+slot*(1<<e)+threadIdx.y*n*(1<<e));		                      \
    float* sst=&smem[8*bdx*threadIdx.y+threadIdx.x];                      \
    float* sld=&smem[  bdx*threadIdx.y+threadIdx.x];                      \
    mLOAD8(c,d_i,4*(1<<e),)                                               \
    mFFT8(c,dir)                                                          \
    mVMRF8(&d_RF[8*n*threadIdx.y],dir,op)                                 \
    mPERMUTE_S8_L4x2(sst,sld,c,bdx,4*bdx,8*bdx,7)                         \
    mFFT4(&c[0],dir)                                                      \
    mFFT4(&c[4],dir)                                                      \
    mISTORE4x2(d_o,c,4*n*(1<<e),8*n*(1<<e),)                              \
}

#if SM==37
#define NUM_CTA_V32 8
#else
#define NUM_CTA_V32 4
#endif

CUDA_VFFT32(,,1, 2,g)
CUDA_VFFT32(,,2, 4,g)
CUDA_VFFT32(,,3, 8,g)
CUDA_VFFT32(,,4,16,g)
CUDA_VFFT32(,,5,32,u)
CUDA_VFFT32(LB(256,NUM_CTA_V32),, 6,64,u)
CUDA_VFFT32(LB(256,NUM_CTA_V32),, 7,64,u)
CUDA_VFFT32(LB(256,NUM_CTA_V32),, 8,64,u)
CUDA_VFFT32(LB(256,NUM_CTA_V32),, 9,64,u)
CUDA_VFFT32(LB(256,NUM_CTA_V32),,10,64,u)
CUDA_VFFT32(LB(256,NUM_CTA_V32),,11,64,u)
CUDA_VFFT32(LB(256,NUM_CTA_V32),,12,64,u)
CUDA_VFFT32(LB(256,NUM_CTA_V32),,13,64,u)
CUDA_VFFT32(LB(256,NUM_CTA_V32),,14,64,u)
CUDA_VFFT32(LB(256,NUM_CTA_V32),,15,64,u)
CUDA_VFFT32(LB(256,NUM_CTA_V32),,16,64,u)

CUDA_VFFT32(,i,1, 2,g)
CUDA_VFFT32(,i,2, 4,g)
CUDA_VFFT32(,i,3, 8,g)
CUDA_VFFT32(,i,4,16,g)
CUDA_VFFT32(,i,5,32,u)
CUDA_VFFT32(LB(256,NUM_CTA_V32),i, 6,64,u)
CUDA_VFFT32(LB(256,NUM_CTA_V32),i, 7,64,u)
CUDA_VFFT32(LB(256,NUM_CTA_V32),i, 8,64,u)
CUDA_VFFT32(LB(256,NUM_CTA_V32),i, 9,64,u)
CUDA_VFFT32(LB(256,NUM_CTA_V32),i,10,64,u)
CUDA_VFFT32(LB(256,NUM_CTA_V32),i,11,64,u)
CUDA_VFFT32(LB(256,NUM_CTA_V32),i,12,64,u)
CUDA_VFFT32(LB(256,NUM_CTA_V32),i,13,64,u)
CUDA_VFFT32(LB(256,NUM_CTA_V32),i,14,64,u)
CUDA_VFFT32(LB(256,NUM_CTA_V32),i,15,64,u)
CUDA_VFFT32(LB(256,NUM_CTA_V32),i,16,64,u)

CUDA_UFFT32X(,,16,1, 2,g)
CUDA_UFFT32X(,,16,2, 4,g)
CUDA_UFFT32X(,,16,3, 8,g)
CUDA_UFFT32X(,,16,4,16,g)
CUDA_UFFT32X(,,16,5,32,u)
CUDA_UFFT32X(LB(256,NUM_CTA_V32),,16, 6,64,u)
CUDA_UFFT32X(LB(256,NUM_CTA_V32),,16, 7,64,u)
CUDA_UFFT32X(LB(256,NUM_CTA_V32),,16, 8,64,u)
CUDA_UFFT32X(LB(256,NUM_CTA_V32),,16, 9,64,u)
CUDA_UFFT32X(LB(256,NUM_CTA_V32),,16,10,64,u)
CUDA_UFFT32X(LB(256,NUM_CTA_V32),,16,11,64,u)
CUDA_UFFT32X(LB(256,NUM_CTA_V32),,16,12,64,u)
CUDA_UFFT32X(LB(256,NUM_CTA_V32),,16,13,64,u)
CUDA_UFFT32X(LB(256,NUM_CTA_V32),,16,14,64,u)
CUDA_UFFT32X(LB(256,NUM_CTA_V32),,16,15,64,u)
CUDA_UFFT32X(LB(256,NUM_CTA_V32),,16,16,64,u)

CUDA_UFFT32X(,i,16,1, 2,g)
CUDA_UFFT32X(,i,16,2, 4,g)
CUDA_UFFT32X(,i,16,3, 8,g)
CUDA_UFFT32X(,i,16,4,16,g)
CUDA_UFFT32X(,i,16,5,32,u)
CUDA_UFFT32X(LB(256,NUM_CTA_V32),i,16, 6,64,u)
CUDA_UFFT32X(LB(256,NUM_CTA_V32),i,16, 7,64,u)
CUDA_UFFT32X(LB(256,NUM_CTA_V32),i,16, 8,64,u)
CUDA_UFFT32X(LB(256,NUM_CTA_V32),i,16, 9,64,u)
CUDA_UFFT32X(LB(256,NUM_CTA_V32),i,16,10,64,u)
CUDA_UFFT32X(LB(256,NUM_CTA_V32),i,16,11,64,u)
CUDA_UFFT32X(LB(256,NUM_CTA_V32),i,16,12,64,u)
CUDA_UFFT32X(LB(256,NUM_CTA_V32),i,16,13,64,u)
CUDA_UFFT32X(LB(256,NUM_CTA_V32),i,16,14,64,u)
CUDA_UFFT32X(LB(256,NUM_CTA_V32),i,16,15,64,u)
CUDA_UFFT32X(LB(256,NUM_CTA_V32),i,16,16,64,u)

CUDA_UFFT32X(,,32,1, 2,g)
CUDA_UFFT32X(,,32,2, 4,g)
CUDA_UFFT32X(,,32,3, 8,g)
CUDA_UFFT32X(,,32,4,16,g)
CUDA_UFFT32X(,,32,5,32,u)
CUDA_UFFT32X(LB(256,NUM_CTA_V32),,32, 6,64,u)
CUDA_UFFT32X(LB(256,NUM_CTA_V32),,32, 7,64,u)
CUDA_UFFT32X(LB(256,NUM_CTA_V32),,32, 8,64,u)
CUDA_UFFT32X(LB(256,NUM_CTA_V32),,32, 9,64,u)
CUDA_UFFT32X(LB(256,NUM_CTA_V32),,32,10,64,u)
CUDA_UFFT32X(LB(256,NUM_CTA_V32),,32,11,64,u)
CUDA_UFFT32X(LB(256,NUM_CTA_V32),,32,12,64,u)
CUDA_UFFT32X(LB(256,NUM_CTA_V32),,32,13,64,u)
CUDA_UFFT32X(LB(256,NUM_CTA_V32),,32,14,64,u)
CUDA_UFFT32X(LB(256,NUM_CTA_V32),,32,15,64,u)
CUDA_UFFT32X(LB(256,NUM_CTA_V32),,32,16,64,u)

CUDA_UFFT32X(,i,32,1, 2,g)
CUDA_UFFT32X(,i,32,2, 4,g)
CUDA_UFFT32X(,i,32,3, 8,g)
CUDA_UFFT32X(,i,32,4,16,g)
CUDA_UFFT32X(,i,32,5,32,u)
CUDA_UFFT32X(LB(256,NUM_CTA_V32),i,32, 6,64,u)
CUDA_UFFT32X(LB(256,NUM_CTA_V32),i,32, 7,64,u)
CUDA_UFFT32X(LB(256,NUM_CTA_V32),i,32, 8,64,u)
CUDA_UFFT32X(LB(256,NUM_CTA_V32),i,32, 9,64,u)
CUDA_UFFT32X(LB(256,NUM_CTA_V32),i,32,10,64,u)
CUDA_UFFT32X(LB(256,NUM_CTA_V32),i,32,11,64,u)
CUDA_UFFT32X(LB(256,NUM_CTA_V32),i,32,12,64,u)
CUDA_UFFT32X(LB(256,NUM_CTA_V32),i,32,13,64,u)
CUDA_UFFT32X(LB(256,NUM_CTA_V32),i,32,14,64,u)
CUDA_UFFT32X(LB(256,NUM_CTA_V32),i,32,15,64,u)
CUDA_UFFT32X(LB(256,NUM_CTA_V32),i,32,16,64,u)

CUDA_VFFT32X(,,32,1, 2,g)
CUDA_VFFT32X(,,32,2, 4,g)
CUDA_VFFT32X(,,32,3, 8,g)
CUDA_VFFT32X(,,32,4,16,g)
CUDA_VFFT32X(,,32,5,32,u)
CUDA_VFFT32X(LB(256,NUM_CTA_V32),,32, 6,64,u)
CUDA_VFFT32X(LB(256,NUM_CTA_V32),,32, 7,64,u)
CUDA_VFFT32X(LB(256,NUM_CTA_V32),,32, 8,64,u)
CUDA_VFFT32X(LB(256,NUM_CTA_V32),,32, 9,64,u)
CUDA_VFFT32X(LB(256,NUM_CTA_V32),,32,10,64,u)
CUDA_VFFT32X(LB(256,NUM_CTA_V32),,32,11,64,u)
CUDA_VFFT32X(LB(256,NUM_CTA_V32),,32,12,64,u)
CUDA_VFFT32X(LB(256,NUM_CTA_V32),,32,13,64,u)
CUDA_VFFT32X(LB(256,NUM_CTA_V32),,32,14,64,u)
CUDA_VFFT32X(LB(256,NUM_CTA_V32),,32,15,64,u)
CUDA_VFFT32X(LB(256,NUM_CTA_V32),,32,16,64,u)

CUDA_VFFT32X(,i,32,1, 2,g)
CUDA_VFFT32X(,i,32,2, 4,g)
CUDA_VFFT32X(,i,32,3, 8,g)
CUDA_VFFT32X(,i,32,4,16,g)
CUDA_VFFT32X(,i,32,5,32,u)
CUDA_VFFT32X(LB(256,NUM_CTA_V32),i,32, 6,64,u)
CUDA_VFFT32X(LB(256,NUM_CTA_V32),i,32, 7,64,u)
CUDA_VFFT32X(LB(256,NUM_CTA_V32),i,32, 8,64,u)
CUDA_VFFT32X(LB(256,NUM_CTA_V32),i,32, 9,64,u)
CUDA_VFFT32X(LB(256,NUM_CTA_V32),i,32,10,64,u)
CUDA_VFFT32X(LB(256,NUM_CTA_V32),i,32,11,64,u)
CUDA_VFFT32X(LB(256,NUM_CTA_V32),i,32,12,64,u)
CUDA_VFFT32X(LB(256,NUM_CTA_V32),i,32,13,64,u)
CUDA_VFFT32X(LB(256,NUM_CTA_V32),i,32,14,64,u)
CUDA_VFFT32X(LB(256,NUM_CTA_V32),i,32,15,64,u)
CUDA_VFFT32X(LB(256,NUM_CTA_V32),i,32,16,64,u)

CUDA_VFFT32X(,,64,1, 2,g)
CUDA_VFFT32X(,,64,2, 4,g)
CUDA_VFFT32X(,,64,3, 8,g)
CUDA_VFFT32X(,,64,4,16,g)
CUDA_VFFT32X(,,64,5,32,u)
CUDA_VFFT32X(LB(256,NUM_CTA_V32),,64, 6,64,u)
CUDA_VFFT32X(LB(256,NUM_CTA_V32),,64, 7,64,u)
CUDA_VFFT32X(LB(256,NUM_CTA_V32),,64, 8,64,u)
CUDA_VFFT32X(LB(256,NUM_CTA_V32),,64, 9,64,u)
CUDA_VFFT32X(LB(256,NUM_CTA_V32),,64,10,64,u)
CUDA_VFFT32X(LB(256,NUM_CTA_V32),,64,11,64,u)
CUDA_VFFT32X(LB(256,NUM_CTA_V32),,64,12,64,u)
CUDA_VFFT32X(LB(256,NUM_CTA_V32),,64,13,64,u)
CUDA_VFFT32X(LB(256,NUM_CTA_V32),,64,14,64,u)
CUDA_VFFT32X(LB(256,NUM_CTA_V32),,64,15,64,u)
CUDA_VFFT32X(LB(256,NUM_CTA_V32),,64,16,64,u)

CUDA_VFFT32X(,i,64,1, 2,g)
CUDA_VFFT32X(,i,64,2, 4,g)
CUDA_VFFT32X(,i,64,3, 8,g)
CUDA_VFFT32X(,i,64,4,16,g)
CUDA_VFFT32X(,i,64,5,32,u)
CUDA_VFFT32X(LB(256,NUM_CTA_V32),i,64, 6,64,u)
CUDA_VFFT32X(LB(256,NUM_CTA_V32),i,64, 7,64,u)
CUDA_VFFT32X(LB(256,NUM_CTA_V32),i,64, 8,64,u)
CUDA_VFFT32X(LB(256,NUM_CTA_V32),i,64, 9,64,u)
CUDA_VFFT32X(LB(256,NUM_CTA_V32),i,64,10,64,u)
CUDA_VFFT32X(LB(256,NUM_CTA_V32),i,64,11,64,u)
CUDA_VFFT32X(LB(256,NUM_CTA_V32),i,64,12,64,u)
CUDA_VFFT32X(LB(256,NUM_CTA_V32),i,64,13,64,u)
CUDA_VFFT32X(LB(256,NUM_CTA_V32),i,64,14,64,u)
CUDA_VFFT32X(LB(256,NUM_CTA_V32),i,64,15,64,u)
CUDA_VFFT32X(LB(256,NUM_CTA_V32),i,64,16,64,u)