﻿#ifndef __xfft_context_h__
#define __xfft_context_h__

#include"xfft_status.h"
#include<cuda.h>

typedef struct xfft_context{
	CUcontext ctx;
	CUmodule  module;
	CUdevice  dev;
	int       arch;
	int       alignment;
	int       max_nblk_x;
	int       max_nblk_y;
	char      padding[28];
} xfft_context_t;

int                     xfft_context_create( xfft_context_t* const );
void                    xfft_context_release( xfft_context_t* const );
__forceinline void	xfft_context_bind( xfft_context_t* const p ){ cuCtxSetCurrent(p->ctx); }
__forceinline void	xfft_context_unbind(){ cuCtxPopCurrent(NULL); }

#endif
