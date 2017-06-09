#ifndef __xfft_platform_h__
#define __xfft_platform_h__

#include<memory.h>
#include<string.h>
#include<cuda.h>
#pragma comment( lib, "cuda.lib" )

#define MAX_DEVICES 64

typedef struct xfft_platform
{
	int			n_devices;
	int			n_sdevices;	
	int			opt_sdev_id;
	CUdevice	devices		[MAX_DEVICES];
	int			clock_rate	[MAX_DEVICES];
	int			nSM			[MAX_DEVICES];
	int			sarch		[MAX_DEVICES];
	int			slist		[MAX_DEVICES+1];
} xfft_platform_t;

int xfft_platform_init( xfft_platform_t* );

#endif