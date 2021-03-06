#ifndef __xfft_calcVRF_h__
#define __xfft_calcVRF_h__

#include<math.h>
#include<vector_types.h>
#include"xfft_bop.h"

int	 xfft_get_size_HRF( int );
int  xfft_get_size_VRF( int );
void xfft_calcHRF( float2* const, int, double );
void xfft_calcVRF( float2* const, int );

#endif