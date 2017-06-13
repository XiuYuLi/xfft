#ifndef __xfft_kernel_h__
#define __xfft_kernel_h__

#include<cuda.h>
#include"xfft_macro.h"

#define XFFT_AM_P		 0x0
#define XFFT_AM_P_P		 0x1
#define XFFT_AM_P_P_P		(0x2|XFFT_AM_P_P  )
#define XFFT_AM_P_P_I		(0x4|XFFT_AM_P_P  )
#define XFFT_AM_P_P_P_I 	(0x8|XFFT_AM_P_P_P)
#define XFFT_AM_P_I		 0x10

typedef struct xfft_kernel{
	CUfunction	id;	
	unsigned int	gdx;
	unsigned int	gdy;
	unsigned int	bdx;
	unsigned int	bdy;
	unsigned int	smemnb;
	unsigned int	arg_size;
	void*			extra[5];
	char			args[32];
	unsigned int 	arg_ofs[4];
	char			padding[48];
} xfft_kernel_t;

__forceinline void xfft_create_kernel( xfft_kernel_t* const p_kernel, CUmodule module, const char* p_name )
{
	cuModuleGetFunction( &p_kernel->id, module, p_name );
	p_kernel->smemnb=0;
	p_kernel->extra[0]=(void*)CU_LAUNCH_PARAM_BUFFER_POINTER;
	p_kernel->extra[1]=(void*)p_kernel->args;
	p_kernel->extra[2]=(void*)CU_LAUNCH_PARAM_BUFFER_SIZE;
	p_kernel->extra[3]=(void*)&p_kernel->arg_size;
	p_kernel->extra[4]=(void*)CU_LAUNCH_PARAM_END;
}
__forceinline void xfft_kernel_sao( xfft_kernel_t* const p, unsigned int mask )
{	
	unsigned int ofs=0;
	ofs=nAFFI(ofs,__alignof(CUdeviceptr)); p->arg_ofs[0]=ofs; ofs+=sizeof(CUdeviceptr);
	if(mask&0x1)
	{
		ofs=nAFFI(ofs,__alignof(CUdeviceptr)); p->arg_ofs[1]=ofs; ofs+=sizeof(CUdeviceptr); 
		if(mask&0x2){
			ofs=nAFFI(ofs,__alignof(CUdeviceptr)); p->arg_ofs[2]=ofs; ofs+=sizeof(CUdeviceptr);
		}
		if(mask&0x4){
			ofs=nAFFI(ofs,__alignof(int)); p->arg_ofs[2]=ofs; ofs+=sizeof(int);
		}
		if(mask&0x8){
			ofs=nAFFI(ofs,__alignof(int)); p->arg_ofs[3]=ofs; ofs+=sizeof(int);
		}
	} else
	if(mask&0x10){
		ofs=nAFFI(ofs,__alignof(int)); p->arg_ofs[1]=ofs; ofs+=sizeof(int);
	}
	p->arg_size=ofs;
}
__forceinline void xfft_kernel_sgl( xfft_kernel_t* const p_kernel, unsigned int gdx, unsigned int gdy )
{
	p_kernel->gdx=gdx; p_kernel->gdy=gdy;
}
__forceinline void xfft_kernel_sbl( xfft_kernel_t* const p_kernel, unsigned int bdx, unsigned int bdy )
{
	p_kernel->bdx=bdx; p_kernel->bdy=bdy;
}
__forceinline void xfft_kernel_sep_ptr( xfft_kernel_t* const p_kernel, int i, CUdeviceptr p )
{
	*((CUdeviceptr*)&p_kernel->args[p_kernel->arg_ofs[i]])=p; 
}
__forceinline void xfft_kernel_sep_i32( xfft_kernel_t* const p_kernel, int i, int p )
{
	*((int*)&p_kernel->args[p_kernel->arg_ofs[i]])=p; 
}
__forceinline void xfft_kernel_sep_f32( xfft_kernel_t* const p_kernel, int i, float p )
{
	*((float*)&p_kernel->args[p_kernel->arg_ofs[i]])=p;
}
__forceinline void xfft_kernel_launch( xfft_kernel_t* const p, CUstream s )
{
	cuLaunchKernel( p->id, p->gdx, p->gdy, 1, p->bdx, p->bdy, 1, p->smemnb, s, NULL, p->extra );
}

#endif
