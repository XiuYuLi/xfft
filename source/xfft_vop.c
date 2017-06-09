#include"../include/xfft_vop.h"

#define STR(s) #s
#define mCASE(S,R,n){ case n:		\
	S[0]=STR(d_vfft##R##x_e##n);	\
	S[1]=STR(d_vifft##R##x_e##n); 	\
	break; 				\
}
#define mCASEUV(S,u,v,n){ case n:		\
	S[0]=STR(d_ufft##u##x_x##v##e##n);	\
	S[1]=STR(d_uifft##u##x_x##v##e##n);	\
	S[2]=STR(d_vfft##v##x_x##u##e##n);	\
	S[3]=STR(d_vifft##v##x_x##u##e##n);	\
	break;					\
}

#define mMATCH(S,R,e)	switch(e){	\
	mCASE(S,R, 1)			\
	mCASE(S,R, 2)			\
	mCASE(S,R, 3)			\
	mCASE(S,R, 4)			\
	mCASE(S,R, 5)			\
	mCASE(S,R, 6)			\
	mCASE(S,R, 7)			\
	mCASE(S,R, 8)			\
	mCASE(S,R, 9)			\
	mCASE(S,R,10)			\
	mCASE(S,R,11)			\
	mCASE(S,R,12)			\
	mCASE(S,R,13)			\
	mCASE(S,R,14)			\
	mCASE(S,R,15)			\
	mCASE(S,R,16)			\
}

#define mMATCHUV(S,u,v,e)	switch(e){	\
	mCASEUV(S,u,v, 1)			\
	mCASEUV(S,u,v, 2)			\
	mCASEUV(S,u,v, 3)			\
	mCASEUV(S,u,v, 4)			\
	mCASEUV(S,u,v, 5)			\
	mCASEUV(S,u,v, 6)			\
	mCASEUV(S,u,v, 7)			\
	mCASEUV(S,u,v, 8)			\
	mCASEUV(S,u,v, 9)			\
	mCASEUV(S,u,v,10)			\
	mCASEUV(S,u,v,11)			\
	mCASEUV(S,u,v,12)			\
	mCASEUV(S,u,v,13)			\
	mCASEUV(S,u,v,14)			\
	mCASEUV(S,u,v,15)			\
	mCASEUV(S,u,v,16)			\
}

static void __vffte01_bk( xfft_kernel_t* const p, CUmodule module, int e )
{
	xfft_create_kernel( &p[0], module, "d_vfft2x"  );
	xfft_create_kernel( &p[1], module, "d_vifft2x" );
	xfft_kernel_sao( &p[0], XFFT_AM_P_P );
	xfft_kernel_sao( &p[1], XFFT_AM_P_P );
}
static void __vffte02_bk( xfft_kernel_t* const p, CUmodule module, int e )
{
	xfft_create_kernel( &p[0], module, "d_vfft4x"  );
	xfft_create_kernel( &p[1], module, "d_vifft4x" );
	xfft_kernel_sao( &p[0], XFFT_AM_P_P );
	xfft_kernel_sao( &p[1], XFFT_AM_P_P );
}
static void __vffte03_bk( xfft_kernel_t* const p, CUmodule module, int e )
{
	xfft_create_kernel( &p[0], module, "d_vfft8x"  );
	xfft_create_kernel( &p[1], module, "d_vifft8x" );
	xfft_kernel_sao( &p[0], XFFT_AM_P_P );
	xfft_kernel_sao( &p[1], XFFT_AM_P_P );
}
static void __vffte04_bk( xfft_kernel_t* const p, CUmodule module, int e )
{
	xfft_create_kernel( &p[0], module, "d_vfft16x"  );
	xfft_create_kernel( &p[1], module, "d_vifft16x" );
	xfft_kernel_sao( &p[0], XFFT_AM_P_P );
	xfft_kernel_sao( &p[1], XFFT_AM_P_P );
}
static void __vffte05_bk( xfft_kernel_t* const p, CUmodule module, int e )
{
	char* knames[2];
	mMATCH(knames,32,e)
	xfft_create_kernel( &p[0], module, knames[0] );
	xfft_create_kernel( &p[1], module, knames[1] );
	xfft_kernel_sao( &p[0], XFFT_AM_P_P_P );
	xfft_kernel_sao( &p[1], XFFT_AM_P_P_P );
}
static void __vffte06_bk( xfft_kernel_t* const p, CUmodule module, int e )
{
	char* knames[2];
	mMATCH(knames,64,e)
	xfft_create_kernel( &p[0], module, knames[0] );
	xfft_create_kernel( &p[1], module, knames[1] );
	xfft_kernel_sao( &p[0], XFFT_AM_P_P_P );
	xfft_kernel_sao( &p[1], XFFT_AM_P_P_P );
}
static void __vffte07_bk( xfft_kernel_t* const p, CUmodule module, int e )
{
	char* knames[2];
	mMATCH(knames,128,e)
	xfft_create_kernel( &p[0], module, knames[0] );
	xfft_create_kernel( &p[1], module, knames[1] );
	xfft_kernel_sao( &p[0], XFFT_AM_P_P_P );
	xfft_kernel_sao( &p[1], XFFT_AM_P_P_P );
}
static void __vffte08_bk( xfft_kernel_t* const p, CUmodule module, int e )
{
	char* knames[2];
	mMATCH(knames,256,e)
	xfft_create_kernel( &p[0], module, knames[0] );
	xfft_create_kernel( &p[1], module, knames[1] );
	xfft_kernel_sao( &p[0], XFFT_AM_P_P_P );
	xfft_kernel_sao( &p[1], XFFT_AM_P_P_P );
}

static void __vffte09_bk( xfft_kernel_t* const p, CUmodule module, int e )
{
	char* knames[4];
	mMATCHUV(knames,32,16,e)
	xfft_create_kernel( &p[0], module, knames[0] );
	xfft_create_kernel( &p[1], module, knames[1] );
	xfft_create_kernel( &p[2], module, knames[2] );
	xfft_create_kernel( &p[3], module, knames[3] );
	xfft_kernel_sao( &p[0], XFFT_AM_P_P_P );
	xfft_kernel_sao( &p[1], XFFT_AM_P_P_P );
	xfft_kernel_sao( &p[2], XFFT_AM_P_P );
	xfft_kernel_sao( &p[3], XFFT_AM_P_P );
}
static void __vffte10_bk( xfft_kernel_t* const p, CUmodule module, int e )
{
	char* knames[4];
	mMATCHUV(knames,32,32,e)
	xfft_create_kernel( &p[0], module, knames[0] );
	xfft_create_kernel( &p[1], module, knames[1] );
	xfft_create_kernel( &p[2], module, knames[2] );
	xfft_create_kernel( &p[3], module, knames[3] );
	xfft_kernel_sao( &p[0], XFFT_AM_P_P_P );
	xfft_kernel_sao( &p[1], XFFT_AM_P_P_P );
	xfft_kernel_sao( &p[2], XFFT_AM_P_P_P );
	xfft_kernel_sao( &p[3], XFFT_AM_P_P_P );
}
static void __vffte11_bk( xfft_kernel_t* const p, CUmodule module, int e )
{
	char* knames[4];
	mMATCHUV(knames,64,32,e)
	xfft_create_kernel( &p[0], module, knames[0] );
	xfft_create_kernel( &p[1], module, knames[1] );
	xfft_create_kernel( &p[2], module, knames[2] );
	xfft_create_kernel( &p[3], module, knames[3] );
	xfft_kernel_sao( &p[0], XFFT_AM_P_P_P );
	xfft_kernel_sao( &p[1], XFFT_AM_P_P_P );
	xfft_kernel_sao( &p[2], XFFT_AM_P_P_P );
	xfft_kernel_sao( &p[3], XFFT_AM_P_P_P );
}
static void __vffte12_bk( xfft_kernel_t* const p, CUmodule module, int e )
{
	char* knames[4];
	mMATCHUV(knames,64,64,e)
	xfft_create_kernel( &p[0], module, knames[0] );
	xfft_create_kernel( &p[1], module, knames[1] );
	xfft_create_kernel( &p[2], module, knames[2] );
	xfft_create_kernel( &p[3], module, knames[3] );
	xfft_kernel_sao( &p[0], XFFT_AM_P_P_P );
	xfft_kernel_sao( &p[1], XFFT_AM_P_P_P );
	xfft_kernel_sao( &p[2], XFFT_AM_P_P_P );
	xfft_kernel_sao( &p[3], XFFT_AM_P_P_P );
}
static void __vffte13_bk( xfft_kernel_t* const p, CUmodule module, int e )
{
	char* knames[4];
	mMATCHUV(knames,128,64,e)
	xfft_create_kernel( &p[0], module, knames[0] );
	xfft_create_kernel( &p[1], module, knames[1] );
	xfft_create_kernel( &p[2], module, knames[2] );
	xfft_create_kernel( &p[3], module, knames[3] );
	xfft_kernel_sao( &p[0], XFFT_AM_P_P_P );
	xfft_kernel_sao( &p[1], XFFT_AM_P_P_P );
	xfft_kernel_sao( &p[2], XFFT_AM_P_P_P );
	xfft_kernel_sao( &p[3], XFFT_AM_P_P_P );
}
static void __vffte14_bk( xfft_kernel_t* const p, CUmodule module, int e )
{
	char* knames[4];
	mMATCHUV(knames,128,128,e)
	xfft_create_kernel( &p[0], module, knames[0] );
	xfft_create_kernel( &p[1], module, knames[1] );
	xfft_create_kernel( &p[2], module, knames[2] );
	xfft_create_kernel( &p[3], module, knames[3] );
	xfft_kernel_sao( &p[0], XFFT_AM_P_P_P );
	xfft_kernel_sao( &p[1], XFFT_AM_P_P_P );
	xfft_kernel_sao( &p[2], XFFT_AM_P_P_P );
	xfft_kernel_sao( &p[3], XFFT_AM_P_P_P );
}
static void __vffte15_bk( xfft_kernel_t* const p, CUmodule module, int e )
{
	char* knames[4];
	mMATCHUV(knames,256,128,e)
	xfft_create_kernel( &p[0], module, knames[0] );
	xfft_create_kernel( &p[1], module, knames[1] );
	xfft_create_kernel( &p[2], module, knames[2] );
	xfft_create_kernel( &p[3], module, knames[3] );
	xfft_kernel_sao( &p[0], XFFT_AM_P_P_P );
	xfft_kernel_sao( &p[1], XFFT_AM_P_P_P );
	xfft_kernel_sao( &p[2], XFFT_AM_P_P_P );
	xfft_kernel_sao( &p[3], XFFT_AM_P_P_P );
}
static void __vffte16_bk( xfft_kernel_t* const p, CUmodule module, int e )
{
	char* knames[4];
	mMATCHUV(knames,256,256,e)
	xfft_create_kernel( &p[0], module, knames[0] );
	xfft_create_kernel( &p[1], module, knames[1] );
	xfft_create_kernel( &p[2], module, knames[2] );
	xfft_create_kernel( &p[3], module, knames[3] );
	xfft_kernel_sao( &p[0], XFFT_AM_P_P_P );
	xfft_kernel_sao( &p[1], XFFT_AM_P_P_P );
	xfft_kernel_sao( &p[2], XFFT_AM_P_P_P );
	xfft_kernel_sao( &p[3], XFFT_AM_P_P_P );
}
static void (*p_bk[])( xfft_kernel_t* const, CUmodule, int )=
{
	__vffte01_bk,
	__vffte02_bk, 
	__vffte03_bk, 
	__vffte04_bk, 
	__vffte05_bk, 
	__vffte06_bk, 
	__vffte07_bk, 
	__vffte08_bk, 
	__vffte09_bk,
	__vffte10_bk,
	__vffte11_bk,
	__vffte12_bk,
	__vffte13_bk,
	__vffte14_bk,
	__vffte15_bk,
	__vffte16_bk
};
static void __vffte01_sgl( xfft_kernel_t* const p, int nx, int bat )
{
	unsigned int nb, nt;
	nt=(nx<=256)?nx:256;
	nb=nx/nt;
	xfft_kernel_sgl( &p[0], nb, bat ); 
	xfft_kernel_sbl( &p[0], nt, 1	);
	xfft_kernel_sgl( &p[1], nb, bat ); 
	xfft_kernel_sbl( &p[1], nt, 1	);
}
static void __vffte02_sgl( xfft_kernel_t* const p, int nx, int bat )
{
	__vffte01_sgl( p, nx, bat );
}
static void __vffte03_sgl( xfft_kernel_t* const p, int nx, int bat )
{
	__vffte01_sgl( p, nx, bat );
}
static void __vffte04_sgl( xfft_kernel_t* const p, int nx, int bat )
{
	__vffte01_sgl( p, nx, bat );
}
static void __vffte05_sgl( xfft_kernel_t* const p, int nx, int bat )
{
	unsigned int nb, nt;
	nt=(nx<=64)?nx:64;
	nb=nx/nt;
	xfft_kernel_sgl( &p[0], nb, bat ); 
	xfft_kernel_sbl( &p[0], nt, 4	);
	xfft_kernel_sgl( &p[1], nb, bat ); 
	xfft_kernel_sbl( &p[1], nt, 4	);
}
static void __vffte06_sgl( xfft_kernel_t* const p, int nx, int bat )
{
	unsigned int nb, nt;
	nt=(nx<=32)?nx:32;
	nb=nx/nt;
	xfft_kernel_sgl( &p[0], nb, bat ); 
	xfft_kernel_sbl( &p[0], nt, 8	);
	xfft_kernel_sgl( &p[1], nb, bat ); 
	xfft_kernel_sbl( &p[1], nt, 8	);
}
static void __vffte07_sgl( xfft_kernel_t* const p, int nx, int bat )
{
	unsigned int nb, nt;
	nt=(nx<=32)?nx:32;
	nb=nx/nt;
	xfft_kernel_sgl( &p[0], nb, bat ); 
	xfft_kernel_sbl( &p[0], nt, 8	);
	xfft_kernel_sgl( &p[1], nb, bat ); 
	xfft_kernel_sbl( &p[1], nt, 8	);
}
static void __vffte08_sgl( xfft_kernel_t* const p, int nx, int bat )
{
	unsigned int nb, nt;
	nt=(nx<=32)?nx:32;
	nb=nx/nt;
	xfft_kernel_sgl( &p[0], nb, bat ); 
	xfft_kernel_sbl( &p[0], nt, 16	);
	xfft_kernel_sgl( &p[1], nb, bat ); 
	xfft_kernel_sbl( &p[1], nt, 16	);
}
static void __vffte09_sgl( xfft_kernel_t* const p, int nx, int bat )
{
	unsigned int nb, nt;
	nt=(nx<64)?nx:64;
	nb=(nx/nt)<<4;
	xfft_kernel_sgl( &p[0], nb, bat ); 
	xfft_kernel_sbl( &p[0], nt, 4	);
	xfft_kernel_sgl( &p[1], nb, bat ); 
	xfft_kernel_sbl( &p[1], nt, 4	);
	nt=(nx<256)?nx:256;
	nb=(nx/nt)<<5;
	xfft_kernel_sgl( &p[2], nb, bat );
	xfft_kernel_sbl( &p[2], nt, 1	);
	xfft_kernel_sgl( &p[3], nb, bat );
	xfft_kernel_sbl( &p[3], nt, 1	);
}
static void __vffte10_sgl( xfft_kernel_t* const p, int nx, int bat )
{
	unsigned int nb, nt, i;
	nt=(nx<64)?nx:64;
	nb=(nx/nt)<<5;
	for( i=0; i<4; ++i ){
		xfft_kernel_sgl( &p[i], nb, bat ); 
		xfft_kernel_sbl( &p[i], nt, 4	);
	}
}
static void __vffte11_sgl( xfft_kernel_t* const p, int nx, int bat )
{
	unsigned int nb, nt;
	nt=(nx<=32)?nx:32;
	nb=(nx/nt)<<5;
	xfft_kernel_sgl( &p[0], nb, bat ); 
	xfft_kernel_sbl( &p[0], nt, 8	);
	xfft_kernel_sgl( &p[1], nb, bat ); 
	xfft_kernel_sbl( &p[1], nt, 8	);
	nt=(nx<=32)?nx:64;
	nb=(nx/nt)<<6;
	xfft_kernel_sgl( &p[2], nb, bat ); 
	xfft_kernel_sbl( &p[2], nt, 4	);
	xfft_kernel_sgl( &p[3], nb, bat ); 
	xfft_kernel_sbl( &p[3], nt, 4	);
}
static void __vffte12_sgl( xfft_kernel_t* const p, int nx, int bat )
{
	unsigned int nb, nt, i;
	nt=(nx<=32)?nx:32;
	nb=(nx/nt)<<6;
	for( i=0; i<4; ++i ){
		xfft_kernel_sgl( &p[i], nb, bat ); 
		xfft_kernel_sbl( &p[i], nt, 8	);
	}
}
static void __vffte13_sgl( xfft_kernel_t* const p, int nx, int bat )
{
	unsigned int nb, nt;
	nt=(nx<=32)?nx:32;
	nb=(nx/nt)<<6;
	xfft_kernel_sgl( &p[0], nb, bat ); 
	xfft_kernel_sbl( &p[0], nt, 8	);
	xfft_kernel_sgl( &p[1], nb, bat ); 
	xfft_kernel_sbl( &p[1], nt, 8	);
	nb<<=1;
	xfft_kernel_sgl( &p[2], nb, bat ); 
	xfft_kernel_sbl( &p[2], nt, 8	);
	xfft_kernel_sgl( &p[3], nb, bat ); 
	xfft_kernel_sbl( &p[3], nt, 8	);
}
static void __vffte14_sgl( xfft_kernel_t* const p, int nx, int bat )
{
	unsigned int nb, nt, i;
	nt=(nx<=32)?nx:32;
	nb=(nx/nt)<<7;
	for( i=0; i<4; ++i ){
		xfft_kernel_sgl( &p[i], nb, bat ); 
		xfft_kernel_sbl( &p[i], nt, 8	);
	}
}
static void __vffte15_sgl( xfft_kernel_t* const p, int nx, int bat )
{
	unsigned int nb, nt;
	nt=(nx<=32)?nx:32;
	nb=(nx/nt)<<7;
	xfft_kernel_sgl( &p[0], nb, bat ); 
	xfft_kernel_sbl( &p[0], nt, 16	);
	xfft_kernel_sgl( &p[1], nb, bat ); 
	xfft_kernel_sbl( &p[1], nt, 16	);
	nb<<=1;
	xfft_kernel_sgl( &p[2], nb, bat ); 
	xfft_kernel_sbl( &p[2], nt, 8	);
	xfft_kernel_sgl( &p[3], nb, bat ); 
	xfft_kernel_sbl( &p[3], nt, 8	);
}
static void __vffte16_sgl( xfft_kernel_t* const p, int nx, int bat )
{
	unsigned int nb, nt, i;
	nt=(nx<=32)?nx:32;
	nb=(nx/nt)<<8;
	for( i=0; i<4; ++i ){
		xfft_kernel_sgl( &p[i], nb, bat ); 
		xfft_kernel_sbl( &p[i], nt, 16	);
	}
}
static void (*p_sgl[])( xfft_kernel_t* const, int, int )=
{ 
	__vffte01_sgl,
	__vffte02_sgl, 
	__vffte03_sgl, 
	__vffte04_sgl, 
	__vffte05_sgl, 
	__vffte06_sgl, 
	__vffte07_sgl, 
	__vffte08_sgl, 
	__vffte09_sgl,
	__vffte10_sgl,
	__vffte11_sgl,
	__vffte12_sgl,
	__vffte13_sgl,
	__vffte14_sgl,
	__vffte15_sgl,
	__vffte16_sgl
};

static __forceinline void vfft_bk( xfft_kernel_t* const p, CUmodule module, int eh, int ev )
{
	p_bk[ev]( p, module, eh+1 );
}
static __forceinline void vfft_sgl( xfft_kernel_t* const p, int nx, int bat, int i )
{
	p_sgl[i]( p, nx, bat );
}
void vfft_bki( xfft_kernel_t* const p, CUmodule module, CUdeviceptr d_RF, int nk, int bat, int ex, int ey )
{
	vfft_bk( p, module, ex, ey );	
	vfft_sgl( p, 1<<(1+ex), bat, ey );
	if(ey>3){
		int i=0, n=(ey<=8)?2:nk;
		do{ xfft_kernel_sep_ptr( &p[i], 2, d_RF ); }while((++i)<n);
	}
}
