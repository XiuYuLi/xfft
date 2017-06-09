__device__ __forceinline__ void d_postproc( float2& a, float2& b, const float2& RF )
{
	float hax=0.5f*a.x;
	float hay=0.5f*a.y;
    float p0=( 0.5f)*b.x+hax;
    float p1=(-0.5f)*b.y+hay;
    float q0=( 0.5f)*b.y+hay;
    float q1=(-0.5f)*b.x+hax;
    a.x=p0+q0*RF.x+q1*RF.y;
    a.y=p1+q0*RF.y-q1*RF.x;
    b.x=p0-q0*RF.x-q1*RF.y;
    b.y=q0*RF.y-q1*RF.x-p1;
}

__device__ __forceinline__ void d_preproc( float2& a, float2& b, const float2& RF )
{
    float p0=a.x+b.x;
    float p1=a.y-b.y;
    float q0=a.y+b.y;
    float q1=a.x-b.x;
    a.x=p0-q0*RF.x+q1*RF.y;
    a.y=p1+q1*RF.x+q0*RF.y;
    b.x=q0*RF.x-q1*RF.y+p0;
    b.y=q1*RF.x+q0*RF.y-p1;
}


__device__ __forceinline__ void d_postproc( float2& a, float2& b, const float2& RF )
{
	float hax=0.5f*a.x;
	float hay=0.5f*a.y;
    float p0=( 0.5f)*b.x+hax;
    float p1=(-0.5f)*b.y+hay;
    float q0=( 0.5f)*b.y+hay;
    float q1=(-0.5f)*b.x+hax;
    a.x=__fmaf_rn( q0, RF.x, __fmaf_rn( q1, RF.y, p0));
    a.y=__fmaf_rn( q0, RF.y, __fmaf_rn(-q1, RF.x, p1));
    b.x=__fmaf_rn(-q0, RF.x, __fmaf_rn(-q1, RF.y, p0));
    b.y=__fmaf_rn( q0, RF.y, __fmaf_rn(-q1, RF.x,-p1));
}
__device__ __forceinline__ void d_preproc( float2& a, float2& b, const float2& RF )
{
    float p0=a.x+b.x;
    float p1=a.y-b.y;
    float q0=a.y+b.y;
    float q1=a.x-b.x;
    a.x=__fmaf_rn(-q0, RF.x, __fmaf_rn( q1, RF.y, p0));
    a.y=__fmaf_rn( q1, RF.x, __fmaf_rn( q0, RF.y, p1));
    b.x=__fmaf_rn( q0, RF.x, __fmaf_rn(-q1, RF.y, p0));
    b.y=__fmaf_rn( q1, RF.x, __fmaf_rn( q0, RF.y,-p1));
}

__device__ __forceinline__ float2 d_cmul( const float2& a, const float2& b, float c, float s )
{
	float ay=s*a.y;
	return make_float2(c*(a.x*b.x-ay*b.y),c*(a.x*b.y+ay*b.x));
}

__global__ void d_postpreacc( float2* d_out, const float2 *d_dat, const float2 *d_ker, const float2* d_RF, unsigned int nx, unsigned int ny, unsigned int nc, float scale, float sign )
{
	unsigned int i, o, x, y, h, u, v, p, q;
	float2 a0, a1, b0, b1, c0, c1, s0,s1, s2, s3, RF;

    i=blockIdx.x*blockDim.x+threadIdx.x;
	d_out+=blockIdx.y*nx*ny;
	d_ker+=blockIdx.y*nx*ny*nc;
	h=ny>>1;
	x=i&(nx-1);
	y=i>>(__ffs(nx)-1);
	u=y*nx+x;
	v=(y?(ny-y):y)*nx+(x?(nx-x):x);
	if(y==0){
		p=h*nx+x;
	    q=h*nx+(x?(nx-x):x);
	}
	s0=make_float2(0.f,0.f);
	s1=make_float2(0.f,0.f);
	s2=make_float2(0.f,0.f);
	s3=make_float2(0.f,0.f);
	RF=d_RF[x];

	for( i=0; i<nc; ++i )
	{
		a0=d_dat[u];
		a1=d_dat[v];
		b0=d_ker[u];
		b1=d_ker[v];
		d_postproc(a0,a1,RF);
		d_postproc(b0,b1,RF);
		c0=d_cmul(a0,b0,scale,sign);
		c1=d_cmul(a1,b1,scale,sign);
		d_preproc(c0,c1,RF);
		s0+=c0; s1+=c1;

		if(y==0)
		{
			a0=d_dat[p];
			a1=d_dat[q];
			b0=d_ker[p];
			b1=d_ker[q];
			d_postproc(a0,a1,RF);
			d_postproc(b0,b1,RF);
			c0=d_cmul(a0,b0,scale,sign);
			c1=d_cmul(a1,b1,scale,sign);
			d_preproc(c0,c1,RF);
			s2+=c0; s3+=c1;
		}
	}
	d_out[u]=s0;
	d_out[v]=s1;
	if(y==0){
		d_out[p]=s2;
		d_out[q]=s3;
	}
}

__global__ void d_padding( float* d_dst, const float* d_src, int dat_nx, int dat_ny, int fft_nx, int fft_ny, int pitch_src )
{
	unsigned int i, u, v;	
	const unsigned int bdx=blockDim.x;
	const unsigned int pitch_dst=fft_nx*fft_ny;	
	const unsigned int s=__ffs(fft_nx)-1;
	d_dst+=blockIdx.y*pitch_dst;
	d_src+=blockIdx.y*pitch_src;	
	i=blockIdx.x*bdx+threadIdx.x;
	while(i<pitch_dst){
		u=i&(fft_nx-1);
		v=i>>s;
		d_dst[i]=((u<dat_nx)&(v<dat_ny))?d_src[v*dat_nx+u]:0.f;
		i+=bdx;
	}
}
__global__ void d_padding( float* d_dst, const float* d_src, int dat_nx, int dat_ny, int fft_nx, int fft_ny, int pitch_src )
{
	unsigned int i, u, v;	
	const unsigned int bdx=blockDim.x;
	const unsigned int pitch_dst=fft_nx*fft_ny;	
	const unsigned int s=__ffs(fft_nx)-1;
	const unsigned int dx=fft_nx-dat_nx;
	const unsigned int dy=fft_ny-dat_ny;
	d_dst+=blockIdx.y*pitch_dst;
	d_src+=blockIdx.y*pitch_src;	
	i=blockIdx.x*bdx+threadIdx.x;
	while(i<pitch_dst){
		u=i&(fft_nx-1);
		v=i>>s;
		d_dst[i]=((u>=dx)&(v>=dy))?d_src[(v-dy)*dat_nx+u-dx]:0.f;
		i+=bdx;
	}
}
__global__ void d_sacc( float4* d_dst, const float4* d_src, int n, int nc )
{
	float4 s, v;
	int i, c;
	const int nt=gridDim.x*blockDim.x;
	i=blockIdx.x*blockDim.x+threadIdx.x;
	d_dst+=blockIdx.y*n;
	d_src+=blockIdx.y*nc*n;
	do{
		for( s=make_float4(0.f,0.f,0.f,0.f), c=0; c<nc; ++c ){ 
			v=d_src[c*n+i]; 
			s.x+=v.x;
			s.y+=v.y;
			s.z+=v.z;
			s.w+=v.w;
		} 
		d_dst[i]=s;
	}while((i+=nt)<n);
}
__global__ void d_sflip( float* d_dst, const float* d_src, int dat_nx, int dat_ny, int fft_nx, int fft_ny, int pitch_dst )
{
	int x, y, z, bdx, pitch_src;
	x=threadIdx.x;
	y=blockIdx.y*blockDim.y+threadIdx.y;
	bdx=blockDim.x;
	if(y>=dat_ny) return;
	pitch_src=fft_nx*fft_ny;
	z=(y!=0)?(fft_ny-y):0;	
	d_dst+=blockIdx.x*pitch_dst+y*dat_nx;
	d_src+=blockIdx.x*pitch_src+z*fft_nx;
	while(x<dat_nx){
		d_dst[x]=d_src[(x!=0)?(fft_nx-x):0];
		x+=bdx;
	}
}

__global__ void d_sflip( float* d_dst, const float* d_src, int dat_nx, int dat_ny, int fft_nx, int fft_ny, int pitch_dst )
{
	int x, y, z, bdx, pitch_src;
	x=threadIdx.x;
	y=blockIdx.y*blockDim.y+threadIdx.y;
	bdx=blockDim.x;
	if(y>=dat_ny) return;
	pitch_src=fft_nx*fft_ny;
	z=dat_ny-y-1;	
	d_dst+=blockIdx.x*pitch_dst+y*dat_nx;
	d_src+=blockIdx.x*pitch_src+z*fft_nx;
	while(x<dat_nx){
		d_dst[x]=d_src[dat_nx-x-1];
		x+=bdx;
	}
}

__global__ void d_get_patches( float* d_dst, const float* d_src, int dat_nx, int dat_ny, int ker_nx, int ker_ny, int patch_nx, int patch_ny, int npx, int npy, int pitch_src )
{
	unsigned int ib, u, v, ox, oy, tx, ty;
	ib=blockIdx.x*blockDim.y+threadIdx.y;
	if(ib>=(npx*npy)) return;
	u=ib%npx;
	v=ib/npx;
	ox=patch_nx-ker_nx+1;
	oy=patch_ny-ker_ny+1;
	d_dst+=(blockIdx.y*npx*npy+ib)*patch_nx*patch_ny+threadIdx.x;
	d_src+=blockIdx.y*pitch_src+v*oy*dat_nx+u*ox+threadIdx.x;

	if((u<npx-1)&(v<npy-1)){
		for( int i=0; i<patch_ny; ++i ){
			d_dst[i*patch_nx]=d_src[i*dat_nx];
		}
	}
	else
	{			
		tx=u*ox+threadIdx.x;
		ty=v*oy;
		for( int i=0; i<patch_ny; ++i ){
			float e=0.f;
			if((tx<dat_nx)&((ty+i)<dat_ny)){
				e=d_src[i*dat_nx];
			}
			d_dst[i*patch_nx]=e;
		}
	}
}
