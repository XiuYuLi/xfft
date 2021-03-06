#ifndef __xfft_h__
#define __xfft_h__

#define PI 3.141592654f

#define BFLYU(a,b,c,s){         \
    temp.x=((c)/(s))*b.x+(-b.y);\
    temp.y=((c)/(s))*b.y+( b.x);\
    b.x=a.x+(-(s))*temp.x;      \
    b.y=a.y+(-(s))*temp.y;      \
    a.x+=(s)*temp.x;            \
    a.y+=(s)*temp.y;            \
}

#define iBFLYU(a,b,c,s){        \
    temp.x=((c)/(s))*b.x+( b.y);\
    temp.y=((c)/(s))*b.y+(-b.x);\
    b.x=a.x+(-(s))*temp.x;      \
    b.y=a.y+(-(s))*temp.y;      \
    a.x+=(s)*temp.x;            \
    a.y+=(s)*temp.y;            \
}

#define BFLYU10(a,b){\
    temp.x=b.x;	     \
    temp.y=b.y;	     \
    b.x=a.x-b.x;     \
    b.y=a.y-b.y;     \
    a.x+=temp.x;     \
    a.y+=temp.y;     \
}

#define BFLYU01(a,b){\
    temp.x=b.y;      \
    temp.y=b.x;      \
    b.x=a.x-temp.x;  \
    b.y=a.y+temp.y;  \
    a.x+=temp.x;     \
    a.y-=temp.y;     \
}

#define iBFLYU01(a,b){\
    temp.x=b.y;       \
    temp.y=b.x;       \
    b.x=a.x+temp.x;   \
    b.y=a.y-temp.y;   \
    a.x-=temp.x;      \
    a.y+=temp.y;      \
}

#define mFFT2(c,dir){     \
    BFLYU10((c)[0],(c)[1])\
}

#define mFFT4(c,dir){          \
    BFLYU10((c)[0],(c)[2])     \
    BFLYU10((c)[1],(c)[3])     \
    BFLYU10((c)[0],(c)[1])     \
    dir##BFLYU01((c)[2],(c)[3])\
}

#define mFFT8(c,dir){                                     \
    BFLYU10((c)[0],(c)[4])                                \
    BFLYU10((c)[1],(c)[5])                                \
    BFLYU10((c)[2],(c)[6])                                \
    BFLYU10((c)[3],(c)[7])                                \
    BFLYU10((c)[0],(c)[2])                                \
    BFLYU10((c)[1],(c)[3])                                \
    dir##BFLYU01((c)[4],(c)[6])                           \
    dir##BFLYU01((c)[5],(c)[7])                           \
    BFLYU10((c)[0],(c)[1])                                \
    dir##BFLYU01((c)[2],(c)[3])                           \
    dir##BFLYU((c)[4],(c)[5], 0.707106781f,-0.707106781f);\
    dir##BFLYU((c)[6],(c)[7],-0.707106781f,-0.707106781f);\
}

#define mFFT16(c,dir){                                     \
    BFLYU10((c)[0],(c)[ 8])                                \
    BFLYU10((c)[1],(c)[ 9])                                \
    BFLYU10((c)[2],(c)[10])                                \
    BFLYU10((c)[3],(c)[11])                                \
    BFLYU10((c)[4],(c)[12])                                \
    BFLYU10((c)[5],(c)[13])                                \
    BFLYU10((c)[6],(c)[14])                                \
    BFLYU10((c)[7],(c)[15])                                \
                                                           \
    BFLYU10((c)[0],(c)[4])                                 \
    BFLYU10((c)[1],(c)[5])                                 \
    BFLYU10((c)[2],(c)[6])                                 \
    BFLYU10((c)[3],(c)[7])                                 \
    dir##BFLYU01((c)[ 8],(c)[12])                          \
    dir##BFLYU01((c)[ 9],(c)[13])                          \
    dir##BFLYU01((c)[10],(c)[14])                          \
    dir##BFLYU01((c)[11],(c)[15])                          \
                                                           \
    BFLYU10((c)[0],(c)[2])                                 \
    BFLYU10((c)[1],(c)[3])                                 \
    dir##BFLYU01((c)[4],(c)[6])	                           \
    dir##BFLYU01((c)[5],(c)[7])	                           \
    dir##BFLYU((c)[ 8],(c)[10], 0.707106781f,-0.707106781f)\
    dir##BFLYU((c)[ 9],(c)[11], 0.707106781f,-0.707106781f)\
    dir##BFLYU((c)[12],(c)[14],-0.707106781f,-0.707106781f)\
    dir##BFLYU((c)[13],(c)[15],-0.707106781f,-0.707106781f)\
                                                           \
    BFLYU10((c)[0],(c)[1])                                 \
    dir##BFLYU01((c)[2],(c)[3])                            \
    dir##BFLYU((c)[ 4],(c)[ 5], 0.707106781f,-0.707106781f)\
    dir##BFLYU((c)[ 6],(c)[ 7],-0.707106781f,-0.707106781f)\
    dir##BFLYU((c)[ 8],(c)[ 9], 0.923879533f,-0.382683432f)\
    dir##BFLYU((c)[10],(c)[11],-0.382683432f,-0.923879533f)\
    dir##BFLYU((c)[12],(c)[13], 0.382683432f,-0.923879533f)\
    dir##BFLYU((c)[14],(c)[15],-0.923879533f,-0.382683432f)\
}

#define mFFT32(c,dir){                                 \
    BFLYU10(c[ 0],c[16])                               \
    BFLYU10(c[ 1],c[17])                               \
    BFLYU10(c[ 2],c[18])                               \
    BFLYU10(c[ 3],c[19])                               \
    BFLYU10(c[ 4],c[20])                               \
    BFLYU10(c[ 5],c[21])                               \
    BFLYU10(c[ 6],c[22])                               \
    BFLYU10(c[ 7],c[23])                               \
    BFLYU10(c[ 8],c[24])                               \
    BFLYU10(c[ 9],c[25])                               \
    BFLYU10(c[10],c[26])                               \
    BFLYU10(c[11],c[27])                               \
    BFLYU10(c[12],c[28])                               \
    BFLYU10(c[13],c[29])                               \
    BFLYU10(c[14],c[30])                               \
    BFLYU10(c[15],c[31])                               \
                                                       \
    BFLYU10(c[0],c[ 8])                                \
    BFLYU10(c[1],c[ 9])                                \
    BFLYU10(c[2],c[10])                                \
    BFLYU10(c[3],c[11])                                \
    BFLYU10(c[4],c[12])                                \
    BFLYU10(c[5],c[13])                                \
    BFLYU10(c[6],c[14])                                \
    BFLYU10(c[7],c[15])                                \
    dir##BFLYU01(c[16],c[24])                          \
    dir##BFLYU01(c[20],c[28])                          \
    dir##BFLYU01(c[18],c[26])                          \
    dir##BFLYU01(c[22],c[30])                          \
    dir##BFLYU01(c[17],c[25])                          \
    dir##BFLYU01(c[21],c[29])                          \
    dir##BFLYU01(c[19],c[27])                          \
    dir##BFLYU01(c[23],c[31])                          \
                                                       \
    BFLYU10(c[0],c[4])                                 \
    BFLYU10(c[1],c[5])                                 \
    BFLYU10(c[2],c[6])                                 \
    BFLYU10(c[3],c[7])                                 \
    dir##BFLYU01(c[ 8],c[12])                          \
    dir##BFLYU01(c[ 9],c[13])                          \
    dir##BFLYU01(c[10],c[14])                          \
    dir##BFLYU01(c[11],c[15])                          \
    dir##BFLYU(c[16],c[20], 0.707106781f,-0.707106781f)\
    dir##BFLYU(c[17],c[21], 0.707106781f,-0.707106781f)\
    dir##BFLYU(c[18],c[22], 0.707106781f,-0.707106781f)\
    dir##BFLYU(c[19],c[23], 0.707106781f,-0.707106781f)\
    dir##BFLYU(c[24],c[28],-0.707106781f,-0.707106781f)\
    dir##BFLYU(c[25],c[29],-0.707106781f,-0.707106781f)\
    dir##BFLYU(c[26],c[30],-0.707106781f,-0.707106781f)\
    dir##BFLYU(c[27],c[31],-0.707106781f,-0.707106781f)\
                                                       \
    BFLYU10(c[0],c[2])                                 \
    BFLYU10(c[1],c[3])                                 \
    dir##BFLYU01(c[4],c[6])                            \
    dir##BFLYU01(c[5],c[7])                            \
    dir##BFLYU(c[ 8],c[10], 0.707106781f,-0.707106781f)\
    dir##BFLYU(c[ 9],c[11], 0.707106781f,-0.707106781f)\
    dir##BFLYU(c[12],c[14],-0.707106781f,-0.707106781f)\
    dir##BFLYU(c[13],c[15],-0.707106781f,-0.707106781f)\
    dir##BFLYU(c[16],c[18], 0.923879533f,-0.382683432f)\
    dir##BFLYU(c[17],c[19], 0.923879533f,-0.382683432f)\
    dir##BFLYU(c[20],c[22],-0.382683432f,-0.923879533f)\
    dir##BFLYU(c[21],c[23],-0.382683432f,-0.923879533f)\
    dir##BFLYU(c[24],c[26], 0.382683432f,-0.923879533f)\
    dir##BFLYU(c[25],c[27], 0.382683432f,-0.923879533f)\
    dir##BFLYU(c[28],c[30],-0.923879533f,-0.382683432f)\
    dir##BFLYU(c[29],c[31],-0.923879533f,-0.382683432f)\
                                                       \
    BFLYU10(c[0],c[1])                                 \
    dir##BFLYU01(c[2],c[3])                            \
    dir##BFLYU(c[ 4],c[ 5], 0.707106781f,-0.707106781f)\
    dir##BFLYU(c[ 6],c[ 7],-0.707106781f,-0.707106781f)\
    dir##BFLYU(c[ 8],c[ 9], 0.923879533f,-0.382683432f)\
    dir##BFLYU(c[10],c[11],-0.382683432f,-0.923879533f)\
    dir##BFLYU(c[12],c[13], 0.382683432f,-0.923879533f)\
    dir##BFLYU(c[14],c[15],-0.923879533f,-0.382683432f)\
    dir##BFLYU(c[16],c[17], 0.980785280f,-0.195090322f)\
    dir##BFLYU(c[18],c[19],-0.195090322f,-0.980785280f)\
    dir##BFLYU(c[20],c[21], 0.555570233f,-0.831469612f)\
    dir##BFLYU(c[22],c[23],-0.831469612f,-0.555570233f)\
    dir##BFLYU(c[24],c[25], 0.831469612f,-0.555570233f)\
    dir##BFLYU(c[26],c[27],-0.555570233f,-0.831469612f)\
    dir##BFLYU(c[28],c[29], 0.195090322f,-0.980785280f)\
    dir##BFLYU(c[30],c[31],-0.980785280f,-0.195090322f)\
}

#define mLOAD2(d,s,n,e){	\
	(d)[0]##e=*((s)+0*(n));	\
	(d)[1]##e=*((s)+1*(n));	\
}

#define mLOAD4(d,s,n,e){	\
	(d)[0]##e=*((s)+0*(n));	\
	(d)[1]##e=*((s)+1*(n));	\
	(d)[2]##e=*((s)+2*(n));	\
	(d)[3]##e=*((s)+3*(n));	\
}

#define mLOAD8(d,s,n,e){	\
	(d)[0]##e=*((s)+0*(n));	\
	(d)[1]##e=*((s)+1*(n));	\
	(d)[2]##e=*((s)+2*(n));	\
	(d)[3]##e=*((s)+3*(n));	\
	(d)[4]##e=*((s)+4*(n));	\
	(d)[5]##e=*((s)+5*(n));	\
	(d)[6]##e=*((s)+6*(n));	\
	(d)[7]##e=*((s)+7*(n));	\
}

#define mLOAD4x2(d,s,m,n,e){   \
    (d)[0]##e=*((s)    +0*(n));\
    (d)[1]##e=*((s)    +1*(n));\
    (d)[2]##e=*((s)    +2*(n));\
    (d)[3]##e=*((s)    +3*(n));\
    (d)[4]##e=*((s)+(m)+0*(n));\
    (d)[5]##e=*((s)+(m)+1*(n));\
    (d)[6]##e=*((s)+(m)+2*(n));\
    (d)[7]##e=*((s)+(m)+3*(n));\
}

#define mLOAD16(d,s,n,e){    \
    (d)[ 0]##e=*((s)+ 0*(n));\
    (d)[ 1]##e=*((s)+ 1*(n));\
    (d)[ 2]##e=*((s)+ 2*(n));\
    (d)[ 3]##e=*((s)+ 3*(n));\
    (d)[ 4]##e=*((s)+ 4*(n));\
    (d)[ 5]##e=*((s)+ 5*(n));\
    (d)[ 6]##e=*((s)+ 6*(n));\
    (d)[ 7]##e=*((s)+ 7*(n));\
    (d)[ 8]##e=*((s)+ 8*(n));\
    (d)[ 9]##e=*((s)+ 9*(n));\
    (d)[10]##e=*((s)+10*(n));\
    (d)[11]##e=*((s)+11*(n));\
    (d)[12]##e=*((s)+12*(n));\
    (d)[13]##e=*((s)+13*(n));\
    (d)[14]##e=*((s)+14*(n));\
    (d)[15]##e=*((s)+15*(n));\
}

#define mLOAD4x4(d,s,m,n,e){      \
    (d)[ 0]##e=*((s)+0*(m)+0*(n));\
    (d)[ 1]##e=*((s)+0*(m)+1*(n));\
    (d)[ 2]##e=*((s)+0*(m)+2*(n));\
    (d)[ 3]##e=*((s)+0*(m)+3*(n));\
    (d)[ 4]##e=*((s)+1*(m)+0*(n));\
    (d)[ 5]##e=*((s)+1*(m)+1*(n));\
    (d)[ 6]##e=*((s)+1*(m)+2*(n));\
    (d)[ 7]##e=*((s)+1*(m)+3*(n));\
    (d)[ 8]##e=*((s)+2*(m)+0*(n));\
    (d)[ 9]##e=*((s)+2*(m)+1*(n));\
    (d)[10]##e=*((s)+2*(m)+2*(n));\
    (d)[11]##e=*((s)+2*(m)+3*(n));\
    (d)[12]##e=*((s)+3*(m)+0*(n));\
    (d)[13]##e=*((s)+3*(m)+1*(n));\
    (d)[14]##e=*((s)+3*(m)+2*(n));\
    (d)[15]##e=*((s)+3*(m)+3*(n));\
}

#define mLOAD8x2(d,s,m,n,e){    \
    (d)[ 0]##e=*((s)    +0*(n));\
    (d)[ 1]##e=*((s)    +1*(n));\
    (d)[ 2]##e=*((s)    +2*(n));\
    (d)[ 3]##e=*((s)    +3*(n));\
    (d)[ 4]##e=*((s)    +4*(n));\
    (d)[ 5]##e=*((s)    +5*(n));\
    (d)[ 6]##e=*((s)    +6*(n));\
    (d)[ 7]##e=*((s)    +7*(n));\
    (d)[ 8]##e=*((s)+(m)+0*(n));\
    (d)[ 9]##e=*((s)+(m)+1*(n));\
    (d)[10]##e=*((s)+(m)+2*(n));\
    (d)[11]##e=*((s)+(m)+3*(n));\
    (d)[12]##e=*((s)+(m)+4*(n));\
    (d)[13]##e=*((s)+(m)+5*(n));\
    (d)[14]##e=*((s)+(m)+6*(n));\
    (d)[15]##e=*((s)+(m)+7*(n));\
}

#define mSTORE2(d,s,n,e){  \
    *((d)+0*(n))=(s)[0]##e;\
    *((d)+1*(n))=(s)[1]##e;\
}

#define mISTORE2 mSTORE2

#define mISTORE4(d,s,n,e){ \
    *((d)+0*(n))=(s)[0]##e;\
    *((d)+1*(n))=(s)[2]##e;\
    *((d)+2*(n))=(s)[1]##e;\
    *((d)+3*(n))=(s)[3]##e;\
}

#define mISTORE8(d,s,n,e){ \
    *((d)+0*(n))=(s)[0]##e;\
    *((d)+1*(n))=(s)[4]##e;\
    *((d)+2*(n))=(s)[2]##e;\
    *((d)+3*(n))=(s)[6]##e;\
    *((d)+4*(n))=(s)[1]##e;\
    *((d)+5*(n))=(s)[5]##e;\
    *((d)+6*(n))=(s)[3]##e;\
    *((d)+7*(n))=(s)[7]##e;\
}

#define mISTORE16(d,s,n,e){  \
    *((d)+ 0*(n))=(s)[ 0]##e;\
    *((d)+ 1*(n))=(s)[ 8]##e;\
    *((d)+ 2*(n))=(s)[ 4]##e;\
    *((d)+ 3*(n))=(s)[12]##e;\
    *((d)+ 4*(n))=(s)[ 2]##e;\
    *((d)+ 5*(n))=(s)[10]##e;\
    *((d)+ 6*(n))=(s)[ 6]##e;\
    *((d)+ 7*(n))=(s)[14]##e;\
    *((d)+ 8*(n))=(s)[ 1]##e;\
    *((d)+ 9*(n))=(s)[ 9]##e;\
    *((d)+10*(n))=(s)[ 5]##e;\
    *((d)+11*(n))=(s)[13]##e;\
    *((d)+12*(n))=(s)[ 3]##e;\
    *((d)+13*(n))=(s)[11]##e;\
    *((d)+14*(n))=(s)[ 7]##e;\
    *((d)+15*(n))=(s)[15]##e;\
}

#define mISTORE4x4(d,s,m,n,e){  \
    *((d)+0*m+0*(n))=(s)[ 0]##e;\
    *((d)+0*m+1*(n))=(s)[ 2]##e;\
    *((d)+0*m+2*(n))=(s)[ 1]##e;\
    *((d)+0*m+3*(n))=(s)[ 3]##e;\
    *((d)+1*m+0*(n))=(s)[ 4]##e;\
    *((d)+1*m+1*(n))=(s)[ 6]##e;\
    *((d)+1*m+2*(n))=(s)[ 5]##e;\
    *((d)+1*m+3*(n))=(s)[ 7]##e;\
    *((d)+2*m+0*(n))=(s)[ 8]##e;\
    *((d)+2*m+1*(n))=(s)[10]##e;\
    *((d)+2*m+2*(n))=(s)[ 9]##e;\
    *((d)+2*m+3*(n))=(s)[11]##e;\
    *((d)+3*m+0*(n))=(s)[12]##e;\
    *((d)+3*m+1*(n))=(s)[14]##e;\
    *((d)+3*m+2*(n))=(s)[13]##e;\
    *((d)+3*m+3*(n))=(s)[15]##e;\
}

#define mISTORE4x2(d,s,m,n,e){ \
    *((d)    +0*(n))=(s)[0]##e;\
    *((d)    +1*(n))=(s)[2]##e;\
    *((d)    +2*(n))=(s)[1]##e;\
    *((d)    +3*(n))=(s)[3]##e;\
    *((d)+(m)+0*(n))=(s)[4]##e;\
    *((d)+(m)+1*(n))=(s)[6]##e;\
    *((d)+(m)+2*(n))=(s)[5]##e;\
    *((d)+(m)+3*(n))=(s)[7]##e;\
}

#define mISTORE8x2(d,s,m,n,e){  \
    *((d)    +0*(n))=(s)[ 0]##e;\
    *((d)    +1*(n))=(s)[ 4]##e;\
    *((d)    +2*(n))=(s)[ 2]##e;\
    *((d)    +3*(n))=(s)[ 6]##e;\
    *((d)    +4*(n))=(s)[ 1]##e;\
    *((d)    +5*(n))=(s)[ 5]##e;\
    *((d)    +6*(n))=(s)[ 3]##e;\
    *((d)    +7*(n))=(s)[ 7]##e;\
    *((d)+(m)+0*(n))=(s)[ 8]##e;\
    *((d)+(m)+1*(n))=(s)[12]##e;\
    *((d)+(m)+2*(n))=(s)[10]##e;\
    *((d)+(m)+3*(n))=(s)[14]##e;\
    *((d)+(m)+4*(n))=(s)[ 9]##e;\
    *((d)+(m)+5*(n))=(s)[13]##e;\
    *((d)+(m)+6*(n))=(s)[11]##e;\
    *((d)+(m)+7*(n))=(s)[15]##e;\
}

#define mSTORE4(d,s,n,e){  \
    *((d)+0*(n))=(s)[0]##e;\
    *((d)+1*(n))=(s)[1]##e;\
    *((d)+2*(n))=(s)[2]##e;\
    *((d)+3*(n))=(s)[3]##e;\
}

#define mSTORE4x2(d,s,m,n){ \
    *((d)    +0*(n))=(s)[0];\
    *((d)    +1*(n))=(s)[1];\
    *((d)    +2*(n))=(s)[2];\
    *((d)    +3*(n))=(s)[3];\
    *((d)+(m)+0*(n))=(s)[4];\
    *((d)+(m)+1*(n))=(s)[5];\
    *((d)+(m)+2*(n))=(s)[6];\
    *((d)+(m)+3*(n))=(s)[7];\
}

#define mSTORE8(d,s,n,e){  \
    *((d)+0*(n))=(s)[0]##e;\
    *((d)+1*(n))=(s)[1]##e;\
    *((d)+2*(n))=(s)[2]##e;\
    *((d)+3*(n))=(s)[3]##e;\
    *((d)+4*(n))=(s)[4]##e;\
    *((d)+5*(n))=(s)[5]##e;\
    *((d)+6*(n))=(s)[6]##e;\
    *((d)+7*(n))=(s)[7]##e;\
}

#define mSTORE8x2(d,s,m,n,e){   \
    *((d)    +0*(n))=(s)[ 0]##e;\
    *((d)    +1*(n))=(s)[ 1]##e;\
    *((d)    +2*(n))=(s)[ 2]##e;\
    *((d)    +3*(n))=(s)[ 3]##e;\
    *((d)    +4*(n))=(s)[ 4]##e;\
    *((d)    +5*(n))=(s)[ 5]##e;\
    *((d)    +6*(n))=(s)[ 6]##e;\
    *((d)    +7*(n))=(s)[ 7]##e;\
    *((d)+(m)+0*(n))=(s)[ 8]##e;\
    *((d)+(m)+1*(n))=(s)[ 9]##e;\
    *((d)+(m)+2*(n))=(s)[10]##e;\
    *((d)+(m)+3*(n))=(s)[11]##e;\
    *((d)+(m)+4*(n))=(s)[12]##e;\
    *((d)+(m)+5*(n))=(s)[13]##e;\
    *((d)+(m)+6*(n))=(s)[14]##e;\
    *((d)+(m)+7*(n))=(s)[15]##e;\
}

#define mSTORE16(d,s,n,e){   \
    *((d)+ 0*(n))=(s)[ 0]##e;\
    *((d)+ 1*(n))=(s)[ 1]##e;\
    *((d)+ 2*(n))=(s)[ 2]##e;\
    *((d)+ 3*(n))=(s)[ 3]##e;\
    *((d)+ 4*(n))=(s)[ 4]##e;\
    *((d)+ 5*(n))=(s)[ 5]##e;\
    *((d)+ 6*(n))=(s)[ 6]##e;\
    *((d)+ 7*(n))=(s)[ 7]##e;\
    *((d)+ 8*(n))=(s)[ 8]##e;\
    *((d)+ 9*(n))=(s)[ 9]##e;\
    *((d)+10*(n))=(s)[10]##e;\
    *((d)+11*(n))=(s)[11]##e;\
    *((d)+12*(n))=(s)[12]##e;\
    *((d)+13*(n))=(s)[13]##e;\
    *((d)+14*(n))=(s)[14]##e;\
    *((d)+15*(n))=(s)[15]##e;\
}

#define mPERMUTE(R,sst,sld,c,m,n,mask){                    \
    mISTORE##R(sst,c,m,.x) if(mask&0x1){ __syncthreads(); }\
    mLOAD##R(c,sld,n,.x)   if(mask&0x2){ __syncthreads(); }\
    mISTORE##R(sst,c,m,.y) if(mask&0x4){ __syncthreads(); }\
    mLOAD##R(c,sld,n,.y)   if(mask&0x8){ __syncthreads(); }\
}

#define mPERMUTE4x4(sst,sld,c,k,m,n,mask){                 \
    mISTORE16(sst,c,k,.x)  if(mask&0x1){ __syncthreads(); }\
    mLOAD4x4(c,sld,m,n,.x) if(mask&0x2){ __syncthreads(); }\
    mISTORE16(sst,c,k,.y)  if(mask&0x4){ __syncthreads(); }\
    mLOAD4x4(c,sld,m,n,.y) if(mask&0x8){ __syncthreads(); }\
}

#define mIPERMUTE4x4(sst,sld,c,m,n,k,mask){                  \
    mISTORE4x4(sst,c,m,n,.x) if(mask&0x1){ __syncthreads(); }\
    mLOAD16(c,sld,k,.x)      if(mask&0x1){ __syncthreads(); }\
    mISTORE4x4(sst,c,m,n,.y) if(mask&0x1){ __syncthreads(); }\
    mLOAD16(c,sld,k,.y)      if(mask&0x2){ __syncthreads(); }\
}

#define mPERMUTE_S8_L4x2(sst,sld,c,k,m,n,mask){            \
    mISTORE8(sst,c,k,.x)   if(mask&0x1){ __syncthreads(); }\
    mLOAD4x2(c,sld,m,n,.x) if(mask&0x2){ __syncthreads(); }\
    mISTORE8(sst,c,k,.y)   if(mask&0x4){ __syncthreads(); }\
    mLOAD4x2(c,sld,m,n,.y) if(mask&0x8){ __syncthreads(); }\
}

#define mPERMUTE_S4x2_L8(sst,sld,c,m,n,k,mask){              \
    mISTORE4x2(sst,c,m,n,.x) if(mask&0x1){ __syncthreads(); }\
    mLOAD8(c,sld,k,.x)       if(mask&0x2){ __syncthreads(); }\
    mISTORE4x2(sst,c,m,n,.y) if(mask&0x4){ __syncthreads(); }\
    mLOAD8(c,sld,k,.y)       if(mask&0x8){ __syncthreads(); }\
}

#define mPERMUTE4x2(sst,sld,c,a,b,m,n,mask){                 \
    mISTORE4x2(sst,c,a,b,.x) if(mask&0x1){ __syncthreads(); }\
    mLOAD4x2(c,sld,m,n,.x)   if(mask&0x2){ __syncthreads(); }\
    mISTORE4x2(sst,c,a,b,.y) if(mask&0x4){ __syncthreads(); }\
    mLOAD4x2(c,sld,m,n,.y)   if(mask&0x8){ __syncthreads(); }\
}

#define mPERMUTE_S16_L8x2(sst,sld,c,k,m,n,mask){           \
    mISTORE16(sst,c,k,.x)  if(mask&0x1){ __syncthreads(); }\
    mLOAD8x2(c,sld,m,n,.x) if(mask&0x2){ __syncthreads(); }\
    mISTORE16(sst,c,k,.y)  if(mask&0x4){ __syncthreads(); }\
    mLOAD8x2(c,sld,m,n,.y) if(mask&0x8){ __syncthreads(); }\
}

#define mPERMUTE_S8x2_L16(sst,sld,c,m,n,k,mask){             \
    mISTORE8x2(sst,c,m,n,.x) if(mask&0x1){ __syncthreads(); }\
    mLOAD16(c,sld,k,.x)      if(mask&0x2){ __syncthreads(); }\
    mISTORE8x2(sst,c,m,n,.y) if(mask&0x4){ __syncthreads(); }\
    mLOAD16(c,sld,k,.y)      if(mask&0x8){ __syncthreads(); }\
}

#define mPERMUTE8x2(sst,sld,c,a,b,m,n,mask){                 \
    mISTORE8x2(sst,c,a,b,.x) if(mask&0x1){ __syncthreads(); }\
    mLOAD8x2(c,sld,m,n,.x)   if(mask&0x2){ __syncthreads(); }\
    mISTORE8x2(sst,c,a,b,.y) if(mask&0x4){ __syncthreads(); }\
    mLOAD8x2(c,sld,m,n,.y)   if(mask&0x8){ __syncthreads(); }\
}

#define LB(nt,nb) __launch_bounds__(nt,nb)

__device__ __forceinline__ float2 d_cmul( const float2& a, const float2& b )
{
	return make_float2(a.x*b.x-a.y*b.y,a.x*b.y+b.x*a.y);
}
__device__ __forceinline__ float2 d_icmul( const float2& a, const float2& b )
{
	return make_float2(a.x*b.x+a.y*b.y,a.x*b.y-b.x*a.y);
}
__device__ __forceinline__ void d_cmulx2( float2& a, float2& b, const float4& c )
{
	float2 temp;
	temp.x=c.x*a.x-c.y*a.y;
	temp.y=c.x*a.y+c.y*a.x;
	a.x=temp.x; a.y=temp.y;
	temp.x=c.z*b.x-c.w*b.y;
	temp.y=c.z*b.y+c.w*b.x;
	b.x=temp.x; b.y=temp.y;
}
__device__ __forceinline__ void d_icmulx2( float2& a, float2& b, const float4& c )
{
	float2 temp;
	temp.x=c.x*a.x+c.y*a.y;
	temp.y=c.x*a.y-c.y*a.x;
	a.x=temp.x; a.y=temp.y;
	temp.x=c.z*b.x+c.w*b.y;
	temp.y=c.z*b.y-c.w*b.x;
	b.x=temp.x; b.y=temp.y;
}

__device__ __forceinline__ void d_cmul2x2( float2& a0, float2& b0, float2& a1, float2& b1, const float4& c )
{
	float2 temp;
	temp.x=c.x*a0.x-c.y*a0.y;
	temp.y=c.x*a0.y+c.y*a0.x;
	a0.x=temp.x; a0.y=temp.y;	
	temp.x=c.z*b0.x-c.w*b0.y;
	temp.y=c.z*b0.y+c.w*b0.x;
	b0.x=temp.x; b0.y=temp.y;
	temp.x=c.x*a1.x-c.y*a1.y;
	temp.y=c.x*a1.y+c.y*a1.x;
	a1.x=temp.x; a1.y=temp.y;
	temp.x=c.z*b1.x-c.w*b1.y;
	temp.y=c.z*b1.y+c.w*b1.x;
	b1.x=temp.x; b1.y=temp.y;
}
__device__ __forceinline__ void d_icmul2x2( float2& a0, float2& b0, float2& a1, float2& b1, const float4& c )
{
	float2 temp;
	temp.x=c.x*a0.x+c.y*a0.y;
	temp.y=c.x*a0.y-c.y*a0.x;
	a0.x=temp.x; a0.y=temp.y;	
	temp.x=c.z*b0.x+c.w*b0.y;
	temp.y=c.z*b0.y-c.w*b0.x;
	b0.x=temp.x; b0.y=temp.y;
	temp.x=c.x*a1.x+c.y*a1.y;
	temp.y=c.x*a1.y-c.y*a1.x;
	a1.x=temp.x; a1.y=temp.y;
	temp.x=c.z*b1.x+c.w*b1.y;
	temp.y=c.z*b1.y-c.w*b1.x;
	b1.x=temp.x; b1.y=temp.y;
}

#define mTLD8(d,o,n){                \
    (d)[0]=tex1Dfetch(d_tex,o+0*(n));\
    (d)[1]=tex1Dfetch(d_tex,o+1*(n));\
    (d)[2]=tex1Dfetch(d_tex,o+2*(n));\
    (d)[3]=tex1Dfetch(d_tex,o+3*(n));\
    (d)[4]=tex1Dfetch(d_tex,o+4*(n));\
    (d)[5]=tex1Dfetch(d_tex,o+5*(n));\
    (d)[6]=tex1Dfetch(d_tex,o+6*(n));\
    (d)[7]=tex1Dfetch(d_tex,o+7*(n));\
}

#define mH2Sx4(s,h){                \
    (s)[0].x=__half2float((h)[0].x);\
    (s)[0].y=__half2float((h)[0].y);\
    (s)[1].x=__half2float((h)[1].x);\
    (s)[1].y=__half2float((h)[1].y);\
    (s)[2].x=__half2float((h)[2].x);\
    (s)[2].y=__half2float((h)[2].y);\
    (s)[3].x=__half2float((h)[3].x);\
    (s)[3].y=__half2float((h)[3].y);\
}

#define mH2Sx8(s,h){                \
    (s)[0].x=__half2float((h)[0].x);\
    (s)[0].y=__half2float((h)[0].y);\
    (s)[1].x=__half2float((h)[1].x);\
    (s)[1].y=__half2float((h)[1].y);\
    (s)[2].x=__half2float((h)[2].x);\
    (s)[2].y=__half2float((h)[2].y);\
    (s)[3].x=__half2float((h)[3].x);\
    (s)[3].y=__half2float((h)[3].y);\
    (s)[4].x=__half2float((h)[4].x);\
    (s)[4].y=__half2float((h)[4].y);\
    (s)[5].x=__half2float((h)[5].x);\
    (s)[5].y=__half2float((h)[5].y);\
    (s)[6].x=__half2float((h)[6].x);\
    (s)[6].y=__half2float((h)[6].y);\
    (s)[7].x=__half2float((h)[7].x);\
    (s)[7].y=__half2float((h)[7].y);\
}

#define mH2Sx16(s,h){                 \
    (s)[ 0].x=__half2float((h)[ 0].x);\
    (s)[ 0].y=__half2float((h)[ 0].y);\
    (s)[ 1].x=__half2float((h)[ 1].x);\
    (s)[ 1].y=__half2float((h)[ 1].y);\
    (s)[ 2].x=__half2float((h)[ 2].x);\
    (s)[ 2].y=__half2float((h)[ 2].y);\
    (s)[ 3].x=__half2float((h)[ 3].x);\
    (s)[ 3].y=__half2float((h)[ 3].y);\
    (s)[ 4].x=__half2float((h)[ 4].x);\
    (s)[ 4].y=__half2float((h)[ 4].y);\
    (s)[ 5].x=__half2float((h)[ 5].x);\
    (s)[ 5].y=__half2float((h)[ 5].y);\
    (s)[ 6].x=__half2float((h)[ 6].x);\
    (s)[ 6].y=__half2float((h)[ 6].y);\
    (s)[ 7].x=__half2float((h)[ 7].x);\
    (s)[ 7].y=__half2float((h)[ 7].y);\
    (s)[ 8].x=__half2float((h)[ 8].x);\
    (s)[ 8].y=__half2float((h)[ 8].y);\
    (s)[ 9].x=__half2float((h)[ 9].x);\
    (s)[ 9].y=__half2float((h)[ 9].y);\
    (s)[10].x=__half2float((h)[10].x);\
    (s)[10].y=__half2float((h)[10].y);\
    (s)[11].x=__half2float((h)[11].x);\
    (s)[11].y=__half2float((h)[11].y);\
    (s)[12].x=__half2float((h)[12].x);\
    (s)[12].y=__half2float((h)[12].y);\
    (s)[13].x=__half2float((h)[13].x);\
    (s)[13].y=__half2float((h)[13].y);\
    (s)[14].x=__half2float((h)[14].x);\
    (s)[14].y=__half2float((h)[14].y);\
    (s)[15].x=__half2float((h)[15].x);\
    (s)[15].y=__half2float((h)[15].y);\
}

#define mS2Hx4(h,s){                   \
    (h)[0].x=__float2half_rn((s)[0].x);\
    (h)[0].y=__float2half_rn((s)[0].y);\
    (h)[1].x=__float2half_rn((s)[1].x);\
    (h)[1].y=__float2half_rn((s)[1].y);\
    (h)[2].x=__float2half_rn((s)[2].x);\
    (h)[2].y=__float2half_rn((s)[2].y);\
    (h)[3].x=__float2half_rn((s)[3].x);\
    (h)[3].y=__float2half_rn((s)[3].y);\
}

#define mS2Hx8(h,s){                   \
    (h)[0].x=__float2half_rn((s)[0].x);\
    (h)[0].y=__float2half_rn((s)[0].y);\
    (h)[1].x=__float2half_rn((s)[1].x);\
    (h)[1].y=__float2half_rn((s)[1].y);\
    (h)[2].x=__float2half_rn((s)[2].x);\
    (h)[2].y=__float2half_rn((s)[2].y);\
    (h)[3].x=__float2half_rn((s)[3].x);\
    (h)[3].y=__float2half_rn((s)[3].y);\
    (h)[4].x=__float2half_rn((s)[4].x);\
    (h)[4].y=__float2half_rn((s)[4].y);\
    (h)[5].x=__float2half_rn((s)[5].x);\
    (h)[5].y=__float2half_rn((s)[5].y);\
    (h)[6].x=__float2half_rn((s)[6].x);\
    (h)[6].y=__float2half_rn((s)[6].y);\
    (h)[7].x=__float2half_rn((s)[7].x);\
    (h)[7].y=__float2half_rn((s)[7].y);\
}

#define mS2Hx16(h,s){                    \
    (h)[ 0].x=__float2half_rn((s)[ 0].x);\
    (h)[ 0].y=__float2half_rn((s)[ 0].y);\
    (h)[ 1].x=__float2half_rn((s)[ 1].x);\
    (h)[ 1].y=__float2half_rn((s)[ 1].y);\
    (h)[ 2].x=__float2half_rn((s)[ 2].x);\
    (h)[ 2].y=__float2half_rn((s)[ 2].y);\
    (h)[ 3].x=__float2half_rn((s)[ 3].x);\
    (h)[ 3].y=__float2half_rn((s)[ 3].y);\
    (h)[ 4].x=__float2half_rn((s)[ 4].x);\
    (h)[ 4].y=__float2half_rn((s)[ 4].y);\
    (h)[ 5].x=__float2half_rn((s)[ 5].x);\
    (h)[ 5].y=__float2half_rn((s)[ 5].y);\
    (h)[ 6].x=__float2half_rn((s)[ 6].x);\
    (h)[ 6].y=__float2half_rn((s)[ 6].y);\
    (h)[ 7].x=__float2half_rn((s)[ 7].x);\
    (h)[ 7].y=__float2half_rn((s)[ 7].y);\
    (h)[ 8].x=__float2half_rn((s)[ 8].x);\
    (h)[ 8].y=__float2half_rn((s)[ 8].y);\
    (h)[ 9].x=__float2half_rn((s)[ 9].x);\
    (h)[ 9].y=__float2half_rn((s)[ 9].y);\
    (h)[10].x=__float2half_rn((s)[10].x);\
    (h)[10].y=__float2half_rn((s)[10].y);\
    (h)[11].x=__float2half_rn((s)[11].x);\
    (h)[11].y=__float2half_rn((s)[11].y);\
    (h)[12].x=__float2half_rn((s)[12].x);\
    (h)[12].y=__float2half_rn((s)[12].y);\
    (h)[13].x=__float2half_rn((s)[13].x);\
    (h)[13].y=__float2half_rn((s)[13].y);\
    (h)[14].x=__float2half_rn((s)[14].x);\
    (h)[14].y=__float2half_rn((s)[14].y);\
    (h)[15].x=__float2half_rn((s)[15].x);\
    (h)[15].y=__float2half_rn((s)[15].y);\
}


#endif