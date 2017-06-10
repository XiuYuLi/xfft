#ifndef __hfft_h__
#define __hfft_h__

#include"xfft.h"

#define mCALRF4(RF){                        \
    RF[1].x=RF[0].x*RF[0].x-RF[0].y*RF[0].y;\
    RF[1].y=RF[0].x*RF[0].y+RF[0].y*RF[0].x;\
    RF[2].x=RF[0].x*RF[1].x-RF[0].y*RF[1].y;\
    RF[2].y=RF[0].x*RF[1].y+RF[0].y*RF[1].x;\
}

#define mCALRF8(RF){                        \
    RF[1].x=RF[0].x*RF[0].x-RF[0].y*RF[0].y;\
    RF[1].y=RF[0].x*RF[0].y+RF[0].y*RF[0].x;\
    RF[2].x=RF[0].x*RF[1].x-RF[0].y*RF[1].y;\
    RF[2].y=RF[0].x*RF[1].y+RF[0].y*RF[1].x;\
    RF[3].x=RF[1].x*RF[1].x-RF[1].y*RF[1].y;\
    RF[3].y=RF[1].x*RF[1].y+RF[1].y*RF[1].x;\
    RF[4].x=RF[1].x*RF[2].x-RF[1].y*RF[2].y;\
    RF[4].y=RF[1].x*RF[2].y+RF[1].y*RF[2].x;\
    RF[5].x=RF[2].x*RF[2].x-RF[2].y*RF[2].y;\
    RF[5].y=RF[2].x*RF[2].y+RF[2].y*RF[2].x;\
    RF[6].x=RF[2].x*RF[3].x-RF[2].y*RF[3].y;\
    RF[6].y=RF[2].x*RF[3].y+RF[2].y*RF[3].x;\
}

#define mCALRF16(RF){                        \
    RF[ 1].x=RF[0].x*RF[0].x-RF[0].y*RF[0].y;\
    RF[ 1].y=RF[0].x*RF[0].y+RF[0].y*RF[0].x;\
    RF[ 2].x=RF[0].x*RF[1].x-RF[0].y*RF[1].y;\
    RF[ 2].y=RF[0].x*RF[1].y+RF[0].y*RF[1].x;\
    RF[ 3].x=RF[1].x*RF[1].x-RF[1].y*RF[1].y;\
    RF[ 3].y=RF[1].x*RF[1].y+RF[1].y*RF[1].x;\
    RF[ 4].x=RF[1].x*RF[2].x-RF[1].y*RF[2].y;\
    RF[ 4].y=RF[1].x*RF[2].y+RF[1].y*RF[2].x;\
    RF[ 5].x=RF[2].x*RF[2].x-RF[2].y*RF[2].y;\
    RF[ 5].y=RF[2].x*RF[2].y+RF[2].y*RF[2].x;\
    RF[ 6].x=RF[2].x*RF[3].x-RF[2].y*RF[3].y;\
    RF[ 6].y=RF[2].x*RF[3].y+RF[2].y*RF[3].x;\
    RF[ 7].x=RF[3].x*RF[3].x-RF[3].y*RF[3].y;\
    RF[ 7].y=RF[3].x*RF[3].y+RF[3].y*RF[3].x;\
    RF[ 8].x=RF[3].x*RF[4].x-RF[3].y*RF[4].y;\
    RF[ 8].y=RF[3].x*RF[4].y+RF[3].y*RF[4].x;\
    RF[ 9].x=RF[4].x*RF[4].x-RF[4].y*RF[4].y;\
    RF[ 9].y=RF[4].x*RF[4].y+RF[4].y*RF[4].x;\
    RF[10].x=RF[4].x*RF[5].x-RF[4].y*RF[5].y;\
    RF[10].y=RF[4].x*RF[5].y+RF[4].y*RF[5].x;\
    RF[11].x=RF[5].x*RF[5].x-RF[5].y*RF[5].y;\
    RF[11].y=RF[5].x*RF[5].y+RF[5].y*RF[5].x;\
    RF[12].x=RF[5].x*RF[6].x-RF[5].y*RF[6].y;\
    RF[12].y=RF[5].x*RF[6].y+RF[5].y*RF[6].x;\
    RF[13].x=RF[6].x*RF[6].x-RF[6].y*RF[6].y;\
    RF[13].y=RF[6].x*RF[6].y+RF[6].y*RF[6].x;\
    RF[14].x=RF[6].x*RF[7].x-RF[6].y*RF[7].y;\
    RF[14].y=RF[6].x*RF[7].y+RF[6].y*RF[7].x;\
}

#define mHMRF2(c,RF){           \
    (c)[1]=d_cmul(RF[0],(c)[1]);\
}

#define mHMRF4(c,RF){           \
    (c)[1]=d_cmul(RF[1],(c)[1]);\
    (c)[2]=d_cmul(RF[0],(c)[2]);\
    (c)[3]=d_cmul(RF[2],(c)[3]);\
}

#define mHMRF8(c,RF){           \
    (c)[1]=d_cmul(RF[3],(c)[1]);\
    (c)[2]=d_cmul(RF[1],(c)[2]);\
    (c)[3]=d_cmul(RF[5],(c)[3]);\
    (c)[4]=d_cmul(RF[0],(c)[4]);\
    (c)[5]=d_cmul(RF[4],(c)[5]);\
    (c)[6]=d_cmul(RF[2],(c)[6]);\
    (c)[7]=d_cmul(RF[6],(c)[7]);\
}

#define mHMRF16(c,RF){             \
    (c)[ 1]=d_cmul(RF[ 7],(c)[ 1]);\
    (c)[ 2]=d_cmul(RF[ 3],(c)[ 2]);\
    (c)[ 3]=d_cmul(RF[11],(c)[ 3]);\
    (c)[ 4]=d_cmul(RF[ 1],(c)[ 4]);\
    (c)[ 5]=d_cmul(RF[ 9],(c)[ 5]);\
    (c)[ 6]=d_cmul(RF[ 5],(c)[ 6]);\
    (c)[ 7]=d_cmul(RF[13],(c)[ 7]);\
    (c)[ 8]=d_cmul(RF[ 0],(c)[ 8]);\
    (c)[ 9]=d_cmul(RF[ 8],(c)[ 9]);\
    (c)[10]=d_cmul(RF[ 4],(c)[10]);\
    (c)[11]=d_cmul(RF[12],(c)[11]);\
    (c)[12]=d_cmul(RF[ 2],(c)[12]);\
    (c)[13]=d_cmul(RF[10],(c)[13]);\
    (c)[14]=d_cmul(RF[ 6],(c)[14]);\
    (c)[15]=d_cmul(RF[14],(c)[15]);\
}

#define mHMRF4x4(c,RF){       \
    c[ 1]=d_cmul(RF[1],c[ 1]);\
    c[ 5]=d_cmul(RF[1],c[ 5]);\
    c[ 9]=d_cmul(RF[1],c[ 9]);\
    c[13]=d_cmul(RF[1],c[13]);\
    c[ 2]=d_cmul(RF[0],c[ 2]);\
    c[ 6]=d_cmul(RF[0],c[ 6]);\
    c[10]=d_cmul(RF[0],c[10]);\
    c[14]=d_cmul(RF[0],c[14]);\
    c[ 3]=d_cmul(RF[2],c[ 3]);\
    c[ 7]=d_cmul(RF[2],c[ 7]);\
    c[11]=d_cmul(RF[2],c[11]);\
    c[15]=d_cmul(RF[2],c[15]);\
}

#endif
