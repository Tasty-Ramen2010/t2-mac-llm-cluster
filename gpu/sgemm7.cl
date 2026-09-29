#pragma OPENCL EXTENSION cl_khr_fp16 : enable
// v7: sgemm3 structure but each 8-half fragment read as ONE 16-byte uint4 load from SLM (full dword bandwidth,
// vs 8 half-rate scalar reads), unpacked with as_half2. Double-buffered global, SIMD8, 8x8 micro-tile.
#ifndef BM
#define BM 128
#define BN 128
#define BK 16
#endif
#define MR 8
#define NR 8
__attribute__((reqd_work_group_size(BN/NR, BM/MR, 1))) __attribute__((intel_reqd_sub_group_size(8)))
__kernel void sgemm(__global const half* W, __global const half* X, __global float* Y, int rows, int toks){
    __local half As[2][BK][BM];
    __local half Bs[2][BK][BN];
    const int lx=get_local_id(0), ly=get_local_id(1);
    const int wgm=get_group_id(1)*BM, wgn=get_group_id(0)*BN;
    const int tid=ly*(BN/NR)+lx, nth=(BM/MR)*(BN/NR);
    float acc[MR][NR]; for(int i=0;i<MR;i++)for(int j=0;j<NR;j++)acc[i][j]=0.0f;
    int buf=0;
    for(int i=tid;i<BM*BK;i+=nth){int c=i/BM,r=i%BM; As[0][c][r]=vload_half(0,W+(size_t)(wgm+r)*K+c);}
    for(int i=tid;i<BN*BK;i+=nth){int c=i/BN,r=i%BN; Bs[0][c][r]=vload_half(0,X+(size_t)(wgn+r)*K+c);}
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int k0=0;k0<K;k0+=BK){
        int nb=buf^1;
        if(k0+BK<K){
            for(int i=tid;i<BM*BK;i+=nth){int c=i/BM,r=i%BM; As[nb][c][r]=vload_half(0,W+(size_t)(wgm+r)*K+k0+BK+c);}
            for(int i=tid;i<BN*BK;i+=nth){int c=i/BN,r=i%BN; Bs[nb][c][r]=vload_half(0,X+(size_t)(wgn+r)*K+k0+BK+c);}
        }
        for(int k=0;k<BK;k++){
            const uint4 ai=vload4(0,(__local const uint*)&As[buf][k][ly*MR]);
            const uint4 bi=vload4(0,(__local const uint*)&Bs[buf][k][lx*NR]);
            float a[MR],b[NR];
            const half2 a0=as_half2(ai.s0),a1=as_half2(ai.s1),a2=as_half2(ai.s2),a3=as_half2(ai.s3);
            const half2 b0=as_half2(bi.s0),b1=as_half2(bi.s1),b2=as_half2(bi.s2),b3=as_half2(bi.s3);
            a[0]=a0.s0;a[1]=a0.s1;a[2]=a1.s0;a[3]=a1.s1;a[4]=a2.s0;a[5]=a2.s1;a[6]=a3.s0;a[7]=a3.s1;
            b[0]=b0.s0;b[1]=b0.s1;b[2]=b1.s0;b[3]=b1.s1;b[4]=b2.s0;b[5]=b2.s1;b[6]=b3.s0;b[7]=b3.s1;
            for(int i=0;i<MR;i++)for(int j=0;j<NR;j++)acc[i][j]=fma(a[i],b[j],acc[i][j]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        buf=nb;
    }
    for(int i=0;i<MR;i++)for(int j=0;j<NR;j++){int r=wgm+ly*MR+i,t=wgn+lx*NR+j; if(r<rows&&t<toks)Y[(size_t)t*rows+r]=acc[i][j];}
}
