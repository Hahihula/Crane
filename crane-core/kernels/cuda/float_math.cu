#include <math.h>
extern "C" __global__ void scalar_power_f32(
    const float* exponent, float* output, unsigned int n, float base) {
    unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) output[i]=powf(base,exponent[i]);
}

// Four independent register accumulators and descending warp reduction,
// preserving a separate F32 square before summation (no contracted FMA).
extern "C" __global__ void rms_norm_f32_ordered(
    const float* x, const float* alpha, float* output,
    unsigned int rows, unsigned int width, float eps) {
    unsigned int lane=threadIdx.x%32;
    unsigned int row=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    if(row>=rows) return;
    float sums[4]={0.f,0.f,0.f,0.f};
    for(unsigned int col=lane*4;col<width;col+=128) {
        #pragma unroll
        for(int j=0;j<4;++j) {
            float v=x[row*width+col+j];
            sums[j]=__fadd_rn(sums[j],__fmul_rn(v,v));
        }
    }
    float sum=__fadd_rn(__fadd_rn(__fadd_rn(sums[0],sums[1]),sums[2]),sums[3]);
    for(int offset=16;offset>0;offset/=2)
        sum=__fadd_rn(sum,__shfl_down_sync(0xffffffff,sum,offset));
    sum=__shfl_sync(0xffffffff,sum,0);
    float inv=rsqrtf(__fadd_rn(sum/width,eps));
    for(unsigned int col=lane;col<width;col+=32) {
        float value=__fmul_rn(x[row*width+col],inv);
        output[row*width+col]=alpha ? __fmul_rn(value,alpha[col]) : value;
    }
}

// Sklansky scan with a carry between segments, matching the small-row
// scan order used by the original CUDA framework.
extern "C" __global__ void prefix_scan_f32(
    const float* x,float* output,unsigned int rows,unsigned int width,
    unsigned int log_threads) {
    extern __shared__ float memory[];
    unsigned int tx=blockDim.x;
    unsigned int lane=threadIdx.x;
    unsigned int row=blockIdx.x*blockDim.y+threadIdx.y;
    float* values=memory+threadIdx.y*2*tx;
    float carry=0.f;
    for(unsigned int base=0;base<width;base+=2*tx) {
        unsigned int i=base+lane,j=i+tx;
        values[lane]=(row<rows && i<width)?x[row*width+i]:0.f;
        values[lane+tx]=(row<rows && j<width)?x[row*width+j]:0.f;
        if(lane==0) values[0]=__fadd_rn(values[0],carry);
        __syncthreads();
        for(unsigned int level=0;level<=log_threads;++level) {
            unsigned int size=1u<<level;
            unsigned int start=((lane>>level)<<(level+1))|size;
            unsigned int target=start+(lane%size);
            values[target]=__fadd_rn(values[target],values[start-1]);
            __syncthreads();
        }
        if(row<rows) {
            if(i<width) output[row*width+i]=values[lane];
            if(j<width) output[row*width+j]=values[lane+tx];
        }
        carry=values[2*tx-1];
        __syncthreads();
    }
}
