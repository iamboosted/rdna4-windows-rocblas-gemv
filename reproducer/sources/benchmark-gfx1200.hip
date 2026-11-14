#define NOMINMAX
#include <Windows.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
constexpr size_t MiB = 1024*1024;
constexpr int GUARD = 32, CALLS = 8;
static uint64_t exact_checks = 0, numeric_checks = 0, guard_checks = 0;
static double peak_allocation_mib = 0;
static unsigned cleanup_errors = 0;
static void cleanup(hipError_t s) { if(s!=hipSuccess) ++cleanup_errors; }
static void need(bool c,const char * s) { if(!c) throw std::runtime_error(s); }
static void ck(hipError_t s) { if(s!=hipSuccess) throw std::runtime_error(hipGetErrorString(s)); }
static void rb(rocblas_status s) { need(s==rocblas_status_success,"rocBLAS call failed"); }
static uint32_t bits(float f) { uint32_t u; std::memcpy(&u,&f,4); return u; }
static uint32_t random_u32(uint32_t &s) { s^=s<<13; s^=s>>17; s^=s<<5; return s; }
static double elapsed(Clock::time_point t) { return std::chrono::duration<double,std::milli>(Clock::now()-t).count(); }

static void adjacent(const wchar_t * name) {
    wchar_t exe[32768]{},dll[32768]{};
    auto h=GetModuleHandleW(name); need(h!=nullptr,"Expected runtime module not loaded");
    auto a=GetModuleFileNameW(nullptr,exe,32768),b=GetModuleFileNameW(h,dll,32768);
    need(a&&a<32768&&b&&b<32768,"Cannot check runtime location");
    std::wstring expected(exe,a); auto slash=expected.find_last_of(L"\\/");
    need(slash!=std::wstring::npos,"Cannot locate executable directory");
    expected.resize(slash+1); expected+=name;
    need(_wcsicmp(expected.c_str(),dll)==0,"Runtime did not load from isolated executable directory");
}

struct Memory {
    std::vector<void*> owned;
    size_t bytes=0;
    hipStream_t stream=nullptr;
    Memory() { need(cleanup_errors==0,"Earlier HIP cleanup failed"); ck(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking)); }
    template<class T> T * alloc(size_t n) {
        need(n>0 && n<=640*MiB/sizeof(T),"Allocation dimension bound");
        size_t want=n*sizeof(T),free=0,total=0;
        need(bytes+want<=640*MiB,"Owned allocation exceeds 640 MiB");
        ck(hipMemGetInfo(&free,&total));
        need(free>=want+1024*MiB,"Insufficient immediate VRAM headroom");
        T *p=nullptr; ck(hipMalloc(&p,want)); owned.push_back(p); bytes+=want;
        peak_allocation_mib=std::max(peak_allocation_mib,double(bytes)/MiB); return p;
    }
    ~Memory() { if(stream) cleanup(hipStreamSynchronize(stream)); for(auto p:owned) cleanup(hipFree(p)); if(stream) cleanup(hipStreamDestroy(stream)); }
};
static __global__ void marker(unsigned long long *t,int i) { t[i]=(unsigned long long)wall_clock64(); }
static __global__ void bounded_delay(unsigned long long *t,unsigned long long ticks) {
    auto a=(unsigned long long)wall_clock64(),b=a;
    while(b-a<ticks) b=(unsigned long long)wall_clock64();
    t[0]=a;t[1]=b;
}
static __global__ void noop() {}

struct Graph {
    hipGraph_t graph=nullptr;
    hipGraphExec_t exec=nullptr;
    hipStream_t stream;
    unsigned long long *ticks;
    std::string name;
    Graph(Memory &m,const char *label,const std::function<void(int)> &launch):stream(m.stream),name(label) {
        ticks=m.alloc<unsigned long long>(2);
        ck(hipStreamBeginCapture(stream,hipStreamCaptureModeThreadLocal));
        marker<<<1,1,0,stream>>>(ticks,0);
        for(int i=0;i<CALLS;++i) launch(i);
        marker<<<1,1,0,stream>>>(ticks,1);
        ck(hipGetLastError());ck(hipStreamEndCapture(stream,&graph));
        ck(hipGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    }
    std::array<double,2> run() {
        unsigned long long t[2]{};
        ck(hipStreamSynchronize(stream));auto start=Clock::now();
        ck(hipGraphLaunch(exec,stream));ck(hipStreamSynchronize(stream));double host=elapsed(start);
        ck(hipMemcpy(t,ticks,sizeof(t),hipMemcpyDeviceToHost));
        need(t[1]>t[0],"Nonpositive device-clock span");
        double gpu=double(t[1]-t[0])/100000.0;
        need(gpu<=host+0.005,"Device-clock span exceeds enclosing host interval");
        return {gpu*1000/CALLS,host*1000/CALLS};
    }
    ~Graph() { if(exec) cleanup(hipGraphExecDestroy(exec)); if(graph) cleanup(hipGraphDestroy(graph)); }
};

static void time_graphs(const char *suite,int M,int K,const char *layout,std::vector<std::unique_ptr<Graph>> &g) {
    for(auto &p:g) for(int i=0;i<2;++i) p->run();
    for(int trial=0;trial<6;++trial) {
        for(size_t j=0;j<g.size();++j) {
            size_t i=trial%2?g.size()-1-j:j;
            auto t=g[i]->run();
            std::printf("{\"kind\":\"timing\",\"suite\":\"%s\",\"M\":%d,\"K\":%d,\"N\":1,\"layout\":\"%s\",\"variant\":\"%s\",\"trial\":%d,\"calls\":%d,\"gpu_us_per_call\":%.9f,\"host_us_per_call\":%.9f}\n",suite,M,K,layout,g[i]->name.c_str(),trial,CALLS,t[0],t[1]);
        }
    }
}

static void calibration() {
    Memory m;auto t=m.alloc<unsigned long long>(2);hipEvent_t a,b;
    ck(hipEventCreate(&a));ck(hipEventCreate(&b));
    for(int us:{100,500,1000}) {
        unsigned long long h[2]{};auto begin=Clock::now();
        ck(hipEventRecord(a,m.stream));bounded_delay<<<1,1,0,m.stream>>>(t,(unsigned long long)us*100);
        ck(hipGetLastError());ck(hipEventRecord(b,m.stream));ck(hipEventSynchronize(b));double host=elapsed(begin);
        float event=0;ck(hipEventElapsedTime(&event,a,b));ck(hipMemcpy(h,t,sizeof(h),hipMemcpyDeviceToHost));
        need(h[1]>h[0],"Calibration clock ordering failed");
        double ms=double(h[1]-h[0])/100000.0;
        need(ms>=us/1000.0 && ms<us/1000.0+0.1 && ms<=host+0.005,"Clock calibration failed");
        bool event_ok=std::isfinite(event)&&event>=ms*0.8&&event<=ms*1.2;
        std::printf("{\"kind\":\"calibration\",\"requested_us\":%d,\"device_ms\":%.9f,\"host_ms\":%.9f,\"hip_event_ms\":%.9f,\"event_agrees\":%s}\n",us,ms,host,std::isfinite(event)?event:0,event_ok?"true":"false");
    }
    ck(hipEventDestroy(a));ck(hipEventDestroy(b));
    std::vector<std::unique_ptr<Graph>> g;
    g.emplace_back(new Graph(m,"empty_launch",[&](int){noop<<<1,1,0,m.stream>>>();}));
    time_graphs("runtime",0,0,"none",g);
}

struct Q8W { half d; int8_t q[32]; };
struct Q8X { half2 ds; int8_t q[32]; };
static_assert(sizeof(Q8W)==34 && alignof(Q8W)==2);
static_assert(sizeof(Q8X)==36 && alignof(Q8X)==4);
static __device__ __forceinline__ int load4(const int8_t *p,int i) {
    auto q=reinterpret_cast<const uint16_t*>(p);return int(unsigned(q[2*i])|(unsigned(q[2*i+1])<<16));
}
static __device__ __forceinline__ int dot4(int a,int b,int c) {
#if defined(__HIP_DEVICE_COMPILE__)
    return __builtin_amdgcn_sudot4(true,a,true,b,c,false);
#else
    return c;
#endif
}

// Synthetic reconstruction of the retained Q8_0/Q8_1 lane and reduction arithmetic.
// No model weights, GGUF parsing, runtime interception, or external patch is used.
template<int K,int W,int R> static __global__ __launch_bounds__(32*W*R)
void q8(const Q8W *a,const Q8X *x,const float *bias,float *out,int M) {
    constexpr int B=K/32;
    unsigned lane=threadIdx.x,wave=threadIdx.y%W,r=threadIdx.y/W,row=blockIdx.x*R+r;
    float sum=0;
    for(unsigned kb=wave*8+lane/4;kb<B;kb+=W*8) {
        if(row<M) {
            const auto &v=a[size_t(row)*B+kb];const auto &y=x[kb];int dot=0;
#pragma unroll
            for(int i=0;i<2;++i) {
                int p=int(lane%4)*2+i;
                dot=dot4(load4(v.q,p),reinterpret_cast<const int*>(y.q)[p],dot);
            }
            sum+=__half2float(v.d)*__half2float(__low2half(y.ds))*float(dot);
        }
    }
    if constexpr(W>1) {
        __shared__ float partial[R][W][32];
        partial[r][wave][lane]=sum;__syncthreads();
        if(wave!=0) return;
        sum=partial[r][0][lane];
#pragma unroll
        for(int w=1;w<W;++w) sum+=partial[r][w][lane];
    }
    // Retain additions of otherwise-empty wave sums for the selected K range.
#pragma unroll
    for(int w=W;w<8;++w) sum+=0.0f;
#pragma unroll
    for(int d=16;d;d>>=1) sum+=__shfl_xor(sum,d,32);
    if(lane==0 && row<M) out[row]=sum+(bias?bias[row]:0.0f);
}
template<int K> static void qlaunch(int v,int M,const Q8W *a,const Q8X *x,const float *b,float *y,hipStream_t s) {
    constexpr int W=(K+255)/256;
    if(v==0) q8<K,8,1><<<M,dim3(32,8),0,s>>>(a,x,b,y,M);
    if(v==1) q8<K,W,1><<<M,dim3(32,W),0,s>>>(a,x,b,y,M);
    if(v==2) q8<K,W,2><<<(M+1)/2,dim3(32,W*2),0,s>>>(a,x,b,y,M);
    if(v==3) q8<K,W,4><<<(M+3)/4,dim3(32,W*4),0,s>>>(a,x,b,y,M);
    if(v==4) q8<K,W,8><<<(M+7)/8,dim3(32,W*8),0,s>>>(a,x,b,y,M);
    ck(hipGetLastError());
}
static std::vector<float> read_output(float *p,int M) {
    std::vector<float> v(M+2*GUARD);ck(hipMemcpy(v.data(),p,v.size()*4,hipMemcpyDeviceToHost));
    for(int i=0;i<GUARD;++i) {
        need(bits(v[i])==0xa5a5a5a5u && bits(v[M+GUARD+i])==0xa5a5a5a5u,"Output guard changed");guard_checks+=2;
    }
    for(int i=0;i<M;++i) need(std::isfinite(v[i+GUARD]),"Nonfinite output");
    return v;
}
template<int K> static void quant_case(int M) {
    need(M>=1&&M<=248144,"Q8 row bound");constexpr int B=K/32;
    Memory m;size_t weights=size_t(M)*B;
    std::vector<Q8W> a(weights*2);std::vector<Q8X> x(B);std::vector<float> bias(M);
    uint32_t rng=2046+K+M;
    for(auto &v:a) {v.d=__float2half(std::ldexp(1.0f,-5-int(random_u32(rng)%4)));for(auto &q:v.q) q=int8_t(int(random_u32(rng)%256)-128);}
    for(auto &v:x) {v.ds=__floats2half2_rn(std::ldexp(1.0f,-5-int(random_u32(rng)%4)),0);for(auto &q:v.q) q=int8_t(int(random_u32(rng)%256)-128);}
    for(auto &v:bias) v=float(int(random_u32(rng)%257)-128)/128;
    auto da=m.alloc<Q8W>(a.size());auto dx=m.alloc<Q8X>(B);auto db=m.alloc<float>(M);
    std::array<float*,5> out{};for(auto &p:out) p=m.alloc<float>(M+2*GUARD);
    ck(hipMemcpyAsync(da,a.data(),a.size()*sizeof(Q8W),hipMemcpyHostToDevice,m.stream));
    ck(hipMemcpyAsync(dx,x.data(),x.size()*sizeof(Q8X),hipMemcpyHostToDevice,m.stream));
    ck(hipMemcpyAsync(db,bias.data(),M*4,hipMemcpyHostToDevice,m.stream));
    auto validate=[&](int mode,int rot) {
        std::vector<float> original;
        for(int v=0;v<5;++v) {
            ck(hipMemsetAsync(out[v],0xa5,(M+2*GUARD)*4,m.stream));float *dst=out[v]+GUARD;
            if(mode==2) ck(hipMemcpyAsync(dst,bias.data(),M*4,hipMemcpyHostToDevice,m.stream));
            qlaunch<K>(v,M,da+rot*weights,dx,mode==2?dst:mode==1?db:nullptr,dst,m.stream);
            ck(hipStreamSynchronize(m.stream));auto values=read_output(out[v],M);
            if(v==0) {
                original=values;
                for(int j=0;j<std::min(M,64);++j) {
                    int row=j<32?j:M-1-(j-32);double expected=mode?bias[row]:0;
                    for(int k=0;k<K;++k) {
                        auto &w=a[rot*weights+size_t(row)*B+k/32];auto &y=x[k/32];
                        expected+=double(w.q[k%32])*y.q[k%32]*__half2float(w.d)*__half2float(__low2half(y.ds));
                    }
                    if(std::fabs(values[row+GUARD]-expected)>0.0002+0.00002*std::fabs(expected)) {
                        std::fprintf(stderr,"Synthetic Q8 mismatch K=%d M=%d mode=%d rotation=%d row=%d GPU=%.9g CPU=%.17g first_weight_scale=%.9g first_input_scale=%.9g\n",K,M,mode,rot,row,values[row+GUARD],expected,__half2float(a[rot*weights+size_t(row)*B].d),__half2float(__low2half(x[0].ds)));
                        throw std::runtime_error("Q8 independent CPU reference failed");
                    }
                    ++numeric_checks;
                }
            } else {
                for(int row=0;row<M;++row) {need(bits(values[row+GUARD])==bits(original[row+GUARD]),"Q8 baseline/candidate bit mismatch");++exact_checks;}
            }
        }
    };
    validate(0,0);validate(1,1);validate(2,0);
    const char *names[]={"eight_waves_one_row","useful_waves_one_row","useful_waves_two_rows","useful_waves_four_rows","useful_waves_eight_rows"};
    std::vector<std::unique_ptr<Graph>> g;
    for(int v=0;v<5;++v) g.emplace_back(new Graph(m,names[v],[&,v](int i){qlaunch<K>(v,M,da+(i%2)*weights,dx,db,out[v]+GUARD,m.stream);}));
    time_graphs("q8",M,K,"block_q8_0_row_major",g);
    auto base=read_output(out[0],M);
    for(int v=1;v<5;++v) {auto r=read_output(out[v],M);for(int i=0;i<M;++i){need(bits(base[i+GUARD])==bits(r[i+GUARD]),"Q8 post-timing mismatch");++exact_checks;}}
    std::printf("{\"kind\":\"case_pass\",\"suite\":\"q8\",\"M\":%d,\"K\":%d,\"weight_matrices\":2,\"bias_none_separate_alias\":true,\"bit_exact\":true,\"allocation_mib\":%.3f}\n",M,K,double(m.bytes)/MiB);
}

template<bool COL> static __global__ void float_gemv(const float *a,const float *x,float *y,int M,int K) {
    int row=blockIdx.x*4+threadIdx.y,lane=threadIdx.x;float sum=0;
    if(row<M) for(int k=lane;k<K;k+=32) sum+=a[COL?size_t(k)*M+row:size_t(row)*K+k]*x[k];
    for(int d=16;d;d>>=1) sum+=__shfl_xor(sum,d,32);
    if(row<M&&lane==0) y[row]=sum;
}
static void float_case(int M,int K,bool col) {
    need(M>0&&M<=248144&&K>=128&&K<=512,"Float shape bound");
    Memory m;size_t elements=size_t(M)*K;std::vector<float>a(elements*2),x(K);
    uint32_t rng=104729+M+K;
    for(auto &v:a) v=float(int(random_u32(rng)%256)-128)/64;
    for(auto &v:x) v=float(int(random_u32(rng)%256)-128)/32;
    auto da=m.alloc<float>(a.size());auto dx=m.alloc<float>(K);
    std::array<float*,2> out{m.alloc<float>(M+2*GUARD),m.alloc<float>(M+2*GUARD)};
    ck(hipMemcpyAsync(da,a.data(),a.size()*4,hipMemcpyHostToDevice,m.stream));ck(hipMemcpyAsync(dx,x.data(),K*4,hipMemcpyHostToDevice,m.stream));
    rocblas_handle handle=nullptr;rb(rocblas_create_handle(&handle));rb(rocblas_set_stream(handle,m.stream));
    rb(rocblas_set_pointer_mode(handle,rocblas_pointer_mode_host));
    const float alpha=1,beta=0;
    auto launch=[&](int v,int rot) {
        const float *matrix=da+elements*rot;
        if(v==0) rb(rocblas_sgemv(handle,col?rocblas_operation_none:rocblas_operation_transpose,col?M:K,col?K:M,&alpha,matrix,col?M:K,dx,1,&beta,out[0]+GUARD,1));
        else if(col) float_gemv<true><<<(M+3)/4,dim3(32,4),0,m.stream>>>(matrix,dx,out[1]+GUARD,M,K);
        else float_gemv<false><<<(M+3)/4,dim3(32,4),0,m.stream>>>(matrix,dx,out[1]+GUARD,M,K);
        ck(hipGetLastError());
    };
    auto validate=[&](int rot) {
        std::vector<float> original;
        for(int v=0;v<2;++v) {
            ck(hipMemsetAsync(out[v],0xa5,(M+2*GUARD)*4,m.stream));launch(v,rot);ck(hipStreamSynchronize(m.stream));
            auto result=read_output(out[v],M);
            if(v==0) original=result;
            for(int i=0;i<M;++i) if(v) {need(std::fabs(result[i+GUARD]-original[i+GUARD])<=0.0002+0.00002*std::fabs(original[i+GUARD]),"rocBLAS/custom numeric mismatch");++numeric_checks;}
            for(int j=0;j<std::min(M,64);++j) {
                int row=j<32?j:M-1-(j-32);double expected=0;
                for(int k=0;k<K;++k) expected+=double(a[rot*elements+(col?size_t(k)*M+row:size_t(row)*K+k)])*x[k];
                need(std::fabs(result[row+GUARD]-expected)<=0.0002+0.00002*std::fabs(expected),"FP32 independent CPU reference failed");++numeric_checks;
            }
        }
    };
    validate(0);validate(1);
    {std::vector<std::unique_ptr<Graph>>g;
    g.emplace_back(new Graph(m,"rocblas_sgemv",[&](int i){launch(0,i%2);}));
    g.emplace_back(new Graph(m,"custom_wave32_four_rows",[&](int i){launch(1,i%2);}));
    time_graphs("fp32",M,K,col?"column_major_nontransposed":"row_major_via_transpose",g);
    auto first=read_output(out[0],M),second=read_output(out[1],M);
    for(int i=0;i<M;++i) {need(std::fabs(first[i+GUARD]-second[i+GUARD])<=0.0002+0.00002*std::fabs(first[i+GUARD]),"FP32 post-timing mismatch");++numeric_checks;}}
    rb(rocblas_destroy_handle(handle));
    std::printf("{\"kind\":\"case_pass\",\"suite\":\"fp32\",\"M\":%d,\"K\":%d,\"layout\":\"%s\",\"allocation_mib\":%.3f}\n",M,K,col?"column_major_nontransposed":"row_major_via_transpose",double(m.bytes)/MiB);
}

static void copies() {
    Memory m;constexpr size_t n=8*MiB;auto src=m.alloc<unsigned char>(n),dst=m.alloc<unsigned char>(n);
    ck(hipMemsetAsync(src,0x5a,n,m.stream));ck(hipMemsetAsync(dst,0,n,m.stream));
    std::vector<std::unique_ptr<Graph>>g;
    g.emplace_back(new Graph(m,"device_copy_8mib",[&](int){ck(hipMemcpyAsync(dst,src,n,hipMemcpyDeviceToDevice,m.stream));}));
    time_graphs("runtime",int(n),0,"bytes",g);
    std::vector<unsigned char> check(n);ck(hipMemcpy(check.data(),dst,n,hipMemcpyDeviceToHost));
    need(std::all_of(check.begin(),check.end(),[](unsigned char c){return c==0x5a;}),"Device copy correctness failed");
    std::puts("{\"kind\":\"copy_pass\",\"bytes_checked\":8388608}");
}

static int run(int argc,char **argv) {
    bool probe=argc==2&&std::string(argv[1])=="--list-devices";
    need(probe||(argc==3&&std::string(argv[1])=="--suite"),"Explicit suite required");std::string suite=probe?"smoke":argv[2];
    need(suite=="smoke"||suite=="q8"||suite=="blas"||suite=="runtime","Unknown suite");
    adjacent(L"amdhip64_7.dll");adjacent(L"rocblas.dll");
    // GEMM backends are optional in the upstream native-BLAS build.
    // Every loaded optional backend must still come from this private directory.
    for (const auto *name : {L"libhipblaslt.dll", L"libtensilelite-host.dll", L"origami.dll"})
        if (GetModuleHandleW(name)) adjacent(name);
    int count=0,selected=-1;ck(hipGetDeviceCount(&count));need(count>0&&count<=16,"GPU count bound");
    hipDeviceProp_t prop{};
    for(int i=0;i<count;++i) {hipDeviceProp_t p{};ck(hipGetDeviceProperties(&p,i));if(!std::strcmp(p.name,"AMD Radeon RX 9060 XT")){need(selected<0,"9060 identity ambiguous");selected=i;prop=p;}}
    need(selected>=0&&!std::strncmp(prop.gcnArchName,"gfx1200",7)&&prop.warpSize==32,"9060 architecture requirement");
    ck(hipSetDevice(selected));int clock=0;ck(hipDeviceGetAttribute(&clock,hipDeviceAttributeWallClockRate,selected));need(clock==100000,"Unvalidated wall-clock frequency");
    size_t free=0,total=0;ck(hipMemGetInfo(&free,&total));need(total/MiB==16304&&free>=2048*MiB,"9060 initial capacity/headroom requirement");
    if(probe) {
        std::printf("{\"kind\":\"memory_probe\",\"device\":%d,\"name\":\"AMD Radeon RX 9060 XT\",\"arch\":\"gfx1200\",\"total_mib\":%zu,\"free_mib\":%zu,\"allocation_bytes\":0,\"kernels_launched\":0,\"runtime_adjacent\":true}\n",selected,total/MiB,free/MiB);
        return 0;
    }
    int runtime=0;ck(hipRuntimeGetVersion(&runtime));char rocversion[128]{};rb(rocblas_get_version_string(rocversion,sizeof(rocversion)));
    for(char *p=rocversion;*p;++p) need(std::isalnum((unsigned char)*p)||*p=='.'||*p=='-'||*p=='_',"Unexpected version string");
    std::printf("{\"kind\":\"configuration\",\"suite\":\"%s\",\"device\":%d,\"gpu\":\"RX 9060 XT\",\"arch\":\"gfx1200\",\"warp\":32,\"clock_khz\":%d,\"runtime_version\":%d,\"rocblas\":\"%s\",\"free_mib\":%zu,\"model_free\":true,\"second_gpu_compute\":false,\"runtime_adjacent\":true,\"graph_batch_calls\":8}\n",suite.c_str(),selected,clock,runtime,rocversion,free/MiB);
    calibration();
    if(suite=="smoke") {quant_case<128>(257);quant_case<256>(4097);quant_case<256>(248144);quant_case<512>(257);quant_case<1024>(257);float_case(4097,256,false);float_case(4097,256,true);copies();}
    if(suite=="q8") for(int M:{257,4096,65536,248144}) {quant_case<128>(M);quant_case<256>(M);quant_case<512>(M);quant_case<1024>(M);}
    if(suite=="blas") for(auto shape:{std::array<int,2>{4096,256},{65536,128},{65536,256},{65536,512},{248144,256}}) {float_case(shape[0],shape[1],false);float_case(shape[0],shape[1],true);}
    if(suite=="runtime") copies();
    ck(hipDeviceSynchronize());need(cleanup_errors==0,"HIP cleanup failure");
    std::printf("{\"kind\":\"complete\",\"suite\":\"%s\",\"q8_bit_comparisons\":%llu,\"numeric_reference_comparisons\":%llu,\"guard_values_checked\":%llu,\"mismatches\":0,\"max_owned_allocation_mib\":%.3f,\"second_gpu_compute\":false}\n",suite.c_str(),(unsigned long long)exact_checks,(unsigned long long)numeric_checks,(unsigned long long)guard_checks,peak_allocation_mib);
    return 0;
}
int main(int argc,char **argv) {std::setvbuf(stdout,nullptr,_IONBF,0);try{return run(argc,argv);}catch(const std::exception &e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 2;}}
