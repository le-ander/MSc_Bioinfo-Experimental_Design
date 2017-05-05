#define NSPECIES 6
#define NPARAM 4
#define NREACT 12

#define leq(a,b) a<=b
#define neq(a,b) a!=b
#define geq(a,b) a>=b
#define lt(a,b) a<b
#define gt(a,b) a>b
#define eq(a,b) a==b
#define and_(a,b) a&&b
#define or_(a,b) a||b
struct myFex{
    __device__ void operator()(int *neq, double *t, double *y, double *ydot/*, void *otherData*/)
    {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;


        ydot[0]= -y[0] + (tex2D(param_tex,2,tid) / ( 1 + powf(y[5],tex2D(param_tex,0,tid)))) + 0.1*tex2D(param_tex,1,tid);
        ydot[1]= -tex2D(param_tex,3,tid) * (y[1] - y[0]);
        ydot[2]= -y[2] + (tex2D(param_tex,2,tid) / (1 + powf(y[1],tex2D(param_tex,0,tid)))) + 0.1*tex2D(param_tex,1,tid);
        ydot[3]= -tex2D(param_tex,3,tid) * (y[3] - y[2]);
        ydot[4]= -y[4] + (tex2D(param_tex,2,tid) / ( 1 + powf(y[3],tex2D(param_tex,0,tid)))) + 0.1*tex2D(param_tex,1,tid);
        ydot[5]= -tex2D(param_tex,3,tid) * (y[5] - y[4]);
       
    }
};


 struct myJex{
    __device__ void operator()(int *neq, double *t, double *y, int ml, int mu, double *pd, int nrowpd/*, void *otherData*/){
        return; 
    }
};
