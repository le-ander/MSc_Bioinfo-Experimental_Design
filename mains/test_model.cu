#define NSPECIES 6
#define NPARAM 5
#define NREACT 12

struct myFex{
    __device__ void operator()(int *neq, double *t, double *y, double *ydot/*, void *otherData*/)
    {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;



        ydot[5]=1.0*(tex2D(param_tex,4,tid)*y[4])-1.0*(tex2D(param_tex,4,tid)*y[5]);
        ydot[4]=1.0*(tex2D(param_tex,2,tid)+tex2D(param_tex,3,tid)/(1+__powf(y[3],tex2D(param_tex,1,tid))))-1.0*(y[4]);
        ydot[3]=1.0*(tex2D(param_tex,4,tid)*y[2])-1.0*(tex2D(param_tex,4,tid)*y[3]);
        ydot[2]=1.0*(tex2D(param_tex,2,tid)+tex2D(param_tex,3,tid)/(1+__powf(y[1],tex2D(param_tex,1,tid))))-1.0*(y[2]);
        ydot[1]=1.0*(tex2D(param_tex,4,tid)*y[0])-1.0*(tex2D(param_tex,4,tid)*y[1]);
        ydot[0]=1.0*(tex2D(param_tex,2,tid)+tex2D(param_tex,3,tid)/(1+__powf(y[5],tex2D(param_tex,1,tid))))-1.0*(y[0]);

    }
};


 struct myJex{
    __device__ void operator()(int *neq, double *t, double *y, int ml, int mu, double *pd, int nrowpd/*, void *otherData*/){
        return; 
    }
};