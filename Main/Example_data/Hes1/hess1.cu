#define NSPECIES 3
#define NPARAM 4
#define NREACT 1



struct myFex{
    __device__ void operator()(int *neq, double *t, double *y, double *ydot/*, void *otherData*/)
    {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;

	ydot[0] = -0.03*y[0]+1/(1+powf(y[2]/tex2D(param_tex,0,tid),tex2D(param_tex,1,tid)));
	ydot[1] = -0.03*y[1]+tex2D(param_tex,2,tid)*y[0]-(tex2D(param_tex,3,tid))*y[1];
	ydot[2] = -0.03*y[2]+tex2D(param_tex,3,tid)*y[1];

    }
};


 struct myJex{
    __device__ void operator()(int *neq, double *t, double *y, int ml, int mu, double *pd, int nrowpd/*, void *otherData*/){
        return; 
    }
};