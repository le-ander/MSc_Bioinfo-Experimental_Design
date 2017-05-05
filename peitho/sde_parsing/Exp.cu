#define NSPECIES 27
#define NPARAM 5
#define NREACT 12

struct myFex{
    __device__ void operator()(int *neq, double *t, double *y, double *ydot/*, void *otherData*/)
    {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;



        ydot[26]=2*tex2D(param_tex,3,tid)*y[25] - 2*tex2D(param_tex,3,tid)*y[26] + tex2D(param_tex,3,tid)*y[4] + tex2D(param_tex,3,tid)*y[5]
        ydot[25]=tex2D(param_tex,3,tid)*y[24] - tex2D(param_tex,3,tid)*y[25] - y[23]*tex2D(param_tex,1,tid)*tex2D(param_tex,2,tid)*__powf(y[3],tex2D(param_tex,2,tid))/(y[3]*__powf((__powf(y[3],tex2D(param_tex,2,tid)) + 1),2)) - y[25]
        ydot[24]=-2*y[22]*tex2D(param_tex,1,tid)*tex2D(param_tex,2,tid)*__powf(y[3],tex2D(param_tex,2,tid))/(y[3]*__powf((__powf(y[3],tex2D(param_tex,2,tid)) + 1),2)) - 2*y[24] + tex2D(param_tex,1,tid)/(__powf(y[3],tex2D(param_tex,2,tid)) + 1) + tex2D(param_tex,4,tid) + y[4]
        ydot[23]=tex2D(param_tex,3,tid)*y[20] + tex2D(param_tex,3,tid)*y[22] - 2*tex2D(param_tex,3,tid)*y[23]
        ydot[22]=tex2D(param_tex,3,tid)*y[19] - tex2D(param_tex,3,tid)*y[22] - y[21]*tex2D(param_tex,1,tid)*tex2D(param_tex,2,tid)*__powf(y[3],tex2D(param_tex,2,tid))/(y[3]*__powf((__powf(y[3],tex2D(param_tex,2,tid)) + 1),2)) - y[22]
        ydot[21]=2*tex2D(param_tex,3,tid)*y[18] - 2*tex2D(param_tex,3,tid)*y[21] + tex2D(param_tex,3,tid)*y[2] + tex2D(param_tex,3,tid)*y[3]
        ydot[20]=tex2D(param_tex,3,tid)*y[19] - tex2D(param_tex,3,tid)*y[20] - y[16]*tex2D(param_tex,1,tid)*tex2D(param_tex,2,tid)*__powf(y[1],tex2D(param_tex,2,tid))/(y[1]*__powf((__powf(y[1],tex2D(param_tex,2,tid)) + 1),2)) - y[20]
        ydot[19]=-y[15]*tex2D(param_tex,1,tid)*tex2D(param_tex,2,tid)*__powf(y[1],tex2D(param_tex,2,tid))/(y[1]*__powf((__powf(y[1],tex2D(param_tex,2,tid)) + 1),2)) - y[18]*tex2D(param_tex,1,tid)*tex2D(param_tex,2,tid)*__powf(y[3],tex2D(param_tex,2,tid))/(y[3]*__powf((__powf(y[3],tex2D(param_tex,2,tid)) + 1),2)) - 2*y[19]
        ydot[18]=tex2D(param_tex,3,tid)*y[17] - tex2D(param_tex,3,tid)*y[18] - y[14]*tex2D(param_tex,1,tid)*tex2D(param_tex,2,tid)*__powf(y[1],tex2D(param_tex,2,tid))/(y[1]*__powf((__powf(y[1],tex2D(param_tex,2,tid)) + 1),2)) - y[18]
        ydot[17]=-2*y[13]*tex2D(param_tex,1,tid)*tex2D(param_tex,2,tid)*__powf(y[1],tex2D(param_tex,2,tid))/(y[1]*__powf((__powf(y[1],tex2D(param_tex,2,tid)) + 1),2)) - 2*y[17] + tex2D(param_tex,1,tid)/(__powf(y[1],tex2D(param_tex,2,tid)) + 1) + tex2D(param_tex,4,tid) + y[2]
        ydot[16]=tex2D(param_tex,3,tid)*y[11] + tex2D(param_tex,3,tid)*y[15] - 2*tex2D(param_tex,3,tid)*y[16]
        ydot[15]=tex2D(param_tex,3,tid)*y[10] - tex2D(param_tex,3,tid)*y[15] - y[14]*tex2D(param_tex,1,tid)*tex2D(param_tex,2,tid)*__powf(y[3],tex2D(param_tex,2,tid))/(y[3]*__powf((__powf(y[3],tex2D(param_tex,2,tid)) + 1),2)) - y[15]
        ydot[14]=tex2D(param_tex,3,tid)*y[9] + tex2D(param_tex,3,tid)*y[13] - 2*tex2D(param_tex,3,tid)*y[14]
        ydot[13]=tex2D(param_tex,3,tid)*y[8] - tex2D(param_tex,3,tid)*y[13] - y[12]*tex2D(param_tex,1,tid)*tex2D(param_tex,2,tid)*__powf(y[1],tex2D(param_tex,2,tid))/(y[1]*__powf((__powf(y[1],tex2D(param_tex,2,tid)) + 1),2)) - y[13]
        ydot[12]=2*tex2D(param_tex,3,tid)*y[7] - 2*tex2D(param_tex,3,tid)*y[12] + tex2D(param_tex,3,tid)*y[0] + tex2D(param_tex,3,tid)*y[1]
        ydot[11]=tex2D(param_tex,3,tid)*y[10] - tex2D(param_tex,3,tid)*y[11] - y[11] - y[26]*tex2D(param_tex,1,tid)*tex2D(param_tex,2,tid)*__powf(y[5],tex2D(param_tex,2,tid))/(y[5]*__powf((__powf(y[5],tex2D(param_tex,2,tid)) + 1),)2,tid)) + 1)**2)
        ydot[10]=-y[9]*tex2D(param_tex,1,tid)*tex2D(param_tex,2,tid)*__powf(y[3],tex2D(param_tex,2,tid))/(y[3]*__powf((__powf(y[3],tex2D(param_tex,2,tid)) + 1),2)) - 2*y[10] - y[25]*tex2D(param_tex,1,tid)*tex2D(param_tex,2,tid)*__powf(y[5],tex2D(param_tex,2,tid))/(y[5]*__powf((__powf(y[5],tex2D(param_tex,2,tid)) + 1),)2,tid)) + 1)**2)
        ydot[9]=tex2D(param_tex,3,tid)*y[8] - tex2D(param_tex,3,tid)*y[9] - y[9] - y[23]*tex2D(param_tex,1,tid)*tex2D(param_tex,2,tid)*__powf(y[5],tex2D(param_tex,2,tid))/(y[5]*__powf((__powf(y[5],tex2D(param_tex,2,tid)) + 1),)2,tid)) + 1)**2)
        ydot[8]=-y[7]*tex2D(param_tex,1,tid)*tex2D(param_tex,2,tid)*__powf(y[1],tex2D(param_tex,2,tid))/(y[1]*__powf((__powf(y[1],tex2D(param_tex,2,tid)) + 1),2)) - 2*y[8] - y[20]*tex2D(param_tex,1,tid)*tex2D(param_tex,2,tid)*__powf(y[5],tex2D(param_tex,2,tid))/(y[5]*__powf((__powf(y[5],tex2D(param_tex,2,tid)) + 1),)2,tid)) + 1)**2)
        ydot[7]=tex2D(param_tex,3,tid)*y[6] - tex2D(param_tex,3,tid)*y[7] - y[7] - y[16]*tex2D(param_tex,1,tid)*tex2D(param_tex,2,tid)*__powf(y[5],tex2D(param_tex,2,tid))/(y[5]*__powf((__powf(y[5],tex2D(param_tex,2,tid)) + 1),)2,tid)) + 1)**2)
        ydot[6]=-2*y[6] - 2*y[11]*tex2D(param_tex,1,tid)*tex2D(param_tex,2,tid)*__powf(y[5],tex2D(param_tex,2,tid))/(y[5]*__powf((__powf(y[5],tex2D(param_tex,2,tid)) + 1),2)) + tex2D(param_tex,1,tid)/(__powf(y[5],tex2D(param_tex,2,tid)) + 1) + tex2D(param_tex,4,tid) + y[0]
        ydot[5]=tex2D(param_tex,3,tid)*y[4] - tex2D(param_tex,3,tid)*y[5]
        ydot[4]=tex2D(param_tex,1,tid)/(__powf(y[3],tex2D(param_tex,2,tid)) + 1) + tex2D(param_tex,4,tid) - y[4]
        ydot[3]=tex2D(param_tex,3,tid)*y[2] - tex2D(param_tex,3,tid)*y[3]
        ydot[2]=tex2D(param_tex,1,tid)/(__powf(y[1],tex2D(param_tex,2,tid)) + 1) + tex2D(param_tex,4,tid) - y[2]
        ydot[1]=tex2D(param_tex,3,tid)*y[0] - tex2D(param_tex,3,tid)*y[1]
        ydot[0]=tex2D(param_tex,1,tid)/(__powf(y[5],tex2D(param_tex,2,tid)) + 1) + tex2D(param_tex,4,tid) - y[0]

    }
};


 struct myJex{
    __device__ void operator()(int *neq, double *t, double *y, int ml, int mu, double *pd, int nrowpd/*, void *otherData*/){
        return; 
    }
};