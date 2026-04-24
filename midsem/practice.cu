%%writefile complement.cu
#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>



__global__ void Addkernel(int *da,int *db, int n, int *dc)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;

    if(i<n)
    {
        dc[i]=da[i]+db[i];
    }
}



int main(){
    int n;
    int *db,*da,*dc;
printf("Enter n:");
scanf("%d",&n);
int size=sizeof(int)*n;


int *ha=(int*)malloc(sizeof(int)*n);
int *hb=(int*)malloc(sizeof(int)*n);
int *hc=(int*)malloc(sizeof(int)*n);
  for(int i = 0; i < n; i++) 
    { 
      ha[i] = i;
      hb[i] = i * 2;
    }

cudaMalloc((void**)&da,sizeof(int)*n);
cudaMalloc((void**)&db,sizeof(int)*n);
cudaMalloc((void**)&dc,sizeof(int)*n);

cudaMemcpy(da,ha,size,cudaMemcpyHostToDevice);
cudaMemcpy(db,hb,size,cudaMemcpyHostToDevice);

Addkernel<<<1,n>>>(da,db,n,dc);
cudaMemcpy(hc,dc,size,cudaMemcpyDeviceToHost);

printf("OUTPUT\n");
  for(int i = 0; i < n; i++) 
    { 
      printf("%d",hc[i]);
    }
cudaFree(da);
cudaFree(db);
cudaFree(dc);


}