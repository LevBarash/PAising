//
// PAising version 1.12. This program employs multi-spin coding.
// This program is introduced in the paper:
// L.Yu. Barash, M. Weigel, M. Borovsky, W. Janke, L.N. Shchur, GPU accelerated population annealing algorithm
// This program is licensed under a Creative Commons Attribution 4.0 International License:
// http://creativecommons.org/licenses/by/4.0/
//
// Use command line option -? to print list of available command line options.
// All of the command line options are optional. 
//

#include <iostream>
#include <fstream>
#include <iomanip>
#include <curand_kernel.h>
#ifdef _WIN32			// this program is compatible with any of the Windows, Unix/Linux, MacOS environments
	#include <direct.h>
#else
	#include <sys/stat.h>
#endif

// #define MHR			// uncomment/comment to enable/disable multi-histogram reweighting
// #define AdaptiveStep		// uncomment/comment to enable/disable adaptive temperature step
// #define EnergiesPopStore	// uncomment/comment to enable/disable storing energies at each T

#define L	64		// linear size of the system in x,y direction
#define Ldiv2   (L/2)
#define N       (L*L)

#define RNGseed	time(NULL)	// Use 32-bit integer as a seed for random number generation, e.g., time(NULL) 

typedef curandStatePhilox4_32_10_t RNGState;

#define MSbits	32		// Use 8, 16, 32 or 64 Multi-spin bits per word

unsigned int EQsweeps = 100;			// number of equilibration sweeps

double Binit = 0;				// initial inverse temperature
double Bfin = 1;				// final inverse temperature
double dBinit = 0.005;				// inverse temperature step

#ifdef AdaptiveStep
	double MinOverlap = 0.85;		// minimal value of acceptable overlap of energy histograms
	double MaxOverlap = 0.87;		// maximal value of acceptable overlap of energy histograms
#endif

int Rinit = 20000;				// Initial size of population of replicas

int runs = 1;					// number of population annealing algorithm independent runs

int OutputPrecision = 11;			// precision (number of digits) of the output

const unsigned int AA = 1664525;		// linear congruential generator parameters
const unsigned int CC = 1013904223;

#ifdef MHR
	const short MHR_Niter = 1;	// number of iterations for multi-histogram analysis (single iteration is usually sufficient)
#endif

const int boltzTableL = 2;			// Boltzmann factor table length
const int nBmax = 10000;			// number of temperature steps should not exceed nBmax

texture<unsigned int,1,cudaReadModeElementType> boltzT;
using namespace std;

#define EQthreads 128	// number of threads per block for the equilibration kernel
#define Nthreads  1024	// number of threads per block for the parallel reduction algorithm
// Use Nthreads=1024 for CUDA compute capability 2.0 and above; Nthreads=512 for old devices with CUDA compute capability 1.x.

double* Qd; double* ioverlapd;

#if   MSbits == 8
	#define MultiSpin signed char
#elif MSbits == 16
	#define MultiSpin signed short
#elif MSbits == 32
	#define MultiSpin signed int
#elif MSbits == 64
	#define MultiSpin signed long long int
#endif

// struct Replica covers all information about the replica including its configuration, sublattice magnetizations,
// internal energy and number of replica's offspring
struct Replica{
		MultiSpin gA[N/2];	// sublattice configurations with multipsin-coding = one value in array represents
		MultiSpin gB[N/2];	// spins of 8 different replicas in the same site in lattice
		int IE[MSbits];				// internal energy
		int M[MSbits];				// magnetization
		unsigned int Roff[MSbits];		// number of replica's offspring
		union{double ValDouble[2]; unsigned int ValInt[MSbits+2];} parSum;  // these variables are used for storing sums
		bool isActive[MSbits];			// isActive[i] determines if the i-th replica is active
};

// CUDA error checking macro
#define CUDAErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) {
		fprintf(stderr,"GPUassert: %s ; %s ; line %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

template <class sometype> __inline__ __device__ sometype smallblockReduceSum(sometype val) // use when blockDim.x < 32
{											   // blockDim.x must be a power of 2
	static __shared__ sometype shared[32];
	shared[threadIdx.x] = val;
	for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1){
		__syncthreads(); if (threadIdx.x < stride)  shared[threadIdx.x] += shared[threadIdx.x+stride];
	}
	__syncthreads(); return shared[0];
}

#if  (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300)
template <class sometype> __inline__ __device__ sometype warpReduceSum(sometype val)
{
	for (int offset = warpSize/2; offset > 0; offset /= 2) val += __shfl_down(val, offset);
	return val;
}

template <class sometype> __inline__ __device__ sometype blockReduceSum(sometype val)	 // use when blockDim.x is divisible by 32
{
	static __shared__ sometype shared[32];			// one needs to additionally synchronize threads after execution
	int lane = threadIdx.x % warpSize;			// in the case of multiple use of blockReduceSum in a single kernel
	int wid = threadIdx.x / warpSize;
	val = warpReduceSum(val);
	if (lane==0) shared[wid]=val;
	__syncthreads();
	val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
	if (wid==0) val = warpReduceSum(val);
	return val;
}
#else
template <class sometype> __inline__ __device__ sometype blockReduceSum(sometype val)	// blockDim.x must be a power of 2
{
	static __shared__ sometype shared[Nthreads];
	shared[threadIdx.x] = val;
	for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1){
		__syncthreads(); if (threadIdx.x < stride)  shared[threadIdx.x] += shared[threadIdx.x+stride];
	}
	__syncthreads(); return shared[0];
}
#endif

#if (__CUDACC_VER_MAJOR__ < 8) || ( defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600 )
	__device__ double atomicAdd(double* address, double val) // allows to use atomicAdd operation for double precision floating point values
	{ 
		unsigned long long int* address_as_ull = (unsigned long long int*)address; 
		unsigned long long int old = *address_as_ull, assumed; 
		do { 
			assumed = old; 
			old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed))); 
		} while (assumed != old); 
		return __longlong_as_double(old); 
	}
#endif

#if  (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 320 && MSbits == 64)
	__device__ unsigned long long int atomicXor(unsigned long long int* address, unsigned long long int val) // allows to use atomicXor operation for 64-bit integers
	{
		unsigned int val1,val2; val1 = val & 0xFFFFFFFF; val2 = val>>32;
		val1 = atomicXor((unsigned int*)address,val1);
		val2 = atomicXor((unsigned int*)address+1,val2);
		return (unsigned long long int)val1 + ((unsigned long long int)val2)<<32;
	}

#endif

__global__ void ReplicaInit(Replica* Rd, int rg, int R, unsigned long long rng_seed, unsigned long long initial_sequence){ // initialization of spin lattices of all replicas
	unsigned int B = blockIdx.x, t = threadIdx.x;
	RNGState localrng; curand_init(rng_seed,initial_sequence+(t+B*EQthreads),0,&localrng);
	for (unsigned int idx = t; idx < (N/2); idx += EQthreads){
			#if   MSbits == 8
				Rd[B].gA[idx] = curand(&localrng) & 0xFF;
				Rd[B].gB[idx] = curand(&localrng) & 0xFF;
			#elif MSbits == 16
				Rd[B].gA[idx] = curand(&localrng) & 0xFFFF;
				Rd[B].gB[idx] = curand(&localrng) & 0xFFFF;
			#elif MSbits == 32
				Rd[B].gA[idx] = curand(&localrng);
				Rd[B].gB[idx] = curand(&localrng);
			#elif MSbits == 64
				Rd[B].gA[idx] = (((unsigned long long int)curand(&localrng))<<32) + curand(&localrng) ;
				Rd[B].gB[idx] = (((unsigned long long int)curand(&localrng))<<32) + curand(&localrng) ;
			#endif
	}
	if(t < MSbits) if((B*MSbits+t)<R) Rd[B].isActive[t] = true; else Rd[B].isActive[t] = false;
}

// parallel spin update
__global__ void checkKerALL(Replica* Rd, int rg, unsigned int sweeps, unsigned long long rng_seed, unsigned long long initial_sequence) // equilibration process
{
	MultiSpin mspin; unsigned int B = blockIdx.x, t = threadIdx.x, ran, idx, i1, i3, i4, tx, ty; // B is replica index

	RNGState localrng; curand_init(rng_seed,initial_sequence+(t+blockIdx.x*EQthreads),0,&localrng);

	for(int sweep=0; sweep<sweeps; sweep++){ // sweeps loop

		// sublattice A

		for (idx = t; idx < (N/2); idx += EQthreads){	// sublattice A
			ty = idx / Ldiv2; tx = idx - ty * Ldiv2;
			i1 = ty * Ldiv2 + ((ty&1) ? (tx + 1) : (tx + Ldiv2 - 1)) % Ldiv2;
			i3 = ((ty + L - 1) % L) * Ldiv2 + tx; i4 = ((ty + 1) % L) * Ldiv2 + tx;
			mspin = Rd[B].gA[idx];
			// detecting anti-parallel orientations with NN (Ii = S ^ Ni)
			MultiSpin I1 = mspin ^ Rd[B].gB[i1]; // left- or right-neighbour in B
			MultiSpin I2 = mspin ^ Rd[B].gB[idx]; // right- or left-neighbour spins in the sublattice B
			MultiSpin I3 = mspin ^ Rd[B].gB[i3]; // lower-neighbour spins in the sublattice B
			MultiSpin I4 = mspin ^ Rd[B].gB[i4]; // upper-neighbour spins in the sublattice B
			// performing summation of anti-parallel couplings
			MultiSpin x12 = I1 ^ I2;
			MultiSpin x34 = I3 ^ I4;
			MultiSpin a12 = I1 & I2;
			MultiSpin a34 = I3 & I4;	
			MultiSpin sum0 = x12 ^ x34;
			MultiSpin sum1 = x12 & x34 ^ a12 ^ a34;
			MultiSpin sum2 = a12 & a34;	
			MultiSpin cond4 = 0;
			MultiSpin cond8 = 0; MultiSpin imask=0x1; ran = curand(&localrng);
			for (unsigned char i = 0; i < MSbits; ++i){
				cond4 |= (-(ran < tex1Dfetch(boltzT, 0))) & imask;
				cond8 |= (-(ran < tex1Dfetch(boltzT, 1))) & imask;
				imask <<= 1;	ran = AA * ran + CC;
			}
			// acceptance mask
			MultiSpin Acc = (sum1|sum2) | ( (~(sum1|sum2)) & ((sum0&cond4) | (~sum0&cond8)) );
			// Metropolis update + store new configuration to global memory
			Rd[B].gA[idx] = mspin ^ Acc;
		}

		__syncthreads();

		// sublattice B

		for (idx = t; idx < (N/2); idx += EQthreads){	// sublattice B
			ty = idx / Ldiv2; tx = idx - ty * Ldiv2;
			i1 = ty * Ldiv2 + ((ty&1) ? (tx + Ldiv2 - 1) : (tx + 1)) % Ldiv2;
			i3 = ((ty + L - 1) % L) * Ldiv2 + tx; i4 = ((ty + 1) % L) * Ldiv2 + tx;
			mspin = Rd[B].gB[idx];
			MultiSpin I1 = mspin ^ Rd[B].gA[i1]; // left- or right-neighbour in A
			MultiSpin I2 = mspin ^ Rd[B].gA[idx];// right- or left-neighbour spins in the sublattice A
			MultiSpin I3 = mspin ^ Rd[B].gA[i3]; // lower-neighbour spins in the sublattice A
			MultiSpin I4 = mspin ^ Rd[B].gA[i4]; // upper-neighbour spins in the sublattice A
			MultiSpin x12 = I1 ^ I2;
			MultiSpin x34 = I3 ^ I4;
			MultiSpin a12 = I1 & I2;
			MultiSpin a34 = I3 & I4;	
			MultiSpin sum0 = x12 ^ x34;
			MultiSpin sum1 = x12 & x34 ^ a12 ^ a34;
			MultiSpin sum2 = a12 & a34;
			MultiSpin cond4 = 0;
			MultiSpin cond8 = 0; MultiSpin imask=0x1; ran = curand(&localrng);
			for (unsigned char i = 0; i < MSbits; ++i){
				cond4 |= (-(ran < tex1Dfetch(boltzT, 0))) & imask;
				cond8 |= (-(ran < tex1Dfetch(boltzT, 1))) & imask;
				imask <<= 1;	ran = AA * ran + CC;
			}	
			MultiSpin Acc = (sum1|sum2) | ( (~(sum1|sum2)) & ((sum0&cond4) | (~sum0&cond8)) );
			Rd[B].gB[idx] = mspin ^ Acc;
		}

		__syncthreads();
	}

}

__global__ void energyKer(Replica* Rd) // calculation of energy and magnetization for each replica
{
	int e, m; unsigned int t = threadIdx.x, idx, iL, iU, B = blockIdx.x, tx, ty;
	MultiSpin sum0, sum1, sum2, sA, sB, Ai2, Bi2, Ai4, Bi4;
	for (idx = t; idx < (N/2); idx += EQthreads){
		if(t < EQthreads){
			sA = Rd[B].gA[idx]; sB = Rd[B].gB[idx];
			ty = idx / Ldiv2; tx = idx - ty * Ldiv2;
			iL = ty * Ldiv2 + (tx + Ldiv2 - 1) % Ldiv2;
			iU = ((ty + 1) % L) * Ldiv2 + tx;
			if(ty&1){ Ai2 = sB; Bi2 = Rd[B].gA[iL];  }
			else{     Ai2 = Rd[B].gB[iL];  Bi2 = sA; }
			Ai4 = Rd[B].gB[iU]; Bi4 = Rd[B].gA[iU];
			// detecting anti-parallel orientations
			MultiSpin I1 = sA ^ Ai2;
			MultiSpin I2 = sA ^ Ai4;
			MultiSpin I3 = sB ^ Bi2;
			MultiSpin I4 = sB ^ Bi4;
			// performing summation of anti-parallel couplings
			MultiSpin x12 = I1 ^ I2;
			MultiSpin x34 = I3 ^ I4;
			MultiSpin a12 = I1 & I2;
			MultiSpin a34 = I3 & I4;
			sum0 = x12 ^ x34;
			sum1 = x12 & x34 ^ a12 ^ a34;
			sum2 = a12 & a34;
		}
		// calculating energy contributions for replicas
		for (unsigned char i = 0; i < MSbits; ++i){
			if(t < EQthreads){
				e = 2*((int)(sum0&0x1) + 2*(int)(sum1&0x1) + 4*(int)(sum2&0x1)) - 4;
				m = 2*((int)(sA&0x1) + (int)(sB&0x1)) - 2;
			} else e = m = 0;
			e = blockReduceSum<int>(e); __syncthreads();
			m = blockReduceSum<int>(m); __syncthreads();
			if (t == 0){
				if (idx==t){
					Rd[B].IE[i] = e;
					Rd[B].M[i]  = m; 
				}else{
					Rd[B].IE[i] += e;
					Rd[B].M[i]  += m; 
				}
			}
			// bit shift operation => moving to next replica in bit string
			sum0 >>= 1;	sum1 >>= 1; sum2 >>= 1;
			sA >>= 1; sB >>= 1;
		}
	}
}

__global__ void QKer(Replica* Rd, int rg, double dB, double Emean, int CalcPart, double* Qd) // calculation of partition function ratio
{
	if(CalcPart==0){			// first part of the calculation
		double factor; int idx = blockIdx.x; int br = threadIdx.x;	// summation of exponential
		factor = Rd[idx].isActive[br] ? exp(-dB*(Rd[idx].IE[br]-Emean)) : 0.0 ;	// Boltzmann-like factors
		#if MSbits < 32
			factor = smallblockReduceSum<double>(factor);
		#else
			factor = blockReduceSum<double>(factor);
		#endif
		if (br == 0) Rd[idx].parSum.ValDouble[0] = factor;	// is saved to global memory
	} else if(CalcPart==1){			// second part of the calculation
		double factor; int t = threadIdx.x; int b = blockIdx.x;
		int idx = t + Nthreads * b;
		factor = (idx < rg) ? Rd[idx].parSum.ValDouble[0]: 0.0;
		factor = blockReduceSum<double>(factor);
		if(t == 0 )  Rd[idx].parSum.ValDouble[1] = factor; // sum for all threads in current block is saved to global memory
	} else{					// third part of the calculation, summation of the partial sums
		double factor; int j, t = threadIdx.x; double MyParSum = 0;
		for (j=0; j<rg; j+=Nthreads){
			factor = (t+j)*Nthreads < rg ? Rd[(t+j)*Nthreads].parSum.ValDouble[1] : 0.0;
			factor = blockReduceSum<double>(factor); __syncthreads();
			MyParSum += factor;
		}
		if(t==0) *Qd = MyParSum;
	}
}

__global__ void CalcTauKer(Replica* Rd, int Rinit, int R, int rg, double lnQ, double dB, unsigned long long rng_seed, unsigned long long initial_sequence) // calculation of numbers of copies for all replicas
{
	int t = threadIdx.x; int b = blockIdx.x;
	unsigned char br = blockIdx.y;			// multispin replica index
	int idx = t + Nthreads * b; double mu, mufloor;
	if (idx < rg) if (Rd[idx].isActive[br]){	// nearest integer resampling
		mu = ((double)Rinit)/R*exp(-dB*(double)Rd[idx].IE[br] - lnQ);
		mufloor = floor(mu);
		RNGState localrng; curand_init(rng_seed,initial_sequence+(br+MSbits*idx),0,&localrng);
		if(curand_uniform(&localrng) < (mu-mufloor))
			Rd[idx].Roff[br] = mufloor + 1;
		else    Rd[idx].Roff[br] = mufloor;	// number of copies 
	} else Rd[idx].Roff[br] = 0;
}

__global__ void CalcParSum(Replica* Rd, int rg, int CalcPart, int* Rnew)
{
	if(CalcPart==0){	// first part of the calculation
		unsigned int parS; int t = threadIdx.x; int b = blockIdx.x;
		parS = Rd[b].Roff[t]; // (Rd[b].Roff[0] + Rd[b].Roff[1] + ... + Rd[b].Roff[MSbits-1]) is saved to global memory
		#if MSbits < 32
			parS = smallblockReduceSum<unsigned int>(parS);
		#else
			parS = blockReduceSum<unsigned int>(parS);
		#endif
		if(t==0) Rd[b].parSum.ValInt[MSbits] = parS;
	} else if(CalcPart==1){	// second part of the calculation
		unsigned int parS; int t = threadIdx.x; int b = blockIdx.x; int idx = t + b*Nthreads;
		parS = (idx < rg) ? Rd[idx].parSum.ValInt[MSbits] : 0;
		parS = blockReduceSum<unsigned int>(parS);
		// sum of partial sums for replica groups b*Nthreads,b*Nthreads+1,...,(b+1)*Nthreads-1 is saved to global memory.
		if(t==0) Rd[idx].parSum.ValInt[MSbits+1] = parS;
	} else{			// third part of the calculation
		unsigned int parS; int j, t = threadIdx.x, b = blockIdx.x;
		unsigned char br = blockIdx.y; __shared__ unsigned int val;
		int idx = t + Nthreads * b; unsigned int MyParSum = 0;
		for (j = 0; j<b; j+=Nthreads){		// we sum of Roff for all blocks from 0 to (b-1) and for all multi-spin indices.
			parS = (t+j < b) ? Rd[(t+j)*Nthreads].parSum.ValInt[MSbits+1] : 0;
			parS = blockReduceSum<unsigned int>(parS);
			if(t==0) val = parS; __syncthreads(); MyParSum += val;
		}
		if(idx < rg){
			for(j=Nthreads*b;j<idx;j++) MyParSum+=Rd[j].parSum.ValInt[MSbits]; // we add parSum[MSbits] for current block threads from 0 to (t-1)
			for(j=0;j<br;j++) MyParSum+=Rd[idx].Roff[j]; // we add Roff for j = 0,1,..., br-1.
			Rd[idx].parSum.ValInt[br] = MyParSum;                                   // we save partial sum
			if(idx==(rg-1)) if(br==(MSbits-1)) *Rnew = MyParSum + Rd[idx].Roff[br]; // we save new population size
		}
	}
}

__global__ void resampleKer(Replica* Rd, Replica* RdNew, int rg) // renumeration and copying of the replicas (the main part of the resampling process)
{	
	int t = threadIdx.x + blockIdx.z*blockDim.x;	// index of spin variable (from 0 -> N/2-1) 
	int bx = blockIdx.x;				// represents index of group of replicas (j)
	signed char by = blockIdx.y;			// represents index of replica in group/word (k)
	int it_k, it_j;
	#if   MSbits == 64
		unsigned long long int mask = 0x1; mask <<= by;	// mask for selecting spin from old population
		unsigned long long int copy_sourceA = mask & Rd[bx].gA[t];	// selected spin from sublattice A
		unsigned long long int copy_sourceB = mask & Rd[bx].gB[t];	// and B
	#else
		unsigned int mask = 0x1; mask <<= by;			// mask for selecting spin from old population
		unsigned int copy_sourceA = mask & Rd[bx].gA[t];	// selected spin from sublattice A
		unsigned int copy_sourceB = mask & Rd[bx].gB[t];	// and B
	#endif
	for (int p = 0; p < Rd[bx].Roff[by]; ++p){
		it_k = (Rd[bx].parSum.ValInt[by] + p) / rg;
		it_j = (Rd[bx].parSum.ValInt[by] + p) % rg;
		#if   MSbits == 8
			mask = 0x1; mask <<= (it_k + ((t&3)<<3));
			if(copy_sourceA!=0) atomicXor((unsigned int*)&(RdNew[it_j].gA[t-(t&3)]),mask);
			if(copy_sourceB!=0) atomicXor((unsigned int*)&(RdNew[it_j].gB[t-(t&3)]),mask);
		#elif MSbits == 16
			mask = 0x1; mask <<= (it_k + ((t&1)<<4));
			if(copy_sourceA!=0) atomicXor((unsigned int*)&(RdNew[it_j].gA[t-(t&1)]),mask);
			if(copy_sourceB!=0) atomicXor((unsigned int*)&(RdNew[it_j].gB[t-(t&1)]),mask);
		#elif MSbits == 32
			mask = 0x1; mask <<= it_k;
			if(copy_sourceA!=0) atomicXor((unsigned int*)&(RdNew[it_j].gA[t]),mask);
			if(copy_sourceB!=0) atomicXor((unsigned int*)&(RdNew[it_j].gB[t]),mask);
		#elif MSbits == 64
			mask = 0x1; mask <<= it_k;
			if(copy_sourceA!=0) atomicXor((unsigned long long int*)&(RdNew[it_j].gA[t]),mask);
			if(copy_sourceB!=0) atomicXor((unsigned long long int*)&(RdNew[it_j].gB[t]),mask);
		#endif
		if(t==0) 	RdNew[it_j].isActive[it_k] = true;
		else if(t==1)	RdNew[it_j].IE[it_k] = Rd[bx].IE[by];
	}	
}

__global__ void CalcAverages(Replica* Repd, int rg, double* Averages) // calculation of observables via averaging over the population
{
	int t = threadIdx.x, b = blockIdx.x, by = blockIdx.y; int idx = t + Nthreads * b;
	double currE,currE2,currM,currM2,currM4;
	if(idx<rg) if(Repd[idx].isActive[by]){
		currE = Repd[idx].IE[by]; currM = Repd[idx].M[by]; if(currM<0) currM=-currM;
	} else{ currE = 0; currM = 0;} else{ currE = 0; currM = 0;}
	currE2 = currE*currE; currM2 = currM*currM; currM4 = currM2*currM2;
	currE  = blockReduceSum<double>(currE);	 if(t==0) atomicAdd(&Averages[0], currE);  __syncthreads();
	currE2 = blockReduceSum<double>(currE2); if(t==0) atomicAdd(&Averages[1], currE2); __syncthreads();
	currM  = blockReduceSum<double>(currM);	 if(t==0) atomicAdd(&Averages[2], currM);  __syncthreads();
	currM2 = blockReduceSum<double>(currM2); if(t==0) atomicAdd(&Averages[3], currM2); __syncthreads();
	currM4 = blockReduceSum<double>(currM4); if(t==0) atomicAdd(&Averages[4], currM4);
}

#ifdef MHR

__global__ void UpdateShistE(Replica* Repd, int rg, int* ShistE) // adding energy histogram of the current temperature step for the MHR analysis
{
	int t = threadIdx.x, b = blockIdx.x, by = blockIdx.y; int idx = t + Nthreads * b;
	if(idx<rg) if(Repd[idx].isActive[by]){
		atomicAdd(&ShistE[(2*N+Repd[idx].IE[by])/4],1);
	}
}

#endif

#ifdef AdaptiveStep

__global__ void HistogramOverlap(Replica* Repd, int Rinit, int R, int rg, double lnQ, double dB, double* overlap) // calculating histogram overlap
{
	double PartialOverlap;
	int t = threadIdx.x, idx = threadIdx.x + Nthreads * blockIdx.x, by = blockIdx.y;
	if(idx<rg && Repd[idx].isActive[by]) 
		PartialOverlap = min(1.0,((double)Rinit)/R*exp(-dB*(double)Repd[idx].IE[by] - lnQ));
	else PartialOverlap = 0;
	PartialOverlap = blockReduceSum<double>(PartialOverlap);
	if(t==0) atomicAdd(overlap,PartialOverlap);
}

double CalcOverlap(Replica* Rep_d, double dB, int R, double Emean){	// Calculates histogram overlap
		double q, lnQ, ioverlaph;
		int rg = (int)ceil(R/(float)MSbits); 
		int NblocksR = (int)ceil(rg/(double)Nthreads);
		dim3 DimGridR(NblocksR,MSbits,1);
		QKer <<< rg, MSbits >>> (Rep_d, rg, dB, Emean, 0, Qd);
		CUDAErrChk( cudaPeekAtLastError() );
		CUDAErrChk( cudaDeviceSynchronize() );
		QKer <<< NblocksR, Nthreads >>> (Rep_d, rg, dB, Emean, 1, Qd);
		CUDAErrChk( cudaPeekAtLastError() );
		CUDAErrChk( cudaDeviceSynchronize() );
		QKer <<< 1, Nthreads >>> (Rep_d, rg, dB, Emean, 2, Qd);
		CUDAErrChk( cudaPeekAtLastError() );
		CUDAErrChk( cudaDeviceSynchronize() );
		CUDAErrChk( cudaMemcpy(&q,Qd,sizeof(double),cudaMemcpyDeviceToHost) );
		lnQ = -dB * Emean + log(q) - log((double)R);
		CUDAErrChk( cudaMemset(ioverlapd, 0, sizeof(double)) );
		HistogramOverlap<<<DimGridR,Nthreads>>>(Rep_d, Rinit, R, rg, lnQ, dB, ioverlapd);
		CUDAErrChk( cudaPeekAtLastError() );
		CUDAErrChk( cudaDeviceSynchronize() );
		CUDAErrChk( cudaMemcpy(&ioverlaph,ioverlapd,sizeof(double),cudaMemcpyDeviceToHost) );
		return (double)ioverlaph/R;
}

#endif

char *optarg; int opterr = 1, optind = 1, optopt, optreset;

int getopt(int nargc, char * const nargv[], const char *ostr)
{
	static char *place = (char*)""; const char *oli;
	if (optreset || !*place) { 
		optreset = 0;
		if (optind >= nargc || *(place = nargv[optind]) != '-') { place = (char*)""; return (-1); }
		if (place[1] && *++place == '-') { ++optind; place = (char*)""; return (-1); }
	}
	if ((optopt = (int)*place++) == (int)':' || !(oli = strchr(ostr, optopt))) {
		if (optopt == (int)'-') return (-1);
		if (!*place) ++optind;
		if (opterr && *ostr != ':') (void)printf("illegal option -- %c\n", optopt);
		return ((int)'?');
	}
	if (*++oli != ':') { optarg = NULL; if (!*place) ++optind; }
	else {
		if (*place) optarg = place; else if (nargc <= ++optind) { 
			place = (char*)""; if (*ostr == ':') return ((int)':');
			if (opterr) (void)printf("option requires an argument -- %c\n", optopt);
			return ((int)'?');
		}
		else optarg = nargv[optind];
		place = (char*)"";  ++optind;
	}
	return (optopt);
}

void PrintParameterUsage(){
	   cout << " Usage: PAisingMSC [options]\n"
		<< " Note: all of the options are optional. Default parameter values are listed in the head of the source code. \n"
		<< " Possible command line options are:\n\n"
		<< " -R Rinit           ( Rinit           = initial size of population of replicas )\n"
		<< " -t EQsweeps        ( EQsweeps        = number of equilibration sweeps )\n"
		<< " -d dBinit          ( dBinit          = inverse temperature step )\n"
		<< " -f Bfin            ( Bfin            = final value of inverse temperature )\n"
		<< " -M runs            ( runs            = number of population annealing algorithm independent runs )\n"
		<< " -s RNGseed         ( RNGseed         = seed for random number generation )\n"
		<< " -P OutputPrecision ( OutputPrecision = precision (number of digits) of the output )\n"
		<< " -o dataDirectory   ( dataDirectory   = data directory name )\n";
}

int main(int argc, char** argv)
{
	// data directory name + create
	char dataDir[200]; unsigned long long rng_seed = RNGseed; int optdir = 0;

	int optc, opti; double optf;
	while ((optc = getopt (argc, argv, "R:t:d:f:M:s:P:o:?")) != -1)	// Processing optional command line options
		switch (optc)
		{
			case 'R': opti = atoi(optarg); if(opti) Rinit = opti; break;           		// -R Rinit
			case 't': opti = atoi(optarg); EQsweeps = opti; break;                 		// -t EQsweeps
			case 'd': optf = atof(optarg); if(optf > 0.0) dBinit = optf; break;     	// -d dBinit
			case 'f': optf = atof(optarg); if(optf > 0.0) Bfin = optf; break;       	// -f Bfin
			case 'M': opti = atoi(optarg); if(opti) runs = opti; break;             	// -M runs
			case 's': opti = atoi(optarg); if(opti) rng_seed = opti; break;         	// -s RNGseed
			case 'P': opti = atoi(optarg); if(opti) OutputPrecision = opti; break;   	// -P OutputPrecision
			case 'o': if(optarg[strlen(optarg)-1]=='/') sprintf(dataDir,"%s",optarg);	// -o dataDir
				  else sprintf(dataDir,"%s/",optarg); optdir = 1; break;
			case '?': PrintParameterUsage();  return 1;
		}
	if(optind < argc){
		for (opti = optind; opti < argc; opti++) fprintf(stderr,"Non-option argument %s\n", argv[opti]);
		return 1;
	}

	#ifdef AdaptiveStep
		if(!optdir) sprintf(dataDir, "./dataMSC_L%d_R%d_EqSw%d/", L, Rinit, EQsweeps);
	#else
		if(!optdir) sprintf(dataDir, "./dataMSC_L%d_R%d_EqSw%d_dB%f/", L, Rinit, EQsweeps, dBinit);
	#endif

	#if defined(_WIN32)
		_mkdir(dataDir);
	#else 
		mkdir(dataDir, 0777);
	#endif
	
	int rmin=0, rmax=runs-1; unsigned long long initial_sequence = 0; int rg;

	double B[nBmax], Binc[nBmax]; B[0]=Binc[0]=Binit; double totPop=0;

	// creating data arrays for thermodynamic variables and errors
	double E[nBmax]; double M[nBmax]; double M2[nBmax]; double M4[nBmax];
	double C[nBmax];
	double lnQ[nBmax]; 			// partition function ratio
	double S[nBmax]; 			// entropy
	double BF[nBmax]; 			// dimensionless free energy estimate
	BF[0] = - N*log(2.0);			// its value at infinite temperature
	int R[nBmax];				// population size
	int nB;

	// CUDAErrChk( cudaSetDevice(0) );  // uncomment to explicitly select device number in a setup with multiple cards
	CUDAErrChk(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1)); // prefer larger L1 cache and smaller shared memory
	// GPU execution time
	cudaEvent_t start, stop; float Etime;
	CUDAErrChk( cudaEventCreate(&start) );
	CUDAErrChk( cudaEventCreate(&stop) );
	// start evaluation time measurement
	cudaEventRecord(start, 0);

	double *Averages; double Averages_h[5]; int* Ridev;
	CUDAErrChk( cudaMalloc((void**)&Averages,5*sizeof(double)) );
	CUDAErrChk( cudaMalloc((void**)&Qd,sizeof(double)) );
	CUDAErrChk( cudaMalloc((void**)&Ridev,sizeof(int)) );
	CUDAErrChk( cudaMalloc((void**)&ioverlapd,sizeof(double)) );

	// random seed
	cout <<"RNG initial seed: "<< rng_seed<<"\n";
	
	R[0] = Rinit;
	cout << "Memory use of one replica: " << sizeof(Replica) / 1024.0 / (double)MSbits << " kB \n";
	cout << "Memory use of the entire population of " << R[0] << " replicas: "
		<< ceil(R[0]/(double)MSbits)*sizeof(Replica) / 1024.0 / 1024.0 << " MB \n"; fflush(stdout);
	
	// creating energy spectrum for multi-histogram reweighting
	#ifdef MHR
		int Ei[N+1];
		for (int i = 0; i < N+1; ++i){
			Ei[i] = 4*i - 2*N;
		}
	#endif
	
	Replica* Rep_d;
	
	unsigned int boltzGPU[boltzTableL]; // Boltzman factor table - host version
	unsigned int* boltztext;
	
	// memory allocation for Boltzmann factor table
	CUDAErrChk( cudaMalloc((void **)&boltztext, boltzTableL * sizeof(unsigned int)) );
	// binding references (global & texture memory buffers)
	CUDAErrChk( cudaBindTexture(NULL,boltzT,boltztext,boltzTableL * sizeof(unsigned int)) );
	
	int Ethreads = 1; while(Ethreads < EQthreads) Ethreads <<= 1;

	for (int r = rmin; r <= rmax; ++r){	
		
		rg = (int)ceil(R[0]/(float)MSbits);		// number of replica groups (R / MSbits)
		double sumlnQ = 0.0; double q; double Emean = 0.0;
		CUDAErrChk( cudaMalloc((void **)&Rep_d,rg*sizeof(Replica)) );
		int NblocksR = (int)ceil(rg/(float)Nthreads);

		ReplicaInit <<< rg, EQthreads >>> (Rep_d,rg,R[0],rng_seed,initial_sequence); initial_sequence+=rg*EQthreads;
		CUDAErrChk( cudaPeekAtLastError() );
		CUDAErrChk( cudaDeviceSynchronize() );
		
		// compute energy of all replicas at zero temperature (for 1st resampling)
		energyKer <<< rg, Ethreads >>> (Rep_d);
		CUDAErrChk( cudaPeekAtLastError() );
		CUDAErrChk( cudaDeviceSynchronize() );
		
		// array for summing the energy histograms over inverse temperatures
		#ifdef MHR
			int ShistE[N+1]; int* ShistEd;
			CUDAErrChk( cudaMalloc((void**)&ShistEd,(N+1)*sizeof(int)) );
			CUDAErrChk( cudaMemset(ShistEd,0,(N+1)*sizeof(int)) );
			dim3 DimGridR(NblocksR,MSbits,1);
			UpdateShistE<<<DimGridR,Nthreads>>> (Rep_d, rg, ShistEd);
			CUDAErrChk( cudaPeekAtLastError() );
			CUDAErrChk( cudaDeviceSynchronize() );
		#endif
		
		// ------------------------------------------------------------------
		// population annealing
		// ------------------------------------------------------------------
		int i=1, iprev=0; double deltaBeta=dBinit; B[i]=Binc[i]=B[iprev]+deltaBeta;
		
		while(B[i]<=Bfin) {
			// Boltzmann factor tabulation (only two are relevant: exp(-4*B);exp(-8*B))
			boltzGPU[0] = ceil(4294967296.*exp(-4*B[i]));
			boltzGPU[1] = ceil(4294967296.*exp(-8*B[i]));

			// copying table to texture memory - boltztext is bounded with boltzT 
			CUDAErrChk( cudaMemcpy(boltztext, boltzGPU, boltzTableL * sizeof(unsigned int),cudaMemcpyHostToDevice) );
			
			// compute the partition function ratio - Q
			NblocksR = (int)ceil(rg/(float)Nthreads);
			
			dim3 DimGridR(NblocksR,MSbits,1);

			QKer <<< rg, MSbits >>> (Rep_d, rg, B[i] - B[i-1], Emean, 0, Qd);
			CUDAErrChk( cudaPeekAtLastError() );
			CUDAErrChk( cudaDeviceSynchronize() );
			QKer <<< NblocksR, Nthreads >>> (Rep_d, rg, B[i] - B[i-1], Emean, 1, Qd);
			CUDAErrChk( cudaPeekAtLastError() );
			CUDAErrChk( cudaDeviceSynchronize() );
			QKer <<< 1, Nthreads >>> (Rep_d, rg, B[i] - B[i-1], Emean, 2, Qd);
			CUDAErrChk( cudaPeekAtLastError() );
			CUDAErrChk( cudaDeviceSynchronize() );
			CUDAErrChk( cudaMemcpy(&q,Qd,sizeof(double),cudaMemcpyDeviceToHost) );

			lnQ[i] = -(B[i] - B[i-1])*Emean + log(q) -log((double)R[i-1]);
			
			CalcTauKer <<< DimGridR, Nthreads >>> (Rep_d, Rinit, R[i-1], rg, lnQ[i], B[i] - B[i-1],rng_seed,initial_sequence); initial_sequence+=rg*MSbits;
			CUDAErrChk( cudaPeekAtLastError() );
			CUDAErrChk( cudaDeviceSynchronize() );

			// resampling new population
			CalcParSum <<< rg,    MSbits   >>> (Rep_d, rg, 0, Ridev);
			CUDAErrChk( cudaPeekAtLastError() );
			CUDAErrChk( cudaDeviceSynchronize() );

			CalcParSum <<< NblocksR, Nthreads >>> (Rep_d, rg, 1, Ridev);
			CUDAErrChk( cudaPeekAtLastError() );
			CUDAErrChk( cudaDeviceSynchronize() );

			CalcParSum <<< DimGridR, Nthreads >>> (Rep_d, rg, 2, Ridev);
			CUDAErrChk( cudaPeekAtLastError() );
			CUDAErrChk( cudaDeviceSynchronize() );
			CUDAErrChk( cudaMemcpy(&R[i], Ridev, sizeof(int),cudaMemcpyDeviceToHost) );

			dim3 DimGridRes(rg,MSbits,N/2/EQthreads);        // resampleKer configuration with old value of rg
			rg = (int)ceil(R[i]/(float)MSbits);		// updated number of replica groups
			Replica* RepNew_d;
			CUDAErrChk( cudaMalloc((void**)&RepNew_d,rg*sizeof(Replica)) );
			CUDAErrChk( cudaMemset(RepNew_d,0,rg*sizeof(Replica)) );
			CUDAErrChk( cudaDeviceSynchronize() );

			resampleKer <<< DimGridRes, EQthreads >>> (Rep_d, RepNew_d, rg);
			CUDAErrChk( cudaPeekAtLastError() );
			CUDAErrChk( cudaDeviceSynchronize() );
				
			Replica* RepDel = Rep_d;
			Rep_d = RepNew_d;
			CUDAErrChk( cudaFree(RepDel) );

			// equilibrate replicas for certain number of sweeps
			checkKerALL <<< rg, EQthreads >>> (Rep_d,rg,EQsweeps,rng_seed,initial_sequence); initial_sequence+=rg*EQthreads;
			CUDAErrChk( cudaPeekAtLastError() ); 
			CUDAErrChk( cudaDeviceSynchronize() );

			// compute observables (E,M,O,F)
			// compute energy and magnetization of all replicas
			energyKer <<< rg, Ethreads >>> (Rep_d);
			CUDAErrChk( cudaPeekAtLastError() );
			CUDAErrChk( cudaDeviceSynchronize() );

			// saving results - energies
			#ifdef EnergiesPopStore
				Replica* Rep_h = (Replica*)malloc(rg*sizeof(Replica));
				CUDAErrChk( cudaMemcpy(Rep_h, Rep_d, rg*sizeof(Replica),cudaMemcpyDeviceToHost) );
				ofstream results;
				char str[100];
				char str2[100];
				strcpy(str, dataDir);
				sprintf(str2,"PA_energies_%d.dat",i);
				strcat(str,str2);
				results.open(str);
				results.precision(OutputPrecision);
				for (int j = 0; j < rg; ++j)
					for (int l = 0; l < MSbits; ++l)
						if(Rep_h[j].isActive[l]) results << Rep_h[j].IE[l] << " ";
				results.close(); free(Rep_h);
			#endif

			#ifdef MHR
				UpdateShistE<<<DimGridR,Nthreads>>>(Rep_d, rg, ShistEd);
				CUDAErrChk( cudaPeekAtLastError() );
				CUDAErrChk( cudaDeviceSynchronize() );
			#endif

			CUDAErrChk( cudaMemset(Averages, 0, 5*sizeof(double)) );
			CalcAverages<<<DimGridR,Nthreads>>>(Rep_d,rg,Averages);
			CUDAErrChk( cudaPeekAtLastError() );
			CUDAErrChk( cudaDeviceSynchronize() );
			CUDAErrChk( cudaMemcpy(Averages_h,Averages,5*sizeof(double),cudaMemcpyDeviceToHost) );

			E[i]  = Emean = Averages_h[0] / R[i];
			C[i]  = (Averages_h[1] / R[i] - E[i]*E[i]) * B[i] * B[i];
			M[i]  = Averages_h[2] / R[i];
			M2[i] = Averages_h[3] / R[i];
			M4[i] = Averages_h[4] / R[i];

			// dimensionless free energy
			sumlnQ -= lnQ[i];
			BF[i] = - N*log(2.0) + sumlnQ;
			// entropy
			S[i] = B[i]*E[i] - BF[i];

	                iprev=i; totPop+=R[i]; i++; 

			if(i>=nBmax){
				#ifdef AdaptiveStep
					fprintf(stderr,"Error: number of temperature steps exceeds nBmax=%d.\n Please consider increasing the population size or decreasing the value of MinOverlap or increasing the value of nBmax.\n",nBmax);
				#else
					fprintf(stderr,"Error: number of temperature steps exceeds nBmax=%d.\n Please consider increasing the inverse temperature step or increasing the value of nBmax.\n",nBmax);
				#endif
				return 1;
			}

			if (r==rmin){
				#ifdef AdaptiveStep
					double overlap, dBmin = 0, dBmax = deltaBeta, dBmean;
					while(1){
						overlap = CalcOverlap ( Rep_d, dBmax, R[iprev], Emean );
						if ( (overlap >= MaxOverlap) && (B[iprev] + dBmax < Bfin) ) dBmax *= 1.1; else break;
					}
					if ( overlap >= MinOverlap ) dBmean = dBmax; 
					else while(1){	// obtaining optimal inverse temperature step with the bisection method
						dBmean = 0.5 * (dBmin + dBmax);
						overlap = CalcOverlap ( Rep_d, dBmean, R[iprev], Emean );
						if ( overlap < MinOverlap ) dBmax = dBmean;
						else if ( overlap >= MaxOverlap ) dBmin = dBmean;
						else break;
					}
					if( (B[iprev] < Bfin) && (B[iprev] + dBmean > Bfin) ) deltaBeta = Bfin - B[iprev]; else deltaBeta = dBmean;
				#endif
				B[i] = Binc[i] = B[iprev] + deltaBeta;
			} else B[i]=Binc[i];
		}
		
		CUDAErrChk( cudaFree(Rep_d) );
		nB=i;
		
		// saving results
		{
			ofstream results;
			char str[100];
			char str2[100];
			strcpy(str, dataDir);
			sprintf(str2, "PA_results_run_%d.dat", r);
			strcat(str,str2);
			results.open(str);
			results.precision(OutputPrecision);
			for (int i = 0; i < nB; ++i) {
				results << B[i] << " "
					<< E[i] / N << " "
					<< C[i] / N << " "
					<< M[i] / N << " "
					<< M2[i] / N / N << " "
					<< M4[i] / N / N / N / N << " "
					<< BF[i] / N << " "
					<< S[i] / N << " "
					<< R[i] << " "
					<< lnQ[i] << "\n";
			}
			results.close();
		}
		
		// multi-histogam reweighting (MHR) analysis
		#ifdef MHR
			// declaring arrays used in MHR analysis
			double lnOmega[N+1];
			double E_MHR[nB*MHR_Niter];
			double C_MHR[nB*MHR_Niter];
			double BF_MHR[nB*MHR_Niter];

			bool relTerm[N+1];

			CUDAErrChk( cudaMemcpy(ShistE,ShistEd,(N+1)*sizeof(int),cudaMemcpyDeviceToHost) );

			for (int l = 0; l < MHR_Niter; ++l){
				// calculate lnOmega
				double Sigma[nB];
				double mSigma;
				for (int k = 0; k < N+1; ++k){				
					// maxima of -S = BF - B*E
					Sigma[0] = BF[0]-B[0]*Ei[k];
					mSigma = Sigma[0];
					for (int i = 1; i < nB; ++i){
						Sigma[i] = BF[i]-B[i]*Ei[k];
						if (mSigma < Sigma[i]){
							mSigma = Sigma[i];
						}
					}
					double sD = 0;
					for (int i = 0; i < nB; ++i){
						sD += R[i]*exp(Sigma[i]-mSigma);
					}
					if ((ShistE[k] == 0) || (sD == 0)){
						relTerm[k] = false;
						lnOmega[k] = 0;
					} else {
						relTerm[k] = true;
						lnOmega[k] = log(ShistE[k]) - mSigma - log(sD);
					}
				}
				// reweigting of observables
				double expOm[N+1];
				double Om[N+1];
				double mOm;
				for (int i = 0; i < nB; ++i){
					// determine the maxima of the reweighting exponent
					mOm = lnOmega[0] - B[i]*Ei[0];
					for (int k = 0; k < N+1; ++k){
						Om[k] = lnOmega[k] - B[i]*Ei[k];
						if (mOm < Om[k]){
							mOm = Om[k];
						}
					}
					// calculate reweighting exponentials
					double p = 0;
					for (int k = 0; k < N+1; ++k){
						expOm[k] = exp(Om[k] - mOm);
						if (relTerm[k])
							p += expOm[k];
					}
					double s = 0; 
					for (int k = 0; k < N+1; ++k){
						if (relTerm[k])
							s += Ei[k]*expOm[k];
					}
					E_MHR[i+l*nB] = s / p / N;
					BF_MHR[i+l*nB] = - mOm - log(p);
					BF[i] = BF_MHR[i+l*nB];
					s = 0;
					for (int k = 0; k < N+1; ++k){
						if (relTerm[k])
							s += pow(Ei[k]-E_MHR[i+l*nB]*N,2)*expOm[k];
					}
					C_MHR[i+l*nB] = B[i]*B[i] * s / p / N;
				}
			}
			// saving results
			{
				ofstream results;
				char MHRDataFile[100];
				char str2[100];
				strcpy(MHRDataFile, dataDir);
				sprintf(str2,"PA_MHR_results_run_%d.dat",r);
				strcat(MHRDataFile,str2);
				results.open(MHRDataFile);
				results.precision(OutputPrecision);
				for (int i = 0; i < nB; ++i){
					results << B[i] << " ";
					for (int l = 0; l < MHR_Niter; ++l){
						results << E_MHR[i+l*nB] << " ";
						results << C_MHR[i+l*nB] << " ";
						results << BF_MHR[i+l*nB] / N << " ";
					}
					results << "\n";
				}
				results.close();
			}

			CUDAErrChk( cudaFree(ShistEd) );

		#endif
	}

	CUDAErrChk( cudaFree(Averages) );
	CUDAErrChk( cudaFree(Ridev) );
	CUDAErrChk( cudaFree(Qd) );	
	CUDAErrChk( cudaFree(ioverlapd) );
	CUDAErrChk( cudaUnbindTexture(boltzT) );
	CUDAErrChk( cudaFree(boltztext));

	CUDAErrChk( cudaThreadSynchronize() );
	CUDAErrChk( cudaEventRecord(stop, 0) );
	CUDAErrChk( cudaEventSynchronize(stop) );	
	CUDAErrChk( cudaEventElapsedTime(&Etime, start, stop) );
	cout << "Elapsed time: " << setprecision(8) << Etime/1000 << " s\n";
	cout << "Time per spin-flip: " << setprecision(8) << Etime*1e6/EQsweeps/N/totPop << " ns\n";

	CUDAErrChk( cudaEventDestroy(start) );
	CUDAErrChk( cudaEventDestroy(stop) );

	return 0;
}
