#include <emmintrin.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#define max(a,b) ((a)>(b)? (a):(b))
#define sum(a,b) ((a)+(b))
#define ALIGNMENT_VALUE 16u
extern "C" {

int fastmeanstd(double *a, int n, double * pmeanstd)
{
    int i;
    register __m128d xmma __asm__("xmm2"),
             xmmm __asm__("xmm3"),
             xmms __asm__("xmm4");

    double meantemp[2] = {0.0,0.0};
    double stdtemp[2] = {0.0,0.0};
    int nsse = n; 
    if ((uintptr_t)a % ALIGNMENT_VALUE)
    {
        meantemp[0] = a[0];
        stdtemp[0] = a[0]*a[0];
        a ++; nsse --;
    }
    xmmm = _mm_load_pd(meantemp);
    xmms = _mm_load_pd(stdtemp);
    for (i = 0; i < nsse-1; i += 2)
    {
        xmma = _mm_load_pd(a);
        xmmm = _mm_add_pd(xmmm,xmma);
        xmms = _mm_add_pd(xmms,_mm_mul_pd(xmma,xmma));
        a += 2;
    }
    _mm_store_pd(meantemp, xmmm);
    _mm_store_pd(stdtemp, xmms);
    
    if (nsse % 2)
    {
        meantemp[0] += meantemp[1]+a[0];
        stdtemp[0] += stdtemp[1]+a[0]*a[0];
    }
    else
    {
        meantemp[0] += meantemp[1];
        stdtemp[0] += stdtemp[1];
    }
    // compute the final number
    pmeanstd[0] = meantemp[0] / n;
    pmeanstd[1] = sqrt( (stdtemp[0]/n) - pmeanstd[0]*pmeanstd[0]);

    return 0;
}

int normalizev(double * a, int n, double mean, double std)
{
    int i;
    register __m128d xmma __asm__("xmm2"),
             xmmm __asm__("xmm3"),
             xmms __asm__("xmm4");
    
    /* we'll use addition and multiplication instead. */
    mean = -mean;
    if (std < 1e-10) {
        /* In this case, do not normalize */
        std = 1.0;
    } else {
        std = 1.0/std;
    }
    if ((uintptr_t)a % ALIGNMENT_VALUE)
    {
        a[0] = (a[0] + mean) * std;
        a ++; n --;
    }

    xmmm = _mm_load1_pd(&mean);
    xmms = _mm_load1_pd(&std);

    for (i = 0; i < n-1; i += 2)
    {
        xmma = _mm_load_pd(a);
        _mm_store_pd(a, _mm_mul_pd(_mm_add_pd(xmma,xmmm),xmms));
        a += 2;
    }
    if (n%2)
    {
        /* the last element */
        a[0] = (a[0] + mean) * std;
    }
    return 0;
}

inline int fastmaxv(double * a, double * b, int n)
{
    /* This is the algorithm that does fast max computation */
    int i;
    register __m128d xmma __asm__("xmm2"), 
             xmmb __asm__("xmm3");

    if ((uintptr_t)a % ALIGNMENT_VALUE)
    {
        /* a is misaligned. We will start using SSE from a[1]. 
         * In this way, we can use _mm_store_pd which is faster*/
        a[0] = max(a[0],b[0]);
        a ++; b ++; n --;
    }

    if ((uintptr_t)b % ALIGNMENT_VALUE) {
        /* pb is misaligned. use _mm_loadu_pd for b */
        for (i = 0; i < n-1; i += 2)
        {
            xmma = _mm_load_pd(a);
            xmmb = _mm_loadu_pd(b);
            _mm_store_pd(a, _mm_max_pd(xmma,xmmb));
            a += 2;
            b += 2;
        }
    } else {
        for (i = 0; i < n-1; i += 2)
        {
            xmma = _mm_load_pd(a);
            xmmb = _mm_load_pd(b);
            _mm_store_pd(a, _mm_max_pd(xmma,xmmb));
            a += 2;
            b += 2;
        }
    }
    if (n%2)
    {
        /* the last element */
        a[0] = max(a[0],b[0]);
    }
    return 0;
}


int fastmaxm(double* out, double* M, int* rows, int nrows, int ncols)
{
    int i;
    int currid = 0;
    double* Mpointer = NULL;
    
    if (nrows < 0) {
    	// in this case, nrows is the size of the matrix
    	// and rows is the 0-1 indicator function
    	while (rows[currid] == 0) {
    		currid ++;
    	}
    	Mpointer = M + currid*ncols;
    	for (i = 0; i < ncols; i ++)
    	{
        	out[i] = Mpointer[i];
    	}
    	currid ++;
    	for (i = currid; i < -nrows; i ++)
    	{
    		if (rows[i] > 0) {
    			fastmaxv(out, M+i*ncols, ncols);
    		}
    	}
    }
    else
    {
        Mpointer = M + rows[0]*ncols;
    	for (i = 0; i < ncols; i ++)
    	{
        	out[i] = Mpointer[i];
    	}
   	    for (i = 1; i < nrows; i ++)
    	{
        	fastmaxv(out, M + rows[i]*ncols, ncols);
    	}
    }
    
    return 0;
}

inline int fastsumv(double * a, double * b, int n)
{
    /* This is the algorithm that does fast sum computation */
    int i;
    register __m128d xmma __asm__("xmm2"), 
             xmmb __asm__("xmm3");

    if ((uintptr_t)a % ALIGNMENT_VALUE)
    {
        /* a is misaligned. We will start using SSE from a[1]. 
         * In this way, we can use _mm_store_pd which is faster*/
        a[0] = sum(a[0],b[0]);
        a ++; b ++; n --;
    }

    if ((uintptr_t)b % ALIGNMENT_VALUE) {
        /* pb is misaligned. use _mm_loadu_pd for b */
        for (i = 0; i < n-1; i += 2)
        {
            xmma = _mm_load_pd(a);
            xmmb = _mm_loadu_pd(b);
            _mm_store_pd(a, _mm_add_pd(xmma,xmmb));
            a += 2;
            b += 2;
        }
    } else {
        for (i = 0; i < n-1; i += 2)
        {
            xmma = _mm_load_pd(a);
            xmmb = _mm_load_pd(b);
            _mm_store_pd(a, _mm_add_pd(xmma,xmmb));
            a += 2;
            b += 2;
        }
    }
    if (n%2)
    {
        /* the last element */
        a[0] = sum(a[0],b[0]);
    }
    return 0;
}


int fastsumm(double* out, double* M, int* rows, int nrows, int ncols)
{
    int i;
    int currid = 0;
    double* Mpointer = NULL;
    
    if (nrows < 0) {
    	// in this case, nrows is the size of the matrix
    	// and rows is the 0-1 indicator function
    	while (rows[currid] == 0) {
    		currid ++;
    	}
    	Mpointer = M + currid*ncols;
    	for (i = 0; i < ncols; i ++)
    	{
        	out[i] = Mpointer[i];
    	}
    	currid ++;
    	for (i = currid; i < -nrows; i ++)
    	{
    		if (rows[i] > 0) {
    			fastsumv(out, M+i*ncols, ncols);
    		}
    	}
    }
    else
    {
        Mpointer = M + rows[0]*ncols;
    	for (i = 0; i < ncols; i ++)
    	{
        	out[i] = Mpointer[i];
    	}
   	    for (i = 1; i < nrows; i ++)
    	{
        	fastsumv(out, M + rows[i]*ncols, ncols);
    	}
    }
    
    return 0;
}
int fastcenters(double* M, double* C, int* Ccounts, int* idx, int nrows, int ncols, int nctrs)
{
    int i;
    memset(C, 0, sizeof(double)*ncols*nctrs);
    memset(Ccounts, 0, sizeof(int)*nctrs);
    for (i = 0; i < nrows; i ++)
    {
        Ccounts[idx[i]] ++;
        fastsumv(C + idx[i]*ncols, M + i*ncols, ncols);
    }
    // divide by the number of members to get the centers
    for (i = 0; i < nctrs; i ++)
    {
        // if a cluster has no member, max() suppresses numerical errors but does not
        // introduce inaccuracy.
        normalizev(C+i*ncols, ncols, 0.0, max(Ccounts[i],1.0));
    }
    return 0;
}

int fastmaximums(double* M, double* C, int* Ccounts, int* idx, int nrows, int ncols, int nctrs)
{
    int i;
    memset(C, 0, sizeof(double)*ncols*nctrs);
    memset(Ccounts, 0, sizeof(int)*nctrs);
    for (i = 0; i < nrows; i ++)
    {
        Ccounts[idx[i]] ++;
        fastmaxv(C + idx[i]*ncols, M + i*ncols, ncols);
    }
    return 0;
}

} // extern C

