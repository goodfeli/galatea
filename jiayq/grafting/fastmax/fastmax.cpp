#include <emmintrin.h>
#include <stdint.h>
#define max(a,b) ((a)>(b)? (a):(b))
#define ALIGNMENT_VALUE 16u
extern "C" {

int normalizev(double * a, int n, double mean, double std)
{
    int i;
    register __m128d xmma __asm__("xmm2"),
             xmmm __asm__("xmm3"),
             xmms __asm__("xmm4");
    
    /* we'll use addition and multiplication instead. */
    mean = -mean;
    std = 1.0/std;

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
    double* Mpointer = M + rows[0]*ncols;
    for (i = 0; i < ncols; i ++)
    {
        out[i] = Mpointer[i];
    }

    for (i = 1; i < nrows; i ++)
    {
        fastmaxv(out, M + rows[i]*ncols, ncols);
    }
    return 0;
}

} // extern C
