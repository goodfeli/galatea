/* use the following lines to use AMD functions
 *
 * #define REPLACE_WITH_AMDLIBM
 * #include "amdlibm/include/amdlibm.h" */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <emmintrin.h>
#define EXP_MAX 100.0
#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))
#define ALIGNMENT_VALUE 16u

void exp_safe(double * x, double * y, int n) 
{
    int i;
    for (i = 0; i < n; i ++) 
    {
        y[i] = exp(min(x[i],EXP_MAX));
    }
}

void log_xincr(double *x, double *y, int n)
{
    int i;
    for (i = 0; i < n; i ++)
    {
        y[i] = log(x[i]+1.0);
    }
}

void offset_minus_yf(double *x, double *y, double *out, double offset, int n)
{
    /* out = x*y+offset */
    int i;
    register __m128d xmmx __asm__("xmm2"),
             xmmy __asm__("xmm3"),
             xmmo __asm__("xmm4"),
             xmmf __asm__("xmm5");
    
    if ((uintptr_t)out % ALIGNMENT_VALUE)
    {
        out[0] = offset - x[0]*y[0];
        out++; x++; y++; n--;
    }
    xmmf = _mm_load1_pd(&offset);
    for (i = 0; i < n-1; i += 2)
    {
        xmmx = _mm_loadu_pd(x);
        xmmy = _mm_loadu_pd(y);
        xmmo = _mm_sub_pd(xmmf, _mm_mul_pd(xmmx,xmmy));
        _mm_store_pd(out, xmmo);
        x += 2;
        y += 2;
        out += 2;
    }
    if (n%2)
    {
        out[0] = offset - x[0]*y[0];
    }
}

void neg_ypu_div_uincr(double * y, double * u, double *out, int n) 
{
    /* compute out = -y*u / (1.0+u) */
    /* u and out can be the same to save memory */
    int i;
    const double negone = -1.0;
    register __m128d xmmy __asm__("xmm2"),
             xmmo __asm__("xmm3"),
             xmmnegop __asm__("xmm4"),
             xmmnegone __asm__("xmm5");
    if ((uintptr_t)out % ALIGNMENT_VALUE)
    {
        out[0] = y[0]*u[0] / (-1.0-u[0]);
        out++; u++; y++; n--;
    }
    xmmnegone = _mm_load1_pd(&negone);
    for (i = 0; i < n-1; i += 2)
    {
        xmmy = _mm_loadu_pd(y);
        xmmo = _mm_loadu_pd(u);
        xmmnegop = _mm_sub_pd(xmmnegone, xmmo);
        xmmo = _mm_mul_pd(xmmo,xmmy);
        xmmo = _mm_div_pd(xmmo, xmmnegop);
        _mm_store_pd(out, xmmo);
        out += 2;
        u += 2;
        y += 2;
    }
    if (n%2)
    {
        out[0] = y[0]*u[0] / (-1.0-out[0]);
    }
}

int gL_bnll(double * y, double * f, double * gL, int n) 
{
    /* compute gL = 1-y*f */
    offset_minus_yf(y,f,gL,1.0,n);
    /* compute exp(gL) */
    exp_safe(gL,gL,n);
    /* compute gL = -y*gL / (1.0+gL) */
    neg_ypu_div_uincr(y, gL, gL, n); 
    return 0;
}

int LgL_bnll(double *y, double *f, double *L, double * gL, int n)
{
    /* compute 1-y*f */
    offset_minus_yf(y,f,gL,1.0, n);
    /* compute exp(gL) */
    exp_safe(gL, gL, n);
    /* compute L first */
    log_xincr(gL, L, n);
    /* compute gL */
    neg_ypu_div_uincr(y, gL, gL, n); 
    return 0;
}
