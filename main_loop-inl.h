#include <cmath>
#include <cstring>

#define MAX(a, b) ((a) > (b) ? (a) : (b))

static void quantize_row_q4_0_reference(const float * __restrict__ x, block_q4_0 * __restrict__ y, int k) {
    const int nb = k / QK;

    uint8_t pp[QK/2];

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int l = 0; l < QK; l++) {
            const float v = x[i*QK + l];
            amax = MAX(amax, fabsf(v));
        }

        const float d = amax / ((1 << 3) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = d;

        for (int l = 0; l < QK; l += 2) {
            const float v0 = x[i*QK + l + 0]*id;
            const float v1 = x[i*QK + l + 1]*id;

            const uint8_t vi0 = (int8_t)roundf(v0) + 8;
            const uint8_t vi1 = (int8_t)roundf(v1) + 8;

            pp[l/2] = vi0 | (vi1 << 4);
        }

        memcpy(y[i].qs, pp, sizeof(pp));
    }
}

void MAIN_FUNC_NAME(
    ggml_tensor* src0,
    ggml_tensor* src1,
    ggml_tensor* dst,
    char* wdata
) {
    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const int ne10 = src1->ne[0];
    const int ne11 = src1->ne[1];
    const int ne12 = src1->ne[2];
    const int ne13 = src1->ne[3];

    const int ne0  = dst->ne[0];
    const int ne1  = dst->ne[1];
    const int ne2  = dst->ne[2];
    const int ne3  = dst->ne[3];

    const int nb00 = src0->nb[0];
    const int nb01 = src0->nb[1];
    const int nb02 = src0->nb[2];
    const int nb03 = src0->nb[3];

    const int nb10 = src1->nb[0];
    const int nb11 = src1->nb[1];
    const int nb12 = src1->nb[2];
    const int nb13 = src1->nb[3];

    const int nb0  = dst->nb[0];
    const int nb1  = dst->nb[1];
    const int nb2  = dst->nb[2];
    const int nb3  = dst->nb[3];

    char* wd = wdata;
    const size_t row_size_q = ne10*20/32;
    for (int i11 = 0; i11 < ne11; ++i11) {
        quantize_row_q4_0_reference((float *)((char *) src1->data + i11*nb11), (block_q4_0 *) wd, ne10);
        wd += row_size_q;
    }

    // total rows in src0
    const int nr = ne01*ne02*ne03;
    const size_t row_size = ne00*20/32;

    for (int ir = 0; ir < nr; ++ir) {
        void * src0_row = (void *) ((char *) src0->data + (ir*nb01));
        char * src1_col =          (char *)      wdata;

        float * dst_col = (float *) ((char *) dst->data + (ir*nb0));

        for (int ic = 0; ic < ne11; ++ic) {
            ggml_vec_dot_q4_0(ne00, &dst_col[ic*ne0], src0_row, (void *) (src1_col + ic*row_size));
        }
    }
}
