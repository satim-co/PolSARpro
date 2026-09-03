#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    float theta;
    float hh;
    float vv;
    float hv;
} ProbeInput;

typedef struct {
    float a;
    float b;
    float c;
    float x;
} ProbeOutput;

static ProbeOutput apply_oh_probe(ProbeInput input, int iterations)
{
    float x = 2;
    float a = 2 * input.theta / M_PI;
    float b = (input.hv / input.vv) / 0.23;
    float c = sqrt(input.hh / input.vv) - 1;

    for (int tt = 0; tt < iterations; tt++) {
        x = x - ((exp((x * x / 3) * log(a)) * (1 - b * x) + c) /
                 (((2 * x / 3 * log(a) * (1 - b * x)) - b) *
                  exp((x * x / 3) * log(a))));
    }

    return (ProbeOutput){a, b, c, x};
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "usage: %s INPUT OUTPUT ITERATIONS\n", argv[0]);
        return EXIT_FAILURE;
    }

    FILE *input_file = fopen(argv[1], "rb");
    FILE *output_file = fopen(argv[2], "wb");
    int iterations = atoi(argv[3]);
    if (input_file == NULL || output_file == NULL || iterations < 0) {
        fprintf(stderr, "could not open files or invalid iteration count\n");
        return EXIT_FAILURE;
    }

    ProbeInput input;
    while (fread(&input, sizeof(input), 1, input_file) == 1) {
        ProbeOutput output = apply_oh_probe(input, iterations);
        if (fwrite(&output, sizeof(output), 1, output_file) != 1) {
            fprintf(stderr, "could not write output\n");
            return EXIT_FAILURE;
        }
    }

    fclose(input_file);
    fclose(output_file);
    return EXIT_SUCCESS;
}
