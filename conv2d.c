#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

/* The string length of every float in the feature map. Example line: "0.594 0.934 0.212\n". 
So, 3 floats, each looks like "X.XXX" which is 5 chars, but then all have a space or new-line 
character. */
#define FLOAT_STRING_LENGTH 6

#define max(a,b) (((a) > (b)) ? (a) : (b))
#define min(a,b) (((a) < (b)) ? (a) : (b))

/*
~~~~~~~~~ IMPORTANT NOTES (for report + meetings) ~~~~~~~~~

- Normally, the kernel is flipped before calculation. In all provided examples, the expected 
    output occurs when the kernel is NOT flipped. This indicates that it is reasonable to assume
    that all input kernel data has already been flipped.

- To handle evenly sized kernels (e.g., 2x2, 4x6, etc.) we use asymmetric centering.

- Need to make sure we optimise for these things: 
    1. High cache-locality,
    2. Avoiding false sharing,
    3. Avoiding race conditions,
    4. Avoiding redundant calculations,
    5. Avoiding unnecessary memory allocations,
*/

/*
* Extracts the dimensions from a file.
* @param filepath     The filepath where the data is stored.
* @param height       Pointer to the location where the height will be stored.
* @param width        Pointer to the location where the width will be stored.
*/
int extract_dimensions(char* filepath, int* height, int* width) {

    if (filepath == NULL) { return 1; }
    char firstline[16];

    FILE* file_ptr = fopen(filepath, "r");
    if (file_ptr == NULL){
        return 1;
    }

    // Reads the first line
    fgets(firstline, sizeof(firstline), file_ptr);
    if (strcmp(firstline, "") == 0){
        return 2;
    }

    char* token = strtok(firstline, " ");
    *height = atoi(token);
    token = strtok(NULL, " ");
    *width = atoi(token);

    fclose(file_ptr);

    return 0;
}


/* 
* Reads an input file and extracts data into an output. 
* @param filepath         The filepath where the data is stored.
* @param width            The number of elements in each line. Width.
* @param height           The number of rows. Height.
* @param padding_width    The number of zeroes to pad the width with.
* @param padding_height   The number of zeroes to pad the height with.
* @param output           The stream into which the inputs will be stored.
*/
int extract_data(char* filepath, int width, int height, int padding_width, int padding_height, float** *output) {
    
    if (filepath == NULL){ return 1; }
    FILE* file_ptr = fopen(filepath, "r");
    if (file_ptr == NULL){ return 1; }

    // Create a buffer to place extracted strings into
    const size_t buffer_size = (FLOAT_STRING_LENGTH * width) + 2; // +2 for new-line and null-byte
    
    char* buffer = (char*)malloc(buffer_size);

    // get the header line here, so we can safely ignore it later
    fgets(buffer, buffer_size, file_ptr);

    // Now loop over each line in the file
    int row_index = 0;
    while (row_index < (height + padding_height)){
        
        if(row_index < padding_height){
            row_index++;
            continue;
        }

        fgets(buffer, buffer_size, file_ptr);

        if (buffer == NULL) {
            continue;
        }

        char* token = strtok(buffer, " ");

        // Now loop over each number in the line
        int column_index = padding_width;
        while (token != NULL){
            float element = (float)atof(token);
            (*output)[row_index][column_index] = element; // Add to output.
            token = strtok(NULL, " ");
            column_index++;
        }
        row_index++;
    }
    free(buffer);
    fclose(file_ptr);
    return 0;
}


/* 
* Performs serial 2D discrete convolutions. 
* @param f             Pointer to the Feature Map.
* @param H            Height of the Feature Map.
* @param W            Width of the Feature Map.
* @param g            Pointer to the Kernel.
* @param kH           Height of the Kernel.
* @param w_padding    Width of the padding in the Feature Map.
* @param h_padding    Height of the padding in the Feature Map.
* @param output       Pointer to the location where outputs are stored.
*/
int conv2d(float** f, int H, int W, float** g, int kH, int kW, int w_padding, int h_padding, float** output){

    const int total_height = H + h_padding;
    const int total_width = W + w_padding;

    // dimensions for convolution window
    const int M = (kH / 2);
    const int N = (kW / 2);

    // Offsets allow for asymmetric centering to account for evenly sized kernels
    const int M_offset = kH % 2 == 0 ? 1 : 0;
    const int N_offset = kW % 2 == 0 ? 1 : 0;

    // Iterate over every value in the feature map
    for (int n = h_padding; n < total_height; n++){       // feature : Iterate over Rows
        for (int k = w_padding; k < total_width; k++){    // feature : Iterate over columns

            float result = 0.0f;

            for (int i = -M; i <= M - M_offset; i++){               // kernel : Iterate over Rows
                for (int j = -N; j <= N - N_offset; j++){           // kernel : Iterate over columns
                    const int col = n + i;
                    const int row = k + j;

                    result += f[col][row] * g[i + M][j + N];
                }
            }
            output[n - h_padding][k - w_padding] = result;
        }
    }
    return 0;
}

/* 
* Performs Parallel 2D discrete convolutions. 
* @param f             Pointer to the Feature Map.
* @param H            Height of the Feature Map.
* @param W            Width of the Feature Map.
* @param g            Pointer to the Kernel.
* @param kH           Height of the Kernel.
* @param w_padding    Width of the padding in the Feature Map.
* @param h_padding    Height of the padding in the Feature Map.
* @param output       Pointer to the location where outputs are stored.
*/
int parallel_conv2d(float** f, int H, int W, float** g, int kH, int kW, int w_padding, int h_padding, float** output){

    const int total_height = H + h_padding;
    const int total_width = W + w_padding;

    // dimensions for convolution window
    const int M = (kH / 2);
    const int N = (kW / 2);

    // Offsets allow for asymmetric centering to account for evenly sized kernels
    const int M_offset = kH % 2 == 0 ? 1 : 0;
    const int N_offset = kW % 2 == 0 ? 1 : 0;

    // TODO: maybe we want to do reduction here idk
    for (int n = h_padding; n < total_height; n++){       // feature : Iterate over Rows
        for (int k = w_padding; k < total_width; k++){    // feature : Iterate over columns

            float result = 0.0f;

            for (int i = -M; i <= M - M_offset; i++){               // kernel : Iterate over Rows
                
                //#pragma omp parallel for reduction(+:result) firstprivate(f, g, n, i, k, M, N, N_offset)
                for (int j = -N; j <= N - N_offset; j++){           // kernel : Iterate over columns
                    const int col = n + i;
                    const int row = k + j;

                    result += f[col][row] * g[i + M][j + N];
                }
            }
            output[n - h_padding][k - w_padding] = result;
        }
    }
    return 0;
}


/*
Writes outputs to a file.
@param filepath     The filepath of where to find/put the output file.
@param outputs      The 2d convolutions outputs. This is what is written to the file.
@param h_dimension  The height of the outputs. Should be the same as the feature map.
@param w_dimension  The width of the outputs. Should be the same as the feature map.
*/
int write_data_to_file(char* filepath, float** outputs, int h_dimension, int w_dimension, int h_padding, int w_padding){
    if (filepath == NULL){ return 1; }
    FILE* file_ptr = fopen(filepath, "w");
    if (file_ptr == NULL){ return 1; }

    // Empty the file, then close it.
    fclose(file_ptr);

    // Reopen file in append mode
    file_ptr = fopen(filepath, "a");

    // Append the dimensions to the file
    fprintf(file_ptr, "%d %d\n", h_dimension, w_dimension);
    
    for (int i = h_padding; i < h_dimension + h_padding; i++){
        for (int j = w_padding; j < w_dimension + w_padding; j++){
            fprintf(file_ptr, "%.3f ", outputs[i-h_padding][j-w_padding]);
        }
        fprintf(file_ptr, "\n");
    }

    fclose(file_ptr);

    return 0;

}


// TODO: Delete. This is for debugging only.
void print2df(char* msg, float** arr, int size_x, int size_y){
    printf("\n%s\n", msg);
    for (int i=0; i<size_y; i++){
        for (int j=0; j<size_x; j++){
            printf("%f ", arr[i][j]);
        }
        printf("\n");
    }
}


/*
Generates a 2d array of random floats.
@param height   The height of the array.
@param width    The width of the array.
@param output   The location where the generated data will be stored.
*/
int generate_data(int height, int width, float** *output){
    
    // Reallocate required memory
    *output = (float**)malloc(height * sizeof(float*));
    for (int i=0; i<height; i++){
        (*output)[i] = (float*)calloc(width, sizeof(float));
    }

    // Seed
    srand(time(0));
    
    for (int i=0; i<height; i++){
        for (int j=0; j<width; j++){
            (*output)[i][j] = (float)rand() / (float)RAND_MAX;
        }
    }
    
    return 0;
}

/*
TODO: Delete this. For debugging only.
Verbose printf. This is used as a debug print, to give verbose updates as the program executes.
*/
void v_printf(char* msg, int verbose_mode){
    if (verbose_mode == 0) printf("%s", msg);
}


// Just a simple test to see if parallelisation is working
void parallel_testing(float** numbers, int height, int width, int threads){
    printf("\n");
    omp_set_num_threads(threads);
    printf("Threads:        %d\n", threads);
    
    // MEMORY HEAVY
    float result = 0.0f;
    clock_t start_time = clock();

    #pragma omp parallel for collapse(2) reduction(+:result) schedule(guided)
    for (int i = 0; i < height; i++){
        for (int j = 0; j < width; j++){
            result += numbers[i][j];
        }
    } 
    printf("Memory Time:    %f\n", (double)(clock() - start_time) / CLOCKS_PER_SEC);

    // COMPUTATION HEAVY
    result = 0.0f;
    start_time = clock();

    #pragma omp parallel for collapse(2) reduction(+:result) schedule(guided)
    for (int i = 0; i < height; i++){
        for (int j = 0; j < width; j++){
            result += numbers[i][j] * numbers[i][j] * numbers[i][j] * numbers[i][j];
        }
    } 
    printf("Compute Time:   %f\n", (double)(clock() - start_time) / CLOCKS_PER_SEC);
}



int main(int argc, char** argv) {
    
    // Initialising variables for future use
    // TODO: we should align all of these, to avoid False Sharing
    int H = 0;
    int W = 0;
    int kH = 0;
    int kW = 0;
    char* feature_file = NULL;
    char* kernel_file = NULL;
    char* output_file = NULL;

    // DEBUG FLAGS
    int benchmark_mode = 1;    // -b ... 1=false, 0=true
    int verbose_mode = 1;      // -v
    int parallel = 1;          // -p

    // Extract arguments into their variables
    for (int i = 1; i < argc; i++) {

        if (i + 1 > argc) { break; }

        // Check all flags
        if (strcmp(argv[i], "-H") == 0) {
            H = atoi(argv[++i]);
            continue;
        }
        if (strcmp(argv[i], "-W") == 0) {
            W = atoi(argv[++i]);
            continue;
        }
        if (strcmp(argv[i], "-kH") == 0) {
            kH = atoi(argv[++i]);
            continue;
        }
        if (strcmp(argv[i], "-kW") == 0) {
            kW = atoi(argv[++i]);
            continue;
        }
        if (strcmp(argv[i], "-f") == 0) {
            feature_file = argv[++i];
            continue;
        }
        if (strcmp(argv[i], "-g") == 0) {
            kernel_file = argv[++i];
            continue;
        }
        if (strcmp(argv[i], "-o") == 0) {
            output_file = argv[++i];
            continue;
        }
        if (strcmp(argv[i], "-b") == 0) {
            benchmark_mode = 0;
            continue;
        }
        if (strcmp(argv[i], "-v") == 0) {
            verbose_mode = 0;
            continue;
        }
        if (strcmp(argv[i], "-p") == 0) {
            parallel = 0;
            continue;
        }
    }

    omp_set_num_threads(4); // TODO: Maybe make this a flag?

    /* 
    TODO: Error catching for incorrect flag usage
        
        Examples: 
        1. No flags provided, 
        2. Generating array but provided only height not width, 
        3. Not generating and provided feature but no kernel
        4. Provided output, but generated/provided no input
        5. Incompatible datatype passed through for that flag.

    */

    /* 
    TODO:
        - (optionally) generate inputs. Both kernel and feature map.
        - Test to see if weirdly shaped kernels also work, e.g., 5x3, 2x1, 1x1, 1x9, 50x1, 25x10, etc
        - Compile with `-Wall -Werror` flags, to catch all potential issues. Fix them all, no exceptions.
    */



    // ~~~~~~~~~~~~~~ KERNEL ~~~~~~~~~~~~~~ // 

    // Check if we need to generate our own kernel
    
    float** kernel;

    // Generate Kernel
    if (kH > 0 || kW > 0){
        v_printf("Generating Kernel...      ", verbose_mode);

        // Allows users to specify only 1 dimension, and prevents them from inputting negative numbers
        kH = max(kH, 1);
        kW = max(kW, 1);

        // Allocating memory
        kernel = (float**)malloc(kH * sizeof(float*));
        for (int i = 0; i < kH; i++){
            kernel[i] = (float*)calloc(kW, sizeof(float));
        }

        generate_data(kH, kW, &kernel);

        // If wanting to save inputs, write to kernel file
        if (kernel_file != NULL){
            int status = write_data_to_file(kernel_file, kernel, kH, kW, 0, 0);
            if (status != 0){
                // TODO: Handle when it can't write to kernel
            }
        }

        parallel_testing(kernel, kH, kW, 1);
        parallel_testing(kernel, kH, kW, 4);
        parallel_testing(kernel, kH, kW, 8);
        parallel_testing(kernel, kH, kW, 16);
        parallel_testing(kernel, kH, kW, 32);
        return 0;

        v_printf("Finished.\n", verbose_mode);

    // Extract Kernel
    } else {
        v_printf("Extracting Kernel...      ", verbose_mode);
        // Extracting dimensions
        if (kernel_file != NULL){
            int status = extract_dimensions(kernel_file, &kH, &kW);
            if (status != 0){ 
                // TODO: Handle when it can't extract file dimensions
            }
        }

        // Allocating memory
        kernel = (float**)malloc(kH * sizeof(float*));
        for (int i = 0; i < kH; i++){
            kernel[i] = (float*)calloc(kW, sizeof(float));
        }

        // Extracting data
        if (kernel_file != NULL){
            int status = extract_data(kernel_file, kW, kH, 0, 0, &kernel);
            if (status != 0){
                // TODO: Handle when can't extract kernel
            }
        }

        parallel_testing(kernel, kH, kW, 1);
        parallel_testing(kernel, kH, kW, 4);
        parallel_testing(kernel, kH, kW, 8);
        parallel_testing(kernel, kH, kW, 16);
        parallel_testing(kernel, kH, kW, 32);
        return 0;

        v_printf("Finished.\n", verbose_mode);
    }

    // This is the "same padding" that'll be added to the feature map.
    const int padding_width = kW / 2;
    const int padding_height = kH / 2;

    
    
    // ~~~~~~~~~~~~~~ FEATURE MAP ~~~~~~~~~~~~~~ // 

    float** feature_map;

    // Generate Feature Map
    if (H > 0 || W > 0){

        v_printf("Generating Feature Map...     ", verbose_mode);

        // Allows users to specify only 1 dimension, and prevents them from inputting negative numbers
        H = max(H, 1);
        W = max(W, 1);

        const int total_width = W + padding_width*2;
        const int total_height = H + padding_height*2;

        // Allocating memory
        feature_map = (float**)malloc(total_height * sizeof(float*));
        for (int i = 0; i < total_height; i++){
            feature_map[i] = (float*)calloc(total_width, sizeof(float)); // calloc sets all to zero, creating padding
        }

        generate_data(total_height, total_width, &feature_map);

        // If wanting to save inputs, write to feature file
        if (feature_file != NULL){
            int status = write_data_to_file(feature_file, feature_map, H, W, padding_height, padding_width);
            if (status != 0){
                // TODO: Handle when it can't write to feature file
            }
        }

        v_printf("Finished.\n", verbose_mode);

    // Extract Feature Map
    } else {

        v_printf("Extracting Feature Map...     ", verbose_mode);
        // Extract dimensions of the feature map
        if (feature_file != NULL){
            int status = extract_dimensions(feature_file, &H, &W);
            if (status != 0){ 
                // TODO: Handle when it can't extract file dimensions
            }
        }

        const int total_width = W + padding_width*2;
        const int total_height = H + padding_height*2;

        // Allocate memory for the feature map of the feature map.
        feature_map = (float**)malloc(total_height * sizeof(float*));
        for (int i = 0; i < total_height; i++){
            feature_map[i] = (float*)calloc(total_width, sizeof(float)); // calloc sets all to zero, creating padding
        }

        // Extract Feature Map
        if (feature_file != NULL){
            int status = extract_data(feature_file, W, H, padding_width, padding_height, &feature_map);
            if (status != 0){
                // TODO: Handle when it can't extract data
            }
        }
        v_printf("Finished.\n", verbose_mode);
    }


    

    
    // ~~~~~~~~~~~~~~ conv2d() ~~~~~~~~~~~~~~ //
    

    if (output_file != NULL){

        v_printf("Creating Outputs...       ", verbose_mode);

        if (kernel[0] == NULL || feature_map[0] == NULL){
            printf("To generate an output, please provide all inputs.\n");
            return 1;
        }

        // Allocating memory
        float** outputs = (float**)malloc(H * sizeof(float*));
        for (int i = 0; i < H; i++){
            outputs[i] = (float*)calloc(W, sizeof(float));
        }

        v_printf("Finished.\n", verbose_mode);
        
        v_printf("Performing Convolutions...        ", verbose_mode);
        // Timing begins here, because implementation only starts here.
        clock_t start_time = clock();

        // Convolutions
        if (parallel == 0){
            int status = parallel_conv2d(feature_map, H, W, kernel, kH, kW, padding_width, padding_height, outputs);
            if (status != 0) {
                // TODO: Handle when can't perform convolutions
            }
        } else {
            int status = conv2d(feature_map, H, W, kernel, kH, kW, padding_width, padding_height, outputs);
            if (status != 0){
                // TODO: Handle when can't perform convolutions
            }
        }
        v_printf("Finished.\n", verbose_mode);
        
        clock_t end_time = clock();
        double total_time_taken = (double)(end_time - start_time) / CLOCKS_PER_SEC;

        if (benchmark_mode == 0) printf("Time:   %f\n", total_time_taken);


    // ~~~~~~~~~~~~~~ Write to Output ~~~~~~~~~~~~~~ //

        v_printf("Writing Outputs To File...        ", verbose_mode);

        int status = write_data_to_file(output_file, outputs, H, W, 0, 0);
        if (status != 0){
            // TODO: Handle when can't write to output.
        }

        v_printf("Finished.\n", verbose_mode);
    }

    v_printf("Finished!\n", verbose_mode);
    return 0;
}