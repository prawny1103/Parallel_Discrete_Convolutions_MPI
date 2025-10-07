// Name: Liam Hearder       Student Number: 23074422
// Name: Pranav Menon       Student Number: 24069351


// ~~~~~~~~~~~~~~ CONTENTS ~~~~~~~~~~~~~~ //
// 1. Includes and Defines
// 2. extract_dimensions()
// 3. extract_data()
// 4. conv2d()
// 5. parallel_conv2d()
// 6. write_data_to_file()
// 7. generate_data()
// 8. main()
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

/* The string length of every float in the feature map. Example line: "0.594 0.934 0.212\n". 
So, 3 floats, each looks like "X.XXX" which is 5 chars, but then all have a space or new-line 
character. */
#define FLOAT_STRING_LENGTH 6

// Macros for max, min,
#define max(a,b) (((a) > (b)) ? (a) : (b))
#define min(a,b) (((a) < (b)) ? (a) : (b))

// Macro for converting 2D indices to 1D index
#define IDX(row, col, step) ((row) * (step) + (col))

// Macro for finding the total number of strides in a row or column
#define TOTAL_STRIDES(size, stride) ((size>0) ? (((size-1)/stride) + 1) : (0))

// A struct to hold a float array and its padding, to prevent false sharing.
typedef struct {
    float* arr;
    char* padding;
} float_array;


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
int extract_data(char* filepath, int width, int height, int padding_width, int padding_height, float* *output) {
    
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
            (*output)[IDX(row_index, column_index, width + 2 * padding_width)] = element; // Add to output.
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
* @param kW           Width of the Kernel.
* @param sH           Stride height.
* @param sW           Stride width.
* @param w_padding    Width of the padding in the Feature Map.
* @param h_padding    Height of the padding in the Feature Map.
* @param output       Pointer to the location where outputs are stored.
*/
int conv2d_stride(float* f, int H, int W, float* g, int kH, int kW, int sH, int sW, int w_padding, int h_padding, float* output){

    const int total_height = H + h_padding*2;
    const int total_width = W + w_padding*2;
    const int total_strides_width = TOTAL_STRIDES(W, sW);

    // dimensions for convolution window
    const int M = (kH - 1) / 2;
    const int N = (kW - 1) / 2;

    // Iterate over every value in the feature map
    for (int n = h_padding; n < total_height - h_padding; n=n+sH){
        for (int k = w_padding; k < total_width - w_padding; k=k+sW){

            float result = 0.0f;

            // Iterate over every value in the kernel
            for (int j = 0; j < kW; j++){
                for (int i = 0; i < kH; i++){
                    result += f[IDX(n + i - M, k + j - N, total_width)] * g[IDX(i, j, kW)];
                }
            }
            output[IDX((n - h_padding)/sH, (k - w_padding)/sW, total_strides_width)] = result;
        }
    }
    return 0;
}


/* 
* Performs Parallel 2D discrete convolutions. 
* @param f              Pointer to the Feature Map.
* @param H              Height of the Feature Map.
* @param W              Width of the Feature Map.
* @param g              Pointer to the Kernel.
* @param kH             Height of the Kernel.
* @param kW             Width of the Kernel.
* @param w_padding      Width of the padding in the Feature Map.
* @param h_padding      Height of the padding in the Feature Map.
* @param padded_output  Location where outputs are stored.
*/
int parallel_conv2d(float* f, int H, int W, float* g, int kH, int kW, int w_padding, int h_padding, float_array padded_output){

    const int total_height = H + h_padding*2;
    const int total_width = W + w_padding*2;

    // dimensions for convolution window
    const int M = (kH - 1) / 2;
    const int N = (kW - 1) / 2;

    #pragma omp parallel for collapse(2) schedule(dynamic, total_width)
    for (int n = h_padding; n < total_height - h_padding; n++){
        for (int k = w_padding; k < total_width - w_padding; k++){
            float result = 0.0f;

            #pragma omp simd collapse(2) reduction(+:result)
            for (int j = 0; j < kW; j++){
                for (int i = 0; i < kH; i++){
                    result += f[IDX(n + i - M, k + j - N, total_width)] * g[IDX(i, j, kW)];
                }
            }
            padded_output.arr[IDX(n - h_padding, k - w_padding, W)] = result;
        }
    }
    return 0;
}


/*
Writes outputs to a file.
@param filepath         The filepath of where to find/put the output file.
@param outputs          A 2d array of float32s. This is what is written to the file for serial convolutions.
@param padded_outputs   A padded 2d array of float32s. This is written to the file, instead of `outputs`, when outputting data from a parallel convolution.
@param h_dimension      The height of the outputs. Should be the same as the feature map.
@param w_dimension      The width of the outputs. Should be the same as the feature map.
*/
int write_data_to_file(char* filepath, float* outputs, float_array padded_outputs, int h_dimension, int w_dimension, int h_padding, int w_padding){
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

            // Depending if paralleism is enabled or not, print the outputs
            if (outputs != NULL){
                fprintf(file_ptr, "%.3f ", outputs[IDX(i-h_padding, j-w_padding, w_dimension)]);
            } else if (padded_outputs.arr != NULL){
                fprintf(file_ptr, "%.3f ", padded_outputs.arr[IDX(i-h_padding, j-w_padding, w_dimension)]);
            } else { return 1; }
            
        }
        fprintf(file_ptr, "\n");
    }

    fclose(file_ptr);

    return 0;

}


/*
Generates a 2d array of random floats.
@param height   The height of the array.
@param width    The width of the array.
@param output   The location where the generated data will be stored.
*/
int generate_data(int height, int width, float* *output){

    // Make a new random seed. This stops f from being the same as g when the code runs too fast.
    srand(rand());
    
    for (int i=0; i<height; i++){
        for (int j=0; j<width; j++){
            (*output)[IDX(i,j,width)] = (float)rand() / (float)RAND_MAX;
        }
    }
    return 0;
}


int main(int argc, char** argv) {
    
    // ~~~~~~~~~~~~~~~ MAIN CONTENTS ~~~~~~~~~~~~~~ //
    // 1. Argument Extraction
    // 2. Error Handling
    // 3. Kernel Generation / Extraction
    // 4. Feature Map Generation / Extraction
    // 5. Serial Convolutions / Parallel Convolutions
    // 6. Write to Output
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //


    // ~~~~~~~~~~~~~~~ 1. Argument Extraction ~~~~~~~~~~~~~~ //


    omp_set_nested(1); // Allow nested parallelism for SIMD

    // Seed for random generation later
    srand(time(0));

    // Initialising variables for future use
    int H = 0;                      // -H <int>
    int W = 0;                      // -W <int>
    int kH = 0;                     // -kH <int>
    int kW = 0;                     // -kW <int>
    int sW = 1;                     // -sW <int>
    int sH = 1;                     // -sH <int>
    char* feature_file = NULL;      // -f <path>
    char* kernel_file = NULL;       // -g <path>
    char* output_file = NULL;       // -o <path>

    // DEBUG FLAGS
    int benchmark_mode = 0;         // -b
    int multi_benchmark_mode = 0;   // -mb <max_iterations>
    int max_iterations = 1;             // Used by multi_benchmark_mode to run the code multiple times, getting an average.
    int threads = 1;                // -t <threads>
    

    // Extract arguments into their variables
    for (int i = 1; i < argc; i++) {

        if (i + 1 > argc) { break; }

        // Check all flags
        if (strcmp(argv[i], "-H") == 0) {
            if (i + 1 >= argc) { printf("Incorrect usage of -H flag. Please provide an input height.\n"); return 1; }
            H = atoi(argv[++i]) > 0 ? atoi(argv[i]) : 1;
            continue;
        }
        if (strcmp(argv[i], "-W") == 0) {
            if (i + 1 >= argc) { printf("Incorrect usage of -W flag. Please provide an input width.\n"); return 1; }
            W = atoi(argv[++i]) > 0 ? atoi(argv[i]) : 1;
            continue;
        }
        if (strcmp(argv[i], "-kH") == 0) {
            if (i + 1 >= argc) { printf("Incorrect usage of -kH flag. Please provide a kernel height.\n"); return 1; }
            kH = atoi(argv[++i]) > 0 ? atoi(argv[i]) : 1;
            continue;
        }
        if (strcmp(argv[i], "-kW") == 0) {
            if (i + 1 >= argc) { printf("Incorrect usage of -kW flag. Please provide a kernel width.\n"); return 1; }
            kW = atoi(argv[++i]) > 0 ? atoi(argv[i]) : 1;
            continue;
        }
        if (strcmp(argv[i], "-sW") == 0) {
            if (i + 1 >= argc) { printf("Incorrect usage of -sW flag. Please provide a stride width.\n"); return 1; }
            sW = atoi(argv[++i]) > 0 ? atoi(argv[i]) : 1;
            continue;
        }
        if (strcmp(argv[i], "-sH") == 0) {
            if (i + 1 >= argc) { printf("Incorrect usage of -sH flag. Please provide a stride height.\n"); return 1; }
            sH = atoi(argv[++i]) > 0 ? atoi(argv[i]) : 1;
            continue;
        }
        if (strcmp(argv[i], "-f") == 0) {
            if (i + 1 >= argc) { printf("Incorrect usage of -f flag. Please provide a filepath.\n"); return 1; }
            feature_file = argv[++i];
            continue;
        }
        if (strcmp(argv[i], "-g") == 0) {
            if (i + 1 >= argc) { printf("Incorrect usage of -g flag. Please provide a filepath.\n"); return 1; }
            kernel_file = argv[++i];
            continue;
        }
        if (strcmp(argv[i], "-o") == 0) {
            if (i + 1 >= argc) { printf("Incorrect usage of -o flag. Please provide a filepath.\n"); return 1; }
            output_file = argv[++i];
            continue;
        }
        if (strcmp(argv[i], "-b") == 0) {
            benchmark_mode = 1;
            continue;
        }
        if (strcmp(argv[i], "-mb") == 0) {
            multi_benchmark_mode = 1;
            if (i + 1 >= argc) { max_iterations = 15; continue; }
            max_iterations = atoi(argv[++i]) > 0 ? atoi(argv[i]) : 15;
            continue;
        }
        if (strcmp(argv[i], "-t") == 0) {
            if (i + 1 >= argc) { printf("Incorrect usage of -t flag. Please provide a number of threads.\n"); return 1; }
            threads = atoi(argv[++i]) > 0 ? atoi(argv[i]) : 1;
            omp_set_num_threads(threads);
            continue;
        }
    }


    // ~~~~~~~~~~~~~~~ 2. Error Handling ~~~~~~~~~~~~~~ //

    if (benchmark_mode) { 
        if (threads > 1){
            printf("Beginning Parallel Convolutions with %d threads...\n", threads);
        } else {
            printf("Beginning Serial Convolutions...\n");
        }
    };

    // Error handling

    if (H < 0 || W < 0 || kH < 0 || kW < 0 || threads < 1 || max_iterations < 1){
        printf("Please provide only positive integers for dimensions and threads.\n");
        return 1;
    }
    if (H == 0 && W == 0 && feature_file == NULL){
        printf("Please provide either a feature map file or dimensions to generate one.\n");
        return 1;
    }
    if (kH == 0 && kW == 0 && kernel_file == NULL){
        printf("Please provide either a kernel file or dimensions to generate one.\n");
        return 1;
    }

    double average_time = 0.0f;
    for (int iteration = 0; iteration < max_iterations; iteration++){

    if (multi_benchmark_mode && (feature_file || kernel_file)) { printf("Do not input a file while running multi-benchmark mode.\n"); return 1; }



    // ~~~~~~~~~~~~~~ 3. Kernel Generation / Extraction ~~~~~~~~~~~~~~ //

    float* kernel = NULL;

    // Generate Kernel
    if (kH > 0 || kW > 0){

        // Allows users to specify only 1 dimension, and prevents them from inputting negative numbers
        kH = max(kH, 1);
        kW = max(kW, 1);

        // Allocating memory
        if (posix_memalign((void**)&kernel, 64, kW * kH * sizeof(float)) != 0){
            printf("Error allocating memory for kernel.\n");
            return 1;
        }

        generate_data(kH, kW, &kernel);

        // If wanting to save inputs, write to kernel file
        if (kernel_file != NULL){
            int status = write_data_to_file(kernel_file, kernel, (float_array){0}, kH, kW, 0, 0);
            if (status != 0){
                printf("Error writing kernel to file.\n");
                return 1;
            }
        }

    // Extract Kernel
    } else if (kernel_file != NULL){

        // Extracting dimensions
        if (extract_dimensions(kernel_file, &kH, &kW) != 0){ 
            printf("Error extracting kernel dimensions from file.\n");
            return 1;
        }
        
        // Allocating memory
        if (posix_memalign((void**)&kernel, 64, kW * kH * sizeof(float)) != 0){
            printf("Error allocating memory for kernel.\n");
            return 1;
        }

        // Extracting data
        if (extract_data(kernel_file, kW, kH, 0, 0, &kernel) != 0){
            printf("Error extracting kernel data from file.\n");
            return 1;
        }
    }

    // This is the "same padding" that'll be added to the feature map.
    const int padding_width = kW / 2;
    const int padding_height = kH / 2;

    
    
    // ~~~~~~~~~~~~~~ 4. Feature Map Generation / Extraction ~~~~~~~~~~~~~~ //

    float* feature_map = NULL;

    // Generate Feature Map
    if (H > 0 || W > 0){

        // Allows users to specify only 1 dimension, and prevents them from inputting negative numbers
        H = max(H, 1);
        W = max(W, 1);

        const int total_width = W + padding_width*2;
        const int total_height = H + padding_height*2;

        // Allocating memory
        if (posix_memalign((void**)&feature_map, 64, total_width * total_height * sizeof(float)) != 0){
            printf("Error allocating memory for feature map.\n");
            return 1;
        }

        // Add zeroes as padding
        for (int i = 0; i < total_width * total_height; i++){
            feature_map[i] = 0.0f;
        }
        

        generate_data(total_height, total_width, &feature_map);

        // If wanting to save inputs, write to feature file
        if (feature_file != NULL){
            if (write_data_to_file(feature_file, feature_map, (float_array){0}, H, W, padding_height, padding_width) != 0){
                printf("Error writing feature map to file.\n");
                return 1;
            }
        }


    // Extract Feature Map
    } else if (feature_file != NULL) {

        // Extract dimensions of the feature map
        if (extract_dimensions(feature_file, &H, &W) != 0){ 
            printf("Error extracting feature map dimensions from file.\n");
            return 1;
        }

        const int total_width = W + padding_width*2;
        const int total_height = H + padding_height*2;

        // Allocate memory for the feature map of the feature map.
        if (posix_memalign((void**)&feature_map, 64, total_width * total_height * sizeof(float)) != 0){
            printf("Error allocating memory for feature map.\n");
            return 1;
        }
        
        // Add zeroes as padding
        for (int i = 0; i < total_width * total_height; i++){
            feature_map[i] = 0.0f;
        }

        // Extract Feature Map
        if (extract_data(feature_file, W, H, padding_width, padding_height, &feature_map) != 0){
            printf("Error extracting feature map data from file.\n");
            return 1;
        }        
    }
        

    
    // ~~~~~~~~~~~~~~ 5. Serial Convolutions / Parallel Convolutions ~~~~~~~~~~~~~~ //
    
    // Check if we have all the inputs we need to perform convolutions
    if (kernel == NULL || feature_map == NULL){
        printf("To generate an output, please provide all inputs.\n");
        return 1;
    }

    // Defining output pointers
    float* outputs = NULL;              // Used for serial convolution
    float_array padded_outputs = {0};   // Used for parallel convolution    
    
    // This is how many times a convolution takes place over each row/col. Also used to determine output array size.
    const int total_strides_width = TOTAL_STRIDES(W, sW);
    const int total_strides_height = TOTAL_STRIDES(H, sH);

    // Parallel Convolutions
    if (threads > 1){
        
        // The size of the array padding. Used to prevent false sharing.
        // Equal to the number of bytes left over in the cache line containing the final element in float array.
        const int cache_padding_size = 64 - ((W * sizeof(float)) % 64);

        if (posix_memalign((void**)&padded_outputs.arr, 64, W * H * sizeof(float)) != 0){
            printf("Error allocating memory for padded output.\n");
            return 1;
        }
        padded_outputs.padding = cache_padding_size == 64 ? NULL : (char*)malloc(cache_padding_size);

        // Timing begins here, because implementation only starts here.
        double start_time = omp_get_wtime();

        if (parallel_conv2d(feature_map, H, W, kernel, kH, kW, padding_width, padding_height, padded_outputs) != 0) {
            printf("Error performing parallel convolutions.\n");
            return 1;
        }

        if (benchmark_mode == 1) { printf("%f\n", (omp_get_wtime() - start_time));}
        if (multi_benchmark_mode == 1) { average_time += (omp_get_wtime() - start_time); }
        
    // Serial Convolutions
    } else {

        if (posix_memalign((void**)&outputs, 64, total_strides_width * total_strides_height * sizeof(float)) != 0){
            printf("Error allocating memory for outputs.\n");
            return 1;
        }

        double start_time = omp_get_wtime();

        if (conv2d_stride(feature_map, H, W, kernel, kH, kW, sH, sW, padding_width, padding_height, outputs) != 0){
            printf("Error performing serial convolutions.\n");
            return 1;
        }

        // Benchmarking
        if (benchmark_mode == 1) {printf("%f\n", (omp_get_wtime() - start_time)); }
        if (multi_benchmark_mode == 1) { average_time += (omp_get_wtime() - start_time); }
    }
        
        


    // ~~~~~~~~~~~~~~ 6. Write to Output ~~~~~~~~~~~~~~ //

    if (output_file != NULL){

        if (write_data_to_file(output_file, outputs, padded_outputs, total_strides_height, total_strides_width, 0, 0) != 0){
            printf("Error writing outputs to file.\n");
            return 1;
        }

        // Free any remaining memory
        if (outputs != NULL) {free(outputs); outputs = NULL;}
        if (padded_outputs.arr != NULL) { free(padded_outputs.arr); padded_outputs.arr = NULL; }
        if (padded_outputs.padding != NULL) { free(padded_outputs.padding); padded_outputs.padding = NULL;}

    }
    
    if (feature_map != NULL) {free(feature_map); feature_map = NULL; }
    if (kernel != NULL) {free(kernel); kernel = NULL; }

    } // End of loop for multi_benchmark_mode

    if (multi_benchmark_mode == 1) {printf("Average Time:   %f\n", average_time/max_iterations);}

    return 0;
}