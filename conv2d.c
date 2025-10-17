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
#include <mpi.h>

/* The string length of every float in the feature map. Example line: "0.594 0.934 0.212\n". 
So, 3 floats, each looks like "X.XXX" which is 5 chars, but then all have a space or new-line 
character. */
#define FLOAT_STRING_LENGTH 6

// Macros for max, min,
#define max(a,b) (((a) > (b)) ? (a) : (b))
#define min(a,b) (((a) < (b)) ? (a) : (b))

// Macro for converting 2D indices to 1D index
#define IDX(row, col, step) ((row) * (step) + (col))

// Macro for a 1D index with padding into a 1D index without padding
#define IDX_MINUS_PADDING(index, step, pH, pW) ( (index - ((step+pW*2)*pH) - (((index/(step+pW*2))*pW*2)-1)) )

// Macro for finding the total number of strides in a row or column
#define TOTAL_STRIDES(size, stride) ((size>0) ? (((size-1)/stride) + 1) : (0))

// Macro for rounding to the nearest integer.
#define ROUND(number) ( ((int)((number) + 0.5)))

// Macro for calculating power
#define POW(base, exponent) (({ int RESULT = 1; for (int INDEX = 0; INDEX < exponent; INDEX++) { RESULT *= (base); } RESULT; }))

// Macro for accurate rounding.
#define ROUNDF(number, precision) ( (float)(ROUND(number * POW(10, precision))) / (float)(POW(10, precision)) )


// A struct to hold a float array and its padding, to prevent false sharing.
typedef struct {
    float* arr;
    char* padding;
} float_array;

int find_char_count(int number){
    if (number == 0) { return 1; }
    int count = 0;
    if (number < 0) { count++; number = -number; } // For negative sign
    while (number != 0) {
        number /= 10;
        count++;
    }
    return count;
}

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


/* TODO: Check if we can remove "sW" and "sH", we might not need them in the end.
* Reads an input file and extracts data into an output. 
* @param filepath           The filepath where the data is stored.
* @param width              The number of elements in each line. Width.
* @param height             The number of rows. Height.
* @param padding_width      The number of zeroes to pad the width with.
* @param padding_height     The number of zeroes to pad the height with.
* @param start_index        The index at which to start extracting rows of data. Used to divide up work using MPI. 
                                To extract all data, set this to zero. To make an "end_index", use "height".
* @param sW                 Stride width.
* @param sH                 Stride height.
* @param output             The stream into which the inputs will be stored.
*/
int extract_data(char* filepath, int width, int height, int padding_width, int padding_height, int start_index, /*int sW, int sH,*/ float* *output) {

    if (filepath == NULL){ return 1; }
    FILE* file_ptr = fopen(filepath, "r");
    if (file_ptr == NULL){ return 1; }

    // Create a buffer to place extracted strings into
    const size_t buffer_size = (FLOAT_STRING_LENGTH * width) + 2; // +2 for new-line and null-byte
    char* buffer = (char*)malloc(buffer_size);

    const int rows_to_read = height + padding_height*2;
    const int total_width = width + padding_width*2;
    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const int row_offset = (rank == 0) ? padding_height : 0;
    
    // Safely get the header here so we can ignore later
    fgets(buffer, buffer_size, file_ptr);

    // Skip all lines before the start
    for (int i = 0; i < start_index-1; i++){
        fgets(buffer, buffer_size, file_ptr);
    }

    // Iterates over every row
    for (int i = 0; i < rows_to_read; i++){
        if (fgets(buffer, buffer_size, file_ptr) == NULL) { break; } // Exit if invalid
        char* token = strtok(buffer, " ");

        // Iterates over every column in the row
        for (int j = 0; j < width; j++) {
            if (token != NULL){

                // Extracted float data from the token 
                const float element = (float)atof(token);

                // Calculations for where to put the extracted data into the output array
                const int output_col_index = i + row_offset;       // TODO: Might need   i/sH + row_offset
                const int output_row_index = j + padding_width;    // TODO: Might need   j/sW + padding_width
                const int index = IDX(output_col_index, output_row_index, total_width); 
                
                if (index >= total_width * (height + padding_height*2)) {continue;}
                (*output)[index] = element; // Add to output.
                token = strtok(NULL, " ");
            }
        }
    }
    free(buffer);
    fclose(file_ptr);
    return 0;
}


/** 
* Performs Parallel 2D discrete convolutions. 
* @param f              Pointer to the Feature Map.
* @param H              Height of the Feature Map.
* @param W              Width of the Feature Map.
* @param g              Pointer to the Kernel.
* @param kH             Height of the Kernel.
* @param kW             Width of the Kernel.
* @param sH             Stride height.
* @param sW             Stride width.
* @param w_padding      Width of the padding in the Feature Map.
* @param h_padding      Height of the padding in the Feature Map.
* @param start_index    The starting index location, of index 0 in the local Feature Map, in the original Feature Map.
                            This is used for calculating stride at runtime, to determine which rows should be skipped.
* @param padded_output  Location where outputs are stored.
*/
int conv2d_stride(float* f, int H, int W, float* g, int kH, int kW, int sH, int sW, int w_padding, int h_padding, int start_index, float_array padded_output){

    // dimensions for convolution window
    const long M_top = (kH - 1) / 2;
    const long M_bot = kH%2==0 ? M_top+1 : M_top;

    const long N_left = (kW - 1) / 2;
    const long N_right = kW%2==0 ? N_left+1 : N_left;

    printf("M=[%ld,%ld], N=[%ld,%ld]\n", M_top, M_bot, N_left, N_right);

    const long total_strides_width = TOTAL_STRIDES(W, sW);
    //const long total_strides_height = TOTAL_STRIDES(H, sH);
    const long feature_length = H * W;
    //const long output_length = total_strides_width * total_strides_height;
    const long kernel_length = (long)kH * (long)kW;


    // This is 1 (an error) until any change is made to the output array, then it is set to 0. If no change is made, then it stays as 1 and outputs an error. 
    // The purpose of this variable is stop processes from outputting no valid elements because there aren't enough rows for the number of processes.
    // Additionally, it is fully intended that this variable is shared amongst threads because if even one thread contains valid data, we should output it.
    _Bool return_code = 1;

    /////////////////   NEW CODE    /////////////////

    double result = 0.0;

    // for (int i = 0; i < H; i++){
    //     for (int j = 0; j < W; j++){
    //         printf("%0.3f ", f[IDX(i,j,W)]);
    //     }
    //     printf("\n");
    // }

    /* Chunk size = row_width * (kernel_length + 1)
    Iterations being (length + 1) is because we add an extra iteration to the end to write the output, allowing us to freely skip all prior iterations if need be.
    TODO: If performance drags dramtically with larger kernels, we need to consider splitting up the work across threads/processes for each individual convolution. This'd be hard */

    #pragma omp parallel for collapse(2) schedule(static, W*(kernel_length+1)) firstprivate(result)
    for (long i = 0; i < feature_length; i++){      // TODO: Need to work out how stride would work...
        for (long j = 0; j < kernel_length + 1; j++){

            // Current 2D write position in the feature_map / output array
            const int f_col = (i % W);
            const int f_row = (i / W);

            if (f_col % sW != 0 || f_row % sH != 0) { continue; }

            // Current 2D iterative position in the kernel.
            const int k_col = j % kW;
            const int k_row = j / kW;

            
            // Reset result at the start of every kernel iteration
            if (j == 0) { result = 0.0; }

            if (j == kernel_length) {
                padded_output.arr[IDX(f_row/sH, f_col/sW, total_strides_width)] = (float)result;
                return_code = 0;
                continue;
            }

  
            // Find the position to read from in the feature map
            const long read_pos = i-N_left-(W*M_top) + (W*(j/kW)) + (j%kW);
            float read_value = f[read_pos];

            // logical padding
            if (f_col - N_left < 0 && N_left-k_col > f_col) continue;               // left
            if (f_col + N_right >= W && k_col-N_left >= (W - f_col)) continue;      // right
            if (f_row - M_top < 0 && M_top-k_row > f_row) continue;                 // top
            if (f_row + M_bot >= H && k_row-M_bot >= (H - f_row)) continue;         // bottom

            // if(i == 6){
            //     printf("Read: [%d,%d]\n", k_row, k_col);
            // }

            result += (double)(read_value) * (double)(g[j]);
        }
    }

    return return_code;
}


/*
Writes outputs to a file.
@param filepath             The filepath of where to find/put the output file.
@param outputs              A 2d array of float32s. This is used only for kernel/feature map writing.
@param padded_outputs       A padded 2d array of float32s. This is written to the file, instead of `outputs`, when outputting data from a parallel convolution.
@param height               The height of the outputs.
@param width                The width of the outputs.
@param append_dimensions    Whether or not the w_dimension and h_dimension fields should be written into the file.
@param w_dimension          The height dimension to write to the file.
@param h_dimension          The width dimension to write to the file.
*/
int write_data_to_file(char* filepath, float* outputs, float_array padded_outputs, int height, int width, int h_padding, int w_padding, int append_dimensions, int w_dimension, int h_dimension ){
    if (filepath == NULL){ return 1; }

    FILE* file_ptr;

    if (append_dimensions == 1){
        file_ptr = fopen(filepath, "w");
        if (file_ptr == NULL){ return 1; }

        // Empty the file, then close it.
        fclose(file_ptr);
    }
    
    // Reopen file in append mode
    file_ptr = fopen(filepath, "a");

    // Append the dimensions to the file
    if (append_dimensions == 1){
        fprintf(file_ptr, "%d %d\n", h_dimension, w_dimension);
    }
    
    for (int i = h_padding; i < height + h_padding; i++){
        for (int j = w_padding; j < width + w_padding; j++){

            // Depending if paralleism is enabled or not, print the outputs
            if (outputs != NULL){
                fprintf(file_ptr, "%0.3f ", ROUNDF(outputs[IDX(i-h_padding, j-w_padding, width)], 3));
            } else if (padded_outputs.arr != NULL){
                fprintf(file_ptr, "%0.3f ", ROUNDF(padded_outputs.arr[IDX(i-h_padding, j-w_padding, width)],3));
            } else { return 1; }
            
        }
        fprintf(file_ptr, "\n");
    }

    fclose(file_ptr);

    return 0;

}


/**
* @brief Generates a 2d array of random floats.
* @param height   The height of the array.
* @param width    The width of the array.
* @param output   The location where the generated data will be stored.
*/
int generate_data(int height, int width, float* *output, int seed){

    // Make a new random seed. This stops f from being the same as g when the code runs too fast.
    if (seed == -1) {
        srand(rand());
    }
    
    for (int i=0; i< height * width; i++){
        (*output)[i] = (float)rand() / (float)RAND_MAX;
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

    // Initialising variables for future use
    int H = 0;                      // @brief -H <int> @details Height of the input feature map.
    int W = 0;                      // -W <int>
    int kH = 0;                     // -kH <int>
    int kW = 0;                     // -kW <int>
    int sW = 1;                     // -sW <int>
    int sH = 1;                     // -sH <int>
    char* feature_file = NULL;      // -f <path>
    char* kernel_file = NULL;       // -g <path>
    char* output_file = NULL;       // -o <path>
    int threads = 1;                // -t <threads>

    // DEBUG FLAGS
    int benchmark_mode = 0;         // -b
    int multi_benchmark_mode = 0;   // -mb <max_iterations>
    int max_iterations = 1;             // Used by multi_benchmark_mode to run the code multiple times, getting an average.

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




    // ~~~~~~~~~~~~~~ MPI Initialisation ~~~~~~~~~~~~~~ //

    int provided;
    // MPI_THREAD_MULTIPLE
    // MPI_THREAD_FUNNELED
    MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);




    // ~~~~~~~~~~~~~~~ Error Handling ~~~~~~~~~~~~~~ //

    if (rank == 0){
        if (benchmark_mode) { 
            if (threads > 1){ printf("Beginning Parallel Convolutions across %d processes with %d threads...\n", world_size, threads);} 
            else { printf("Beginning Serial Convolutions across %d processes...\n", world_size); }
        }
        if (H < 0 || W < 0 || kH < 0 || kW < 0 || threads < 1 || max_iterations < 1){
            printf("Please provide only positive integers for dimensions and threads.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return 1;
        }
        if (H == 0 && W == 0 && feature_file == NULL){
            printf("Please provide either a feature map file or dimensions to generate one.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return 1;
        }
        if (kH == 0 && kW == 0 && kernel_file == NULL){
            printf("Please provide either a kernel file or dimensions to generate one.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return 1;
        }

        if (multi_benchmark_mode && (feature_file || kernel_file)) { 
            printf("Do not input a file while running multi-benchmark mode.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return 1;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD); 

    double average_time = 0.0f;
    for (int iteration = 0; iteration < max_iterations; iteration++){

    // Seed for random generation later. This ensures the seed is identical across all processes.
    time_t featureMapSeed;
    if (rank == 0) { featureMapSeed = time(0);}
    MPI_Bcast(&featureMapSeed, 1, MPI_LONG, 0, MPI_COMM_WORLD);




    // ~~~~~~~~~~~~~~ 4. Kernel Generation / Extraction ~~~~~~~~~~~~~~ //

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

        // Generates a random kernel. 
        generate_data(kH, kW, &kernel, -1);

        // If wanting to save inputs, write to kernel file
        if (kernel_file != NULL && rank == 0){
            int status = write_data_to_file(kernel_file, kernel, (float_array){0}, kH, kW, 0, 0, 1, kW, kH);
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
        if (extract_data(kernel_file, kW, kH, 0, 0, 0,/* sW, sH,*/ &kernel) != 0){
            printf("Error extracting kernel data from file.\n");
            return 1;
        }
    }

    // This is the "same padding" that'll be added to the feature map.
    const int padding_width = kW / 2;
    const int padding_height = kH / 2;


    // Determine the number of rows each process should use in convolutions.
    // This ONLY thinks about the outer loop iterations. The real data size will have padding_height*2 added (because inner loops use more rows)
    int rowCount;

    // This just includes padding
    int totalRowCount;



    // ~~~~~~~~~~~~~~ 5. Feature Map Generation / Extraction ~~~~~~~~~~~~~~ //

    float* feature_map = NULL;

    // Used to determine the starting index in the feature map where relevant data is found for a process.
    int start_index;

    // Generate Feature Map
    if (H > 0 || W > 0){

        // Allows users to specify only 1 dimension, and prevents them from inputting negative numbers
        H = max(H, 1);
        W = max(W, 1);

        start_index = rank * (H / world_size) + min(rank, (H % world_size));

        //const int total_width = W + padding_width*2;
        //const int total_height = H + padding_height*2;

        // Determine the number of rows each process should use in convolutions.
        rowCount = (H / world_size) + (rank < (H % world_size) ? 1 : 0);

        int overlapBefore = 0;
        int overlapAfter = 0;
        
        if (rank > 0) overlapBefore = padding_height;
        if (rank < world_size - 1) overlapAfter = padding_height;
        
        totalRowCount = rowCount + overlapBefore + overlapAfter;

        int startRow = 0;
        for (int i = 0; i < rank; i++) {
            startRow += (H / world_size) + (i < (H % world_size) ? 1 : 0);
        }
        
        int startWithOverlap = startRow - overlapBefore;

        // Allocating memory
        if (posix_memalign((void**)&feature_map, 64, W * totalRowCount * sizeof(float)) != 0){     // Originally was total_width * total_height * sizeof(float)
            printf("Error allocating memory for feature map.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return 1;
        }

        // Add zeroes
        for (int i = 0; i < W * totalRowCount; i++){
            feature_map[i] = 0.0f;
        }

        srand(featureMapSeed);
        
        int values_to_skip = startWithOverlap * W;
        for (int i = 0; i < values_to_skip; i++) {
            rand();
        }

        generate_data(totalRowCount, W, &feature_map, featureMapSeed);


        // If wanting to save inputs, write to feature file
        if (feature_file != NULL){

            // Open the file
            MPI_File handle;
            int access_mode = MPI_MODE_CREATE | MPI_MODE_RDWR;
            if (MPI_File_open(MPI_COMM_WORLD, feature_file, access_mode, MPI_INFO_NULL, &handle) != MPI_SUCCESS) {
                printf("Process %d, Failure in opening file.\n", rank);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }

            // number of chars in the "H W\n" header
            int dimensions_offset = find_char_count(H) + 1 + find_char_count(W) + 1; // e.g. "6 6\n"

            // characters per row including trailing space/newline
            const int line_len = W * FLOAT_STRING_LENGTH + 1;
            const size_t local_chars = (size_t)line_len * (size_t)rowCount;
            const size_t buffer_bytes = local_chars + (rank == 0 ? (size_t)dimensions_offset : 0);

            // allocate buffer (bytes)
            char* buffer;
            if (posix_memalign((void**)&buffer, 64, buffer_bytes) != 0) {
                printf("Error allocating memory for feature map buffer.\n");
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                return 1;
            }

            // fill header for rank 0
            size_t data_start = 0;
            if (rank == 0) {
                sprintf(buffer, "%d %d\n", H, W);
                data_start = (size_t)dimensions_offset;
            }

            // Write only the non-overlap rows belonging to this process:
            // rows in feature_map that correspond to file are [overlapBefore .. overlapBefore+rowCount-1]
            for (int i = 0; i < rowCount; i++) {
                int src_row = overlapBefore + i; // index into feature_map
                char* row_ptr = buffer + data_start + (size_t)i * line_len;
                // write each value into row_ptr
                char* p = row_ptr;
                for (int j = 0; j < W; j++) {
                    // ensure we don't overrun the per-value slot; FLOAT_STRING_LENGTH is reserved per value
                    // use snprintf to be safer with bounds
                    int written = snprintf(p, FLOAT_STRING_LENGTH + 1, "%0.3f ", ROUNDF(feature_map[IDX(src_row, j, W)], 3));
                    if (written < 0) written = 0;
                    p += FLOAT_STRING_LENGTH; // advance fixed slot
                }
                // replace trailing slot last char with newline (or append newline at end)
                row_ptr[line_len - 1] = '\n';
            }

            // compute file offset: header sits at 0, rank 0 writes header+its chunk at offset 0.
            // other ranks write starting at dimensions_offset + rank * local_chars
            MPI_Offset write_offset = (rank == 0) ? 0 : (MPI_Offset)dimensions_offset + (MPI_Offset)rank * (MPI_Offset)local_chars;
            int write_len = (int)(rank == 0 ? buffer_bytes : local_chars);

            MPI_File_write_at(handle, write_offset, buffer, write_len, MPI_CHAR, MPI_STATUS_IGNORE);

            free(buffer);
            MPI_File_close(&handle);
        }

        

    // Extract Feature Map
    } else if (feature_file != NULL) {

        // Extract dimensions of the feature map
        if (extract_dimensions(feature_file, &H, &W) != 0){ 
            printf("Error extracting feature map dimensions from file.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return 1;
        }

        // Determine the number of rows each process should use in convolutions.
        // This ONLY thinks about the outer loop iterations. To get total iterations, multiply this by kernel length.
        rowCount = (H / world_size) + (rank < (H % world_size) ? 1 : 0);

        // This just includes padding
        totalRowCount = rowCount + padding_height*2;
        
        // Tells each process where to start reading from the feature map (to extract only relevant data)
        start_index = max(0, rank * (H / world_size) + min(rank, (H % world_size)));

        // TODO: See if you can use an MPI function for this. Is there something to read data from a file across all processes?


        // Allocate memory for the feature map of the feature map.
        if (posix_memalign((void**)&feature_map, 64, W * totalRowCount * sizeof(float)) != 0){
            printf("Error allocating memory for feature map.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return 1;
        }

        // Extract Feature Map
        if (extract_data(feature_file, W, rowCount, 0, 0, start_index, &feature_map) != 0){
            printf("Error extracting feature map data from file.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return 1;
        }    
    }
        

    
    // ~~~~~~~~~~~~~~ 6. Convolutions ~~~~~~~~~~~~~~ //
    
    // Check if we have all the inputs we need to perform convolutions
    if (kernel == NULL || feature_map == NULL){
        printf("To generate an output, please provide all inputs.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return 1;
    }

    // Defining output pointers
    //float* outputs = NULL;              // Used for serial convolution
    float_array padded_outputs = {0};   // Used for parallel convolution
    
    // This is how many times a convolution takes place over each row/col. Also used to determine output array size.
    const int total_strides_width = TOTAL_STRIDES(W, sW);
    const int total_strides_height = TOTAL_STRIDES(rowCount, sH); // TODO: Fix this to work properly. This assumes stride = 1

    // Used to determine if a process should write to file. This changes to 0 if an error occurs in the convolutions.
    int should_write_to_file = 1;

    // The size of the array padding. Used to prevent false sharing.
    // Equal to the number of bytes left over in the cache line containing the final element in float array.
    const int cache_padding_size = 64 - ((W * sizeof(float)) % 64);

    if (posix_memalign((void**)&padded_outputs.arr, 64, total_strides_width * total_strides_height * sizeof(float)) != 0){
        printf("Error allocating memory for padded output.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return 1;
    }
    padded_outputs.padding = cache_padding_size == 64 ? NULL : (char*)malloc(cache_padding_size);

    // Timing begins here, because implementation only starts here.
    MPI_Barrier(MPI_COMM_WORLD); // Make sure all processes start at the same time, for accurate benchmarks.
    const double start_time = omp_get_wtime();

    if (conv2d_stride(feature_map, rowCount, W, kernel, kH, kW, sH, sW, padding_width, padding_height, start_index, padded_outputs) != 0) {
        should_write_to_file = 0;
    }

    // Need to wait for every process to be done.
    MPI_Barrier(MPI_COMM_WORLD);


    if (benchmark_mode == 1 && rank == 0) { printf("%f\n", (omp_get_wtime() - start_time));}
    if (multi_benchmark_mode == 1 && rank == 0) { average_time += (omp_get_wtime() - start_time); }

    // ~~~~~~~~~~~~~~ 7. Write to Output ~~~~~~~~~~~~~~ //

    if (output_file != NULL){

        // TODO: IMPORTANT, replace all of this with other MPI calls, like MPI_Gather or MPI_Gatherv, or MPI_SUM

        // Sync up the total height of the feature map
        int total_height = total_strides_height;
        if (rank > 0){
            MPI_Recv(&total_height, 1, MPI_INT, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            const int height_to_add = should_write_to_file == 1 ? total_strides_height : 0; // This is to avoid adding any height if this process shouldn't output.
            total_height = total_height + height_to_add;
        }
        const int destination = rank+1 < world_size ? rank+1 : 0;

        MPI_Send(&total_height, 1, MPI_INT, destination, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            MPI_Recv(&total_height, 1, MPI_INT, world_size-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        
        // Create status code for when a process is finished writing to file.
        int finished_code;

        if (rank == 0){
            finished_code = write_data_to_file(output_file, NULL, padded_outputs, total_strides_height, total_strides_width, 0, 0, 1, total_strides_width, total_height);
        } else {
            MPI_Recv(&finished_code, 1, MPI_INT, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (finished_code == 0 && should_write_to_file == 1){
                finished_code = write_data_to_file(output_file, NULL, padded_outputs, total_strides_height, total_strides_width, 0, 0, 0, 0, 0); // Does not append dimensions
            }
        }

        if(rank + 1 < world_size){
            MPI_Send(&finished_code, 1, MPI_INT, rank+1, 0, MPI_COMM_WORLD);
        }

        // Free any remaining memory
        //if (outputs != NULL) {free(outputs); outputs = NULL;}
        if (padded_outputs.arr != NULL) { free(padded_outputs.arr); padded_outputs.arr = NULL; }
        if (padded_outputs.padding != NULL) { free(padded_outputs.padding); padded_outputs.padding = NULL;}

    }
    
    if (feature_map != NULL) {free(feature_map); feature_map = NULL; }
    if (kernel != NULL) {free(kernel); kernel = NULL; }
    
    // Make sure all processes have finished with this loop, before we start the next one.
    MPI_Barrier(MPI_COMM_WORLD);

    } // End of loop for multi_benchmark_mode

    if (multi_benchmark_mode == 1 && rank == 0) {printf("Threads=%d, sH=%d, sW=%d, Average Time:  %0.15f\n", threads, sH, sW, average_time/max_iterations); }

    MPI_Finalize();

    return 0;
}