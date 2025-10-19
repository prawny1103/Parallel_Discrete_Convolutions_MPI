// Name: Liam Hearder       Student Number: 23074422
// Name: Pranav Menon       Student Number: 24069351


// ~~~~~~~~~~~~~~~ CONTENTS ~~~~~~~~~~~~~~~~ //
//  1. Includes, Macros, Structures          //
//  2. extract_dimensions()                  //
//  3. extract_data()                        //
//  4. conv2d_stride()                       //
//  5. find_char_count()                     //
//  6. write_data_to_file()                  //
//  7. generate_data()                       //
//  8. main()                                //
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //




/* ~~~~~~~~~~~~~~~~~~~~ Includes ~~~~~~~~~~~~~~~~~~~~ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <omp.h>
#include <mpi.h>



/* ~~~~~~~~~~~~~~~~~~~~ Macros ~~~~~~~~~~~~~~~~~~~~ */

#define FLOAT_STRING_LENGTH 6       // Each input float will be "0.XXX", 5 chars. But all have a space/new line afterwards, so 6.

// Maths macros
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define POW(base, exponent) (({ int RESULT = 1; for (int INDEX = 0; INDEX < exponent; INDEX++) { RESULT *= (base); } RESULT; }))
#define ROUND(number) ( ((int)((number) + 0.5)))
#define TOTAL_STRIDES(size, stride) ((size>0) ? (((size-1)/stride) + 1) : (0))
#define ROUNDF(number, precision) ( (float)(ROUND(number * POW(10, precision))) / (float)(POW(10, precision)) )

// Macro for converting 2D indices to a 1D index
#define IDX(row, col, step) ((long)((row)) * (long)((step)) + (long)((col)))

// During development, my 7 key broke and now gets no response. This makes using && very hard without copy-paste. So I made this.
#define and &&



/* ~~~~~~~~~~~~~~~~~~~~ Structures ~~~~~~~~~~~~~~~~~~~~ */

/**
 *  A struct to hold a float array and its padding, to prevent false sharing.
 * @param   arr     Float array. Generally used to store all data being written to by threads.
 * @param   padding Data used specifically for padding. This avoids multiple threads writing to the same cache line.
 */ 
typedef struct {
    float* arr;
    char* padding;
} f_FloatArray;

/** 
 * Only used by data generation to tell the function `generate_data()` where to start/end with data inputs.
 * @param   start_row   
 * @param   start_col
 * @param   end_row
 * @param   end_col
 */
typedef struct {
    long start_row;
    long start_col;
    long end_row;
    long end_col;
} f_StartEnd;

/**
 * A structure containing information about the currently running process.
 * @param   rank        The rank of the running process.
 * @param   world_size  The total number of running processes.
 * @param   comm        The communicator that this process uses.
 */
typedef struct {
    int rank;
    int world_size;
    MPI_Comm comm;
} f_MPI;

/**
 * A structure containing information about all inputs provided to the program.
 * Does not include debug or benchmark flags, as these are never passed to functions.
 * @param  H               Height of the Feature Map.       `-H <int>`
 * @param  W               Width of the Feature Map.        `-W <int>`
 * @param  kH              Height of the Kernel.            `-kH <int>`
 * @param  kW              Width of the Kernel.             `-kW <int>` 
 * @param  sW              Stride width.                    `-sW <int>`
 * @param  sH              Stride height.                   `-sH <int>`
 * @param  feature_file    Filepath to the Feature Map.     `-f <path>`
 * @param  kernel_file     Filepath to the Kernel.          `-g <path>`
 * @param  output_file     Filepath to the Output.          `-o <path>`
 * @param  threads         Number of threads to use.        `-t <int>`
 */
typedef struct {
    long H;
    long W;
    long kH;
    long kW;
    long sW;
    long sH;
    char* feature_file;
    char* kernel_file;
    char* output_file;
    int threads;
} f_InputData;



/* ~~~~~~~~~~~~~~~~~~~~ Functions ~~~~~~~~~~~~~~~~~~~~ */


/**
 * This is a utility function only meant to clear a file.
 * @param   filepath    The path to the file being emptied/cleared.
 */
void clear_file(char* filepath){
    FILE *fp = fopen(filepath, "w");
    fclose(fp);
    return;
}


/*
* Extracts the dimensions from a file.
* @param filepath     The filepath where the data is stored.
* @param height       Pointer to the location where the height will be stored.
* @param width        Pointer to the location where the width will be stored.
*/
int extract_dimensions(char* filepath, long* height, long* width) {

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


/** 
* Reads an input file and extracts data into an output. 
* @param filepath           The filepath where the data is stored.
* @param width              The number of elements in each line. Width.
* @param height             The number of rows. Height.
* @param padding_width      The number of zeroes to pad the width with.
* @param padding_height     The number of zeroes to pad the height with.
* @param start_index        Where to start extracting data. Used to divide up work for MPI. To extract all data, set this to zero. To make an "end_index", use "height".
* @param sW                 Stride width.
* @param sH                 Stride height.
* @param output             The stream into which the inputs will be stored.
* @param process_data       Information about the currently running process.
*/
int extract_data(char* filepath, int width, int height, int padding_width, int padding_height, int start_index, float* *output, f_MPI process_data) {

    if (filepath == NULL){ return 1; }
    FILE* file_ptr = fopen(filepath, "r");
    if (file_ptr == NULL){ return 1; }

    // Create a buffer to place extracted strings into
    const size_t buffer_size = (FLOAT_STRING_LENGTH * width) + 2; // +2 for new-line and null-byte
    char* buffer = (char*)malloc(buffer_size);

    const long rows_to_read = height + padding_height*2;
    const long total_width = width + padding_width*2;
    const long row_offset = (process_data.rank == 0) ? padding_height : 0;
    
    // Safely get the header here so we can ignore later
    fgets(buffer, buffer_size, file_ptr);

    // Skip all lines before the start
    for (long i = 0; i < start_index-1; i++){
        fgets(buffer, buffer_size, file_ptr);
    }

    // Iterates over every row
    for (long i = 0; i < rows_to_read; i++){
        if (fgets(buffer, buffer_size, file_ptr) == NULL) { break; } // Exit if invalid
        char* token = strtok(buffer, " ");

        // Iterates over every column in the row
        for (long j = 0; j < width; j++) {
            if (token != NULL){

                // Extracted float data from the token 
                const float element = (float)atof(token);

                // Calculations for where to put the extracted data into the output array
                const long output_col_index = i + row_offset;
                const long output_row_index = j + padding_width;
                const long index = IDX(output_col_index, output_row_index, total_width); 
                
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
* @param    f               Pointer to the Feature Map.
* @param    H               Height of the Feature Map.
* @param    W               Width of the Feature Map.
* @param    g               Pointer to the Kernel.
* @param    inputs          Information about the input paramters passed to this program.
* @param    w_padding       Width of the padding in the Feature Map.
* @param    h_padding       Height of the padding in the Feature Map.
* @param    start_index     The starting index location, of index 0 in the local Feature Map, in the original Feature Map.
                                This is used for calculating stride at runtime, to determine which rows should be skipped.
* @param    padded_output   Location where outputs are stored.
*/
int conv2d_stride(float* f, int H, int W, float* g, f_InputData inputs, int w_padding, int h_padding, int start_index, f_FloatArray padded_output){
    
    // Used to stop a process from writing if it did no work.
    bool any_written = false;

    const long total_height = H + h_padding*2;
    const long total_width = W + w_padding*2;
    const long n_end = total_height - h_padding;
    const long k_end = total_width - w_padding;
    const long total_strides_width = TOTAL_STRIDES(W, inputs.sW);

    // dimensions for convolution window
    const long M = (inputs.kH - 1) / 2;
    const long N = (inputs.kW - 1) / 2;

    // Early exits for issues
    if (H < 1 || W < 1 || inputs.H < 1 || inputs.W < 1)     { return any_written; }
    if (f == NULL || g == NULL || padded_output.arr == NULL){ return any_written; }
    if (w_padding < 0 || h_padding < 0 || start_index < 0)  { return any_written; }    
    
    // Swap between Serialized and Parallel
    if (inputs.threads == 1) {

        // ~~~~~~~~~~~~ Serial ~~~~~~~~~~~~ //
        for (long n = h_padding; n < n_end; n++){
            for (long k = w_padding; k < k_end; k=k+inputs.sW){
                if (( start_index + n-h_padding) % inputs.sH != 0){ continue; }
                float result = 0.0;
                for (long i = 0; i < inputs.kH; i++){
                    for (long j = 0; j < inputs.kW; j++){
                        result += f[(n+i-M) * (total_width) + (k+j-N)] * g[i * inputs.kW + j];
                    }
                }
                padded_output.arr[(long)(((n - h_padding)/inputs.sH) * total_strides_width + ((k - w_padding)/inputs.sW))] = result;
                any_written = true;
            }
        }

    } else {

        // ~~~~~~~~~~~~ Parallel ~~~~~~~~~~~~ //
        #pragma omp parallel for collapse(2) schedule(static, W) 
        for (long n = h_padding; n < n_end; n++){
            for (long k = w_padding; k < k_end; k=k+inputs.sW){
                if (( start_index + n-h_padding) % inputs.sH != 0){ continue; }
                float result = 0.0;
                #pragma omp simd collapse(2) reduction(+:result)
                for (long i = 0; i < inputs.kH; i++){
                    for (long j = 0; j < inputs.kW; j++){
                        result += f[(n+i-M) * (total_width) + (k+j-N)] * g[i * inputs.kW + j];
                    }
                }
                padded_output.arr[(long)(((n - h_padding)/inputs.sH) * total_strides_width + ((k - w_padding)/inputs.sW))] = result;
                any_written = true;
            }
        }

    }
    return any_written;
}


/**
 * Finds the number of characters in an integer.
 * @param   number    The integer to find the number of characters in.
 */
int find_char_count(int number){
    if (number < 10) return 1;
    return find_char_count(number/10) + 1;
}


/**
 * Writes outputs to a file. This is a multi-process function that utilises MPI to write data to a file across all processes.
 * @param filepath              The filepath of where to find/put the output file.
 * @param outputs               A 2d array of float32s. This is used only for kernel/feature map writing.
 * @param lines_to_write        The number of rows that should be written to file.
 * @param height                The height of the outputs.
 * @param width                 The width of the outputs.
 * @param w_dimension           The height dimension to write to the file.
 * @param h_dimension           The width dimension to write to the file.
 * @param inputs                Information about the input paramters passed to this program.
 * @param process_data          Information about the currently running process.
 * @param root_only             Whether or not this function should be run by the root process only.
*/
int write_data_to_file(char* filepath, float* outputs, int lines_to_write, int height, int width, int h_padding, int w_padding, bool should_write, f_InputData inputs, f_MPI process_data, bool root_only){

    // Open the file
    MPI_File handle;
    int access_mode = MPI_MODE_CREATE | MPI_MODE_WRONLY;
    if (MPI_File_open(process_data.comm, filepath, access_mode, MPI_INFO_NULL, &handle) != MPI_SUCCESS) {
        printf("Process %d, Failure in opening file.\n", process_data.rank);
    }

    if (process_data.rank == 0){
        clear_file(filepath);
    }

    const bool this_process_writes = should_write && (!root_only || process_data.rank == 0);

    MPI_Barrier(process_data.comm);

    // number of chars in the dimensions header
    const int dimensions_offset = find_char_count(lines_to_write) + 1 + find_char_count(width) + 1;

    // MAke a much larger estimate for the size of the line than necessary. This is to ensure they fit.
    const int largest_float_possible = find_char_count(inputs.kH*inputs.kW);
    size_t buffer_bytes = lines_to_write * width * (largest_float_possible + 5) + (process_data.rank == 0 ? (size_t)dimensions_offset : 0) + 1; // +5 because ".XXX " and +1 because null-byte

    char* buffer;
    size_t bytes_written = 0;

    if (this_process_writes == 1){
        buffer = malloc(buffer_bytes);
        char* moving_ptr = buffer;

        // fill header for rank 0
        if (process_data.rank == 0) {
            moving_ptr += sprintf(moving_ptr, "%ld %ld\n", TOTAL_STRIDES(inputs.H, inputs.sH), TOTAL_STRIDES(inputs.W, inputs.sW));
        }

        for (long i = 0; i < lines_to_write; i++){
            for (long j = 0; j < width; j++){
                moving_ptr += sprintf(moving_ptr, "%.3f", outputs[IDX(i, j, width)]);   // Add float to buffer
                moving_ptr += sprintf(moving_ptr, (j<width-1) ? (" ") : ("\n"));        // Add a new-line if at end of row, else add a space.
            }
        }
        // Get the total number of bytes we've actually written
        bytes_written = moving_ptr - buffer;

        if (bytes_written > buffer_bytes) {
            printf("Buffer Overflow! Tried to write %ld bytes to %ld bytes in memory...\n", bytes_written, buffer_bytes);
            free(buffer);
        }
    }

    

    

    MPI_Offset write_offset;
    MPI_Exscan(&bytes_written, &write_offset, 1, MPI_UNSIGNED_LONG, MPI_SUM, process_data.comm);

    if (process_data.rank == 0){ write_offset = 0; }

    MPI_File_write_at_all(handle, write_offset, buffer, bytes_written, MPI_CHAR, MPI_STATUS_IGNORE);
    if (this_process_writes == 1) free(buffer);
    MPI_File_close(&handle);



    return 0;
}


/**
 * Generates a 2d array of random floats.
 * @param   height      The height of the array.
 * @param   width       The width of the array.
 * @param   h_padding   Height padding. Data will only be written in non-padded rows.
 * @param   w_padding   Width Padding. Data will only be written in non-padded columns.
 * @param   output      The location where the generated data will be stored.
*/
int generate_data(int width, f_StartEnd start_end, float* *output, int seed){

    // Make a new random seed. This stops f from being the same as g when the code runs too fast.
    if (seed == -1) {
        srand(time(NULL));
        seed = rand();
        
    }

    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (long i = start_end.start_row; i < start_end.end_row; i++) {
            unsigned int row_seed = seed + (unsigned int)i;
            
            size_t row_offset = (size_t)i * (size_t)width;
            
            for (long j = start_end.start_col; j < start_end.end_col; j++) {
                (*output)[row_offset + j] = (float)rand_r(&row_seed) / (float)RAND_MAX;
            }
        }
    }
    return 0;
}


int main(int argc, char** argv) {

    /* ~~~~~~~~~~~~~~~ MAIN CONTENTS ~~~~~~~~~~~~~~ //
        1. Argument Extraction
        2. MPI Initialisation

        3. Error Handling
        4. Kernel
            a) generation,
            b) extraction
        5. Feature Map
            a) generation, 
            b) extraction
        6. Convolutions
        7. Write to Output
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

    

    // ~~~~~~~~~~~~~~~ 1. Argument Extraction ~~~~~~~~~~~~~~ //

    // Default input values
    f_InputData inputs = {
        0,      // H
        0,      // W
        0,      // kH
        0,      // kW
        1,      // sW
        1,      // sH
        NULL,   // feature_file
        NULL,   // kernel_file
        NULL,   // output_file
        1       // threads
    };

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
            inputs.H = atoi(argv[++i]) > 0 ? atoi(argv[i]) : 1;
            continue;
        }
        if (strcmp(argv[i], "-W") == 0) {
            if (i + 1 >= argc) { printf("Incorrect usage of -W flag. Please provide an input width.\n"); return 1; }
            inputs.W = atoi(argv[++i]) > 0 ? atoi(argv[i]) : 1;
            continue;
        }
        if (strcmp(argv[i], "-kH") == 0) {
            if (i + 1 >= argc) { printf("Incorrect usage of -kH flag. Please provide a kernel height.\n"); return 1; }
            inputs.kH = atoi(argv[++i]) > 0 ? atoi(argv[i]) : 1;
            continue;
        }
        if (strcmp(argv[i], "-kW") == 0) {
            if (i + 1 >= argc) { printf("Incorrect usage of -kW flag. Please provide a kernel width.\n"); return 1; }
            inputs.kW = atoi(argv[++i]) > 0 ? atoi(argv[i]) : 1;
            continue;
        }
        if (strcmp(argv[i], "-sW") == 0) {
            if (i + 1 >= argc) { printf("Incorrect usage of -sW flag. Please provide a stride width.\n"); return 1; }
            inputs.sW = atoi(argv[++i]) > 0 ? atoi(argv[i]) : 1;
            continue;
        }
        if (strcmp(argv[i], "-sH") == 0) {
            if (i + 1 >= argc) { printf("Incorrect usage of -sH flag. Please provide a stride height.\n"); return 1; }
            inputs.sH = atoi(argv[++i]) > 0 ? atoi(argv[i]) : 1;
            continue;
        }
        if (strcmp(argv[i], "-f") == 0) {
            if (i + 1 >= argc) { printf("Incorrect usage of -f flag. Please provide a filepath.\n"); return 1; }
            inputs.feature_file = argv[++i];
            continue;
        }
        if (strcmp(argv[i], "-g") == 0) {
            if (i + 1 >= argc) { printf("Incorrect usage of -g flag. Please provide a filepath.\n"); return 1; }
            inputs.kernel_file = argv[++i];
            continue;
        }
        if (strcmp(argv[i], "-o") == 0) {
            if (i + 1 >= argc) { printf("Incorrect usage of -o flag. Please provide a filepath.\n"); return 1; }
            inputs.output_file = argv[++i];
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
            inputs.threads = atoi(argv[++i]) > 0 ? atoi(argv[i]) : 1;
            continue;
        } 
    }

    omp_set_num_threads(inputs.threads);

    // ~~~~~~~~~~~~~~ 2. MPI Initialisation ~~~~~~~~~~~~~~ //

    int provided;
    // MPI_THREAD_MULTIPLE
    // MPI_THREAD_FUNNELED
    MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided);

    f_MPI process_data = {0};
    process_data.comm = MPI_COMM_WORLD;
    MPI_Comm_rank(process_data.comm, &process_data.rank);
    MPI_Comm_size(process_data.comm, &process_data.world_size);



    // ~~~~~~~~~~~~~~~ 3. Error Handling ~~~~~~~~~~~~~~ //

    if (process_data.rank == 0){
        if (benchmark_mode) { 
            if (inputs.threads > 1){ printf("Beginning Parallel Convolutions across %d processes with %d threads...\n", process_data.world_size, inputs.threads);} 
            else { printf("Beginning Serial Convolutions across %d processes...\n", process_data.world_size); }
        }
        if (inputs.H < 0 || inputs.W < 0 || inputs.kH < 0 || inputs.kW < 0 || inputs.threads < 1 || max_iterations < 1){
            printf("Please provide only positive integers for dimensions and threads.\n");
            MPI_Abort(process_data.comm, EXIT_FAILURE);
            return 1;
        }
        if (inputs.H == 0 && inputs.W == 0 && inputs.feature_file == NULL){
            printf("Please provide either a feature map file or dimensions to generate one.\n");
            MPI_Abort(process_data.comm, EXIT_FAILURE);
            return 1;
        }
        if (inputs.kH == 0 && inputs.kW == 0 && inputs.kernel_file == NULL){
            printf("Please provide either a kernel file or dimensions to generate one.\n");
            MPI_Abort(process_data.comm, EXIT_FAILURE);
            return 1;
        }

        if (multi_benchmark_mode && (inputs.feature_file || inputs.kernel_file)) { 
            printf("Do not input a file while running multi-benchmark mode.\n");
            MPI_Abort(process_data.comm, EXIT_FAILURE);
            return 1;
        }
    }
    

    MPI_Barrier(process_data.comm); 

    double average_time = 0.0f;
    for (int mb_iteration = 0; mb_iteration < max_iterations; mb_iteration++){

    // Seed for random generation later. This ensures the seed is identical across all processes.
    time_t featureMapSeed;
    if (process_data.rank == 0) { featureMapSeed = time(0);}
    MPI_Bcast(&featureMapSeed, 1, MPI_LONG, 0, process_data.comm);



    // ~~~~~~~~~~~~~~ 4. Kernel Generation / Extraction ~~~~~~~~~~~~~~ //

    float* kernel = NULL;

    // Generate Kernel
    if (inputs.kH > 0 || inputs.kW > 0){

        // Allows users to specify only 1 dimension, and prevents them from inputting negative numbers
        inputs.kH = MAX(inputs.kH, 1);
        inputs.kW = MAX(inputs.kW, 1);

        // Allocating memory
        if (posix_memalign((void**)&kernel, 64, (long)(inputs.kW) * (long)(inputs.kH) * sizeof(float)) != 0){
            printf("Error allocating memory for kernel.\n");
            return 1;
        }
        f_StartEnd kernel_generation_data = {0};
        kernel_generation_data.start_col = 0; 
        kernel_generation_data.start_row = 0;
        kernel_generation_data.end_col = inputs.kW;
        kernel_generation_data.end_row = inputs.kH;


        generate_data(inputs.kW, kernel_generation_data, &kernel, -1);

        // If wanting to save inputs, write to kernel file
        if (inputs.kernel_file != NULL){
            if (write_data_to_file(inputs.kernel_file, kernel, inputs.kH, inputs.kH, inputs.kW, 0, 0, true, inputs, process_data, true) != 0) {
                MPI_Abort(process_data.comm, EXIT_FAILURE);
                return 1;
            }
        }

    // Extract Kernel
    } else if (inputs.kernel_file != NULL){

        // Extracting dimensions
        if (extract_dimensions(inputs.kernel_file, &(inputs.kH), &(inputs.kW)) != 0){ 
            printf("Error extracting kernel dimensions from file.\n");
            MPI_Abort(process_data.comm, EXIT_FAILURE);
            return 1;
        }
        
        // Allocating memory
        if (posix_memalign((void**)&kernel, 64, (long)(inputs.kW) * (long)(inputs.kH) * sizeof(float)) != 0){
            printf("Error allocating memory for kernel.\n");
            MPI_Abort(process_data.comm, EXIT_FAILURE);
            return 1;
        }

        // Extracting data
        if (extract_data(inputs.kernel_file, inputs.kW, inputs.kH, 0, 0, 0, &kernel, process_data) != 0){
            printf("Error extracting kernel data from file.\n");
            MPI_Abort(process_data.comm, EXIT_FAILURE);
            return 1;
        }
    }


    // This is the "same padding" that'll be added to the feature map.
    const int padding_width = inputs.kW / 2;
    const int padding_height = inputs.kH / 2;


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
    if (inputs.H > 0 || inputs.W > 0){

        // Allows users to specify only 1 dimension, and prevents them from inputting negative numbers
        inputs.H = MAX(inputs.H, 1);
        inputs.W = MAX(inputs.W, 1);

        start_index = process_data.rank * (inputs.H / process_data.world_size) + MIN(process_data.rank, (inputs.H % process_data.world_size));

        const int total_width = inputs.W + padding_width*2;
        //const int total_height = H + padding_height*2;

        // Determine the number of rows each process should use in convolutions.
        rowCount = (inputs.H / process_data.world_size) + (process_data.rank < (inputs.H % process_data.world_size) ? 1 : 0);
        
        // Total number of rows that each process should create memory for. This includes the rows it will do convolutions on (rowCount) AND padding.
        totalRowCount = rowCount + padding_height*2;

        const long startRow = (rowCount * process_data.rank) + padding_height*(process_data.rank==0?1:0);

        f_FloatArray feature_map_padded = {0};
        const int cache_padding_size = 64 - ((inputs.W * sizeof(float)) % 64);

        if (posix_memalign((void**)&feature_map_padded.arr, 64, (long)(total_width) * (long)(totalRowCount) * sizeof(float)) != 0){
            printf("Error allocating memory for padded output.\n");
            MPI_Abort(process_data.comm, EXIT_FAILURE);
            return 1;
        }
        feature_map_padded.padding = cache_padding_size == 64 ? NULL : (char*)malloc(cache_padding_size);

        // Add zeroes as padding
        for (long i = 0; i < total_width * totalRowCount; i++){
            feature_map_padded.arr[i] = 0.0f;
        }

        feature_map = feature_map_padded.arr;

        srand(featureMapSeed);
        
        long values_to_skip = startRow * inputs.W;
        for (long i = 0; i < values_to_skip; i++) {
            rand();
        }

        f_StartEnd generation_data = {0};
        generation_data.start_row   = (process_data.rank==0) ? padding_height : 0;
        generation_data.end_row     = padding_height + rowCount + ((process_data.rank == process_data.world_size - 1) ? 0 : padding_height);
        generation_data.start_col   = padding_width;
        generation_data.end_col     = total_width - padding_width;

        generate_data(total_width, generation_data, &feature_map_padded.arr, featureMapSeed);

        

        // for (int i = 0; i < totalRowCount; i++){
        //     for (int j = 0; j < total_width; j++) {
        //         printf("%.3f ", feature_map_padded.arr[IDX(i,j,total_width)]);
        //     }
        //     printf("\n");
        // }

        // If wanting to save inputs, write to feature file
        if (inputs.feature_file != NULL){

            float* temp_data;
                if (posix_memalign((void**)&temp_data, 64, (long)(inputs.W) * (long)(rowCount) * sizeof(float)) != 0){
                printf("Error allocating memory for padded output.\n");
                MPI_Abort(process_data.comm, EXIT_FAILURE);
                return 1;
            }

            for (long i = 0; i < (long)(rowCount); i++){
                for (long j = 0; j < (long)(inputs.W); j++){
                    temp_data[IDX(i,j,inputs.W)] = feature_map[IDX(i+padding_height, j+padding_width, total_width)];
                }
            }

            write_data_to_file(inputs.feature_file, temp_data, rowCount, inputs.H, inputs.W, padding_height, padding_width, true, inputs, process_data, false);

            // todo remove
            //printf("Process %d got here\n", process_data.rank);
            free(temp_data);
        }


    // Extract Feature Map
    } else if (inputs.feature_file != NULL) {

        // Extract dimensions of the feature map
        if (extract_dimensions(inputs.feature_file, &(inputs.H), &(inputs.W)) != 0){ 
            printf("Error extracting feature map dimensions from file.\n");
            MPI_Abort(process_data.comm, EXIT_FAILURE);
            return 1;
        }

        const int total_width = inputs.W + padding_width*2;

        // Determine the number of rows each process should use in convolutions.
        // This ONLY thinks about the outer loop iterations. The real data size will have padding_height*2 added (because inner loops use more rows)
        rowCount = (inputs.H / process_data.world_size) + (process_data.rank < (inputs.H % process_data.world_size) ? 1 : 0);

        // This just includes padding
        totalRowCount = rowCount + padding_height*2;
        

        start_index = MAX(0, process_data.rank * (inputs.H / process_data.world_size) + MIN(process_data.rank, (inputs.H % process_data.world_size)) - MAX(0, padding_height-1));


        // Allocate memory for the feature map of the feature map.
        if (posix_memalign((void**)&feature_map, 64, (long)(total_width) * (long)(totalRowCount) * sizeof(float)) != 0){
            printf("Error allocating memory for feature map.\n");
            MPI_Abort(process_data.comm, EXIT_FAILURE);
            return 1;
        }

        // Add zeroes as padding
        for (long i = 0; i < total_width * totalRowCount; i++){
            feature_map[i] = 0.0f;
        }

        // Extract Feature Map
        if (extract_data(inputs.feature_file, inputs.W, rowCount, padding_width, padding_height, start_index, &feature_map, process_data) != 0){
            printf("Error extracting feature map data from file.\n");
            MPI_Abort(process_data.comm, EXIT_FAILURE);
            return 1;
        } 
    }


    // ~~~~~~~~~~~~~~ 6. Convolutions ~~~~~~~~~~~~~~ //
    
    // Check if we have all the inputs we need to perform convolutions
    if (kernel == NULL || feature_map == NULL){
        printf("To generate an output, please provide all inputs.\n");
        MPI_Abort(process_data.comm, EXIT_FAILURE);
        return 1;
    }

    // Defining output pointers
    f_FloatArray padded_outputs = {0};   // Used for parallel convolution
    
    // This is how many times a convolution takes place over each row/col. Also used to determine output array size.
    const int total_strides_width = TOTAL_STRIDES(inputs.W, inputs.sW);
    const int total_strides_height = TOTAL_STRIDES(rowCount, inputs.sH);

    // The size of the array padding. Used to prevent false sharing.
    // Equal to the number of bytes left over in the cache line containing the final element in float array.
    const int cache_padding_size = 64 - ((inputs.W * sizeof(float)) % 64);

    if (posix_memalign((void**)&padded_outputs.arr, 64, (long)(total_strides_width) * (long)(total_strides_height) * sizeof(float)) != 0){
        printf("Error allocating memory for padded output.\n");
        MPI_Abort(process_data.comm, EXIT_FAILURE);
        return 1;
    }
    padded_outputs.padding = cache_padding_size == 64 ? NULL : (char*)malloc(cache_padding_size);


    MPI_Barrier(process_data.comm);


    // Timing begins here because convolutions start here.
    const double start_time = omp_get_wtime();

    // Perform convolutions
    bool wrote_anything = conv2d_stride(feature_map, rowCount, inputs.W, kernel, inputs, padding_width, padding_height, start_index, padded_outputs);


    // Need to wait for every process to be done before we can write to file (or benchmark).
    MPI_Barrier(process_data.comm);

    if (benchmark_mode == 1 && process_data.rank == 0) { printf("%f\n", (omp_get_wtime() - start_time));}
    if (multi_benchmark_mode == 1 && process_data.rank == 0) { average_time += (omp_get_wtime() - start_time); }



    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 7. Write to Output ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

    if (inputs.output_file != NULL){
        write_data_to_file(inputs.output_file, padded_outputs.arr, total_strides_height, inputs.H, total_strides_width, 0, 0, wrote_anything, inputs, process_data, false);
    }



    // Free any remaining memory
    if (padded_outputs.arr != NULL) { free(padded_outputs.arr); padded_outputs.arr = NULL; }
    if (padded_outputs.padding != NULL) { free(padded_outputs.padding); padded_outputs.padding = NULL;}
    if (feature_map != NULL) {free(feature_map); feature_map = NULL; }
    if (kernel != NULL) {free(kernel); kernel = NULL; }
    
    // Make sure all processes have finished with this loop, before we start the next one.
    MPI_Barrier(process_data.comm);

    } // End of loop for multi_benchmark_mode

    if (multi_benchmark_mode == 1 && process_data.rank == 0) {printf("Threads=%d, sH=%ld, sW=%ld, Average Time:  %0.15f\n", inputs.threads, inputs.sH, inputs.sW, average_time/max_iterations); }

    MPI_Finalize();

    return 0;
}