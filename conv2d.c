#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* The string length of every float in the feature map. Example line: "0.594 0.934 0.212\n". 
So, 3 floats, each looks like "X.XXX" which is 5 chars, but then all have a space or new-line 
character. */
#define FLOAT_STRING_LENGTH 6

#define max(a,b) (((a) > (b)) ? (a) : (b))

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
    Reads an input file and extracts data into an output. 
    @param filepath         The filepath where the data is stored.
    @param width            The number of elements in each line. Width.
    @param height           The number of rows. Height.
    @param padding_width    The number of zeroes to pad the width with.
    @param padding_height   The number of zeroes to pad the height with.
    @param output           The stream into which the inputs will be stored.
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


// Function to perform 2D discrete convolution
int conv2d(float** f, int H, int W, float** g, int kH, int kW, int w_padding, int h_padding, float** output){

    const int total_height = H + h_padding*2;
    const int total_width = W + w_padding*2;

    int max_height = H;
    int max_width = W;

    // Iterate every value in the feature map
    for (int n = h_padding; n < total_height-h_padding; n++){
        for (int k = w_padding; k < total_width-w_padding; k++){
            
            // dimensions for convolution window
            int M = kH / 2;
            int N = kW / 2;
            float result = 0;

            // Convolution formula applied here, extra dimension (N) to make it 2d
            // Iterate every value in the kernel
            for (int i = 0-M; i <= M; i++){
                for (int j = 0-N; j <= N; j++){
                    int col = n+i;
                    int row = k+j;

                    // Stop if we're about to check a row/column that's OOB
                    if (0 <= row && row < total_width && 0 <= col && col < total_height){
                        result = result + (f[col][row] * g[i+M][j+N]);
                    }
                }
            }
            output[n-h_padding][k-w_padding] = result;
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


int main(int argc, char** argv) {
    
    // Initialising variables for future use
    // TODO: we should alignas(64) all of these, to avoid False Sharing
    int H = 0;
    int W = 0;
    int kH = 0;
    int kW = 0;
    char* feature_file = NULL;
    char* kernel_file = NULL;
    char* output_file = NULL;

    // Extract arguments into their variables
    for (int i = 1; i < argc; i++) {

        if (i + 1 >= argc) { break; }

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
    }
    
    /* 
    TODO: Error catching for incorrect flag usage
        
        Examples: 
        1. No flags provided, 
        2. Generating array but provided only height not width, 
        3. Not generating and provided feature but no kernel
        4. Provided output, but generated/provided no input

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

    // Extract Kernel
    } else {

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
    }

    // This is the "same padding" that'll be added to the feature map.
    const int padding_width = kW / 2;
    const int padding_height = kH / 2;

    
    
    // ~~~~~~~~~~~~~~ FEATURE MAP ~~~~~~~~~~~~~~ // 

    float** feature_map;

    // Generate Feature Map
    if (H > 0 || W > 0){

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

    // Extract Feature Map
    } else {
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
    }

    
    // ~~~~~~~~~~~~~~ conv2d() ~~~~~~~~~~~~~~ //
    

    if (output_file != NULL){

        if (kernel[0] == NULL || feature_map[0] == NULL){
            printf("To generate an output, please provide all inputs.\n");
            return 1;
        }

        // Allocating memory
        float** outputs = (float**)malloc(H * sizeof(float*));
        for (int i = 0; i < H; i++){
            outputs[i] = (float*)calloc(W, sizeof(float));
        }
        
        // Convolutions
        int status = conv2d(feature_map, H, W, kernel, kH, kW, padding_width, padding_height, outputs);
        if (status != 0){
            // TODO: Handle when can't perform convolutions
        }


    // ~~~~~~~~~~~~~~ Write to Output ~~~~~~~~~~~~~~ //

        status = write_data_to_file(output_file, outputs, H, W, 0, 0);
        if (status != 0){
            // TODO: Handle when can't write to output.
        }
    }
    return 0;
}








/* 
PYTHON CODE EXAMPLE
    - This works perfectly (for serial convolution)
    - It does NOT work to the specs of the project, but this is a good starting point for when we need to write conv2d()
    - The variable names all specifically follow the formula in the project description.
    - The variable values of `feature` and `kernel` are exactly the same as in Figure 1 from the project description, showing that it works.


feature = [
    [0.1,1.0,0.1,0.7,0.3,1.0],
    [0.7,0.8,0.1,0.5,0.7,1.0],
    [1.0,0.5,0.2,0.3,0.6,0.9],
    [0.2,0.0,0.9,0.1,0.5,0.8],
    [0.1,0.7,0.9,0.0,0.2,0.6],
    [0.9,0.2,0.6,0.1,0.7,0.4]
    ]
    
kernel = [
    [1.0,0.7,0.6],
    [0.0,0.4,0.1],
    [0.1,0.1,0.5]
    ]
*/

/*
def conv(f,g):
    max_height = len(f) - 2                                     H - 2
    max_width = len(f[0]) - 2                                   W - 2
    output = [[0]*max_width for _ in range(max_height)]         output

    # Loop width/height to make the output array
    for n in range(1, max_width+1):
        for k in range(1, max_height+1):
            
            # dimensions for convolution window
            M = len(g) // 2
            N = len(g[0]) // 2
            result = 0

            # Convolution formula applied here, extra dimension (N) to make it 2d
            for i in range(-M, M+1):
                for j in range(-N, N+1):
                    row = n+i
                    col = k+j

                    # Stop if we're about to check a row/column that's OOB
                    if 0 <= row < len(f) and 0 <= col < len(f[0]): 
                        result += f[row][col] * g[i+M][j+N]
            output[n-1][k-1] = round(result,1)
    return output

output = conv(feature,kernel)
for row in output:
    print(row)

*/
