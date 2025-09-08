#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// The string length of every float in the feature map. Example line: "0.594 0.934 0.212\n". So, 3 floats, each looks like "X.XXX" which is 5 chars, but then all have a space or
#define FLOAT_STRING_LENGTH 6

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

    return 0;
}


/* 
    Reads an input file and extracts data into an output. 
    @param filepath The filepath where the data is stored.
    @param f_width  The number of elements in each line. Width.
    @param f_height The number of rows. Height.
    @param padding_width The number of zeroes to pad the width with.
    @param padding_height The number of zeroes to pad the height with.
    @param output   The stream into which the inputs will be stored.
*/
int extract_data(char* filepath, int f_width, int f_height, int padding_width, int padding_height, float** *output) {
    
    if (filepath == NULL){ return 1; }
    FILE* file_ptr = fopen(filepath, "r");
    if (file_ptr == NULL){ return 1; }

    // Create a buffer to place extracted strings into
    const size_t buffer_size = (FLOAT_STRING_LENGTH * f_width) + 1; // +1 for null-byte
    
    char* buffer = (char*)malloc(buffer_size);

    // Now loop over each line in the file
    int row_index = 0 + padding_height;
    while (row_index < (f_height + padding_height*2)){
        
        if (fgets(buffer, buffer_size, file_ptr) == NULL){
            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ADD SOMETHING HERE TO MAYBE ADD AN EXTRA LINE? BUT ALSO CHECK IF IT ALREADY WORKS?
        }

        if (row_index == 0 + padding_height) {
            row_index++;
            continue;
        }
        char* token = strtok(buffer, " ");

        // Now loop over each number in the line
        int column_index = 0 + padding_width;
        while (token != NULL){
            float element = (float)atof(token);
            (*output)[row_index-1][column_index] = element; // Add to output. Need to do "-1" to avoid the first line.
            token = strtok(NULL, " ");
            column_index++;
        }
        row_index++;
    }
    
    return 0;
}

// Function to perform 2D discrete convolution
int conv2d(float** f, int H, int W, float** g, float kH, int kW, float** output){

    int max_height = H - 2;
    int max_width = W - 2;

    for (int n = 1; n < max_width; n++){
        for (int k = 1; k < max_height; k++){

            // dimensions for convolution window
            int M = kW / 2;
            int N = kH / 2;
            float result = 0;

            // Convolution formula applied here, extra dimension (N) to make it 2d
            for (int i = 0-M; i <= M; i++){
                for (int j = 0-N; j <= N; j++){
                    int row = n+i;
                    int col = k+j;

                    // Stop if we're about to check a row/column that's OOB
                    if (0 <= row && row < W && 0 <= col && col < H){
                        result = result + (f[row][col] * g[i+M][j+N]);
                    }
                }
            }
            output[n-1][k-1] = result;
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


    // TODO: (optionally) generate inputs
    // TODO: Add check to see if the user wants to generate their own feature map.


    // ~~~~~~~~~~~~~~ KERNEL ~~~~~~~~~~~~~~ // 

    // Extracting dimensions
    if (kernel_file != NULL && (kH <= 0 || kW <= 0)){
        int status = extract_dimensions(kernel_file, &kH, &kW);
        if (status != 0){ 
            // TODO: Handle when it can't extract file dimensions
        }
    }

    // Allocating memory
    float** kernel = (float**)malloc(kH * sizeof(float*));
    for (int i = 0; i < kH; i++){
        kernel[i] = (float*)malloc(kW * sizeof(float));
    }

    // Extracting data
    if (kernel_file != NULL){
        int status = extract_data(kernel_file, kW, kH, 0, 0, &kernel);
        for(int i = 0; i < 5; i++){
            //printf("%f\n", kernel[1][i]);
        }
    }

    // This is the "same padding" added to the feature map. Width = 0, Height = 1
    const int padding_width = kW / 2;
    const int padding_height = kH / 2;


    // ~~~~~~~~~~~~~~ FEATURE MAP ~~~~~~~~~~~~~~ // 

    // Extract dimensions of the feature map
    if (feature_file != NULL && (H <= 0 || W <= 0)){
        int status = extract_dimensions(feature_file, &H, &W);
        if (status != 0){ 
            // TODO: Handle when it can't extract file dimensions
        }
    }

    // Allocate memory for the feature map of the feature map.
    float** feature_map = (float**)malloc((H + padding_height) * sizeof(float*));
    for (int i = 0; i < (H + padding_height); i++){
        feature_map[i] = (float*)malloc((W + padding_width) * sizeof(float));
    }
    
    // Add zeroes to each element in the 2d array
    for (int i = 0; i < (W + padding_width); i++){
        for (int j = 0; j < (H + padding_height); j++){
            feature_map[i][j] = 0.0;
        }
    }

    // Extract Feature Map
    if (feature_file != NULL){
        int status = extract_data(feature_file, W, H, padding_width, padding_height, &feature_map);
        for(int i = 0; i <= (W + padding_width); i++){
            for (int j = 0; j <= (H + padding_height); j++){
                printf("%f ", feature_map[i][j]);
            }
            printf("\n");
        }
    }

    



    


    return 0;

    // ~~~~~~~~~~~~~~ conv2d() ~~~~~~~~~~~~~~ //

    // Allocating memory
    float** outputs = (float**)malloc(W * sizeof(float*));
    for (int i = 0; i < H; i++){
        outputs[i] = (float*)malloc(W * sizeof(float));
    }
    
    int status = conv2d(feature_map, H, W, kernel, kH, kW, outputs);
    if (status == 0){
        for (int i = 0; i < 5; i++){
            printf("%f\n", outputs[0][i]);
        }
    }

    
    // TODO: (optionally) write to output file

    // ...

    // profit?
        
    
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
