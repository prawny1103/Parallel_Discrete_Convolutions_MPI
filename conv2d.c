#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// The string length of every float in the feature map. Example line: "0.594 0.934 0.212\n". So, 3 floats, each looks like "X.XXX" which is 5 chars, but then all have a space or
#define FLOAT_STRING_LENGTH 6

int extract_dimensions(char* filepath, int* height, int* width) {

    if (filepath == NULL) { return 1; }
    char firstline[16];

    FILE* file_ptr = fopen(filepath, "r");

    // Reads the first line
    fgets(firstline, sizeof(firstline), file_ptr);

    char* token = strtok(firstline, " ");
    *height = atoi(token);
    token = strtok(NULL, " ");
    *width = atoi(token);

    return 0;
}

/* 
    Reads an input file and extracts data into an output. 
    @param filepath The filepath where the feature map is stored.
    @param f_width  The number of elements in each line of the feature map. The width of f.
    @param f_height The number of rows in the feature map. The height of f.
    @param output   The stream into which the inputs will be stored.
*/
int extract_feature_map(char* filepath, int f_width, int f_height, float** *output) {
    const size_t buffer_size = (FLOAT_STRING_LENGTH * f_width) + 1; // +1 for null-byte

    if (filepath == NULL){ return 1; }
    char* buffer = (char*)malloc(buffer_size);

    FILE* file_ptr = fopen(filepath, "r");

    // Now loop over each line in the file
    int row_index = 0;
    while (fgets(buffer, buffer_size, file_ptr) != NULL){
        if (row_index == 0) {
            row_index++;
            continue;
        }
        char* token = strtok(buffer, " ");

        // Now loop over each number in the line
        int column_index = 0;
        while (token != NULL){
            float element = (float)atof(token);
            (*output)[row_index-1][column_index] = element; // Add to feature_map. Need to do "-1" to avoid the first line.
            token = strtok(NULL, " ");
            column_index++;
        }
        row_index++;
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

    // Extract dimensions of the feature map
    if (feature_file != NULL && (H <= 0 || W <= 0)){
        int status = extract_dimensions(feature_file, &H, &W);
        if (status != 0){ 
            // TODO: Handle when it can't extract file dimensions
        }
    }
    
    // TODO: Add check to see if the user wants to generate their own feature map.

    // Allocate memory for the feature map of the feature map.
    float** feature_map = (float**)malloc(H * sizeof(float*));
    for (int i = 0; i < H; i++){
        feature_map[i] = (float*)malloc(W * sizeof(float));
    }

    // Extract Feature Map
    if (feature_file != NULL){
        int status = extract_feature_map(feature_file, W, H, &feature_map);
        for(int i = 0; i < 5; i++){
            printf("%f\n", feature_map[0][i]);
        }
    }

    // TODO: Extract Kernel

    // TODO: conv2d()

    // TODO: (optionally) write to output file

    // ...

    // profit?
        
    
    return 0;
}


void extract_kernel(char* filepath, float** output){

}



// Function to perform 2D discrete convolution
void conv2d(float** f, float** g, float** output){
    
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

def conv(f,g):
    max_height = len(f) - 2
    max_width = len(f[0]) - 2
    output = [[0]*max_width for _ in range(max_height)]

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
