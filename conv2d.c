#include <stdio.h>
#include <stdlib.h>
#include <string.h>


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
    for (int i = 1; i < argc; i++){

        if (i + 1 >= argc){ break; }

        // Check all flags
        if (strcmp(argv[i], "-H") == 0) {
            H = atoi(argv[++i]);
            i++;
            continue;
        }
        if (strcmp(argv[i], "-W") == 0) {
            W = atoi(argv[++i]);
            i++;
            continue;
        }
        if (strcmp(argv[i], "-kH") == 0) {
            kH = atoi(argv[++i]);
            i++;
            continue;
        }
        if (strcmp(argv[i], "-kW") == 0) {
            kW = atoi(argv[++i]);
            i++;
            continue;
        }
        if (strcmp(argv[i], "-f") == 0) {
            feature_file = argv[++i];
            i++;
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

        /* 
        Error catching for incorrect flag usage
        
            Examples: 
            1. No flags provided, 
            2. Generating array but provided only height not width, 
            3. Not generating and provided feature but no kernel
            4. Provided output, but generated/provided no input

        */


        // (optionally) generate inputs

        // Extract Feature Map

        // Extract Kernel

        // conv2d()

        // (optionally) write to output file

        // ...

        // profit?

    }
    return 0;
}

void extract_feature_map(char* filepath, float** output) {

}

void extract_kernel(char* filepath, float** output){

}



// Function to perform 2D discrete convolution
void conv2d(float**f, float** g, float** output){
    
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
