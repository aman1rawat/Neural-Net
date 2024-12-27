#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#include "./header_files/matrix.h"
#include "./header_files/NN.h"
#include "./header_files/NN_components.h"

//gcc main.c -Iheader_files header_files/NN.c header_files/NN_components.c header_files/matrix.c

int main(){
	Matrix **input = (Matrix**)malloc(8*sizeof(Matrix*));
	Matrix **output = (Matrix**)malloc(8*sizeof(Matrix*));
	for(int i=0;i<8;i++){
		for(int j=0;j<8;j++){
			input[i] = createMatrix(1,1);
			output[i] = createMatrix(1,1);
		}
	}

	NeuralNet *network = buildNetwork(1, "MSE");
	
	
	return 0;
}