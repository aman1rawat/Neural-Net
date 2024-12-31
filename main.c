#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#include "./header_files/matrix.h"
#include "./header_files/NN.h"
#include "./header_files/NN_components.h"

#define lr 0.001

//gcc main.c -Iheader_files header_files/NN.c header_files/NN_components.c header_files/matrix.c

int main(){
	Matrix **input = (Matrix**)malloc(8*sizeof(Matrix*));
	Matrix **output = (Matrix**)malloc(8*sizeof(Matrix*));
	for(int i=0;i<8;i++){
		input[i] = createMatrix(1,1);
		output[i] = createMatrix(1,1);
	}

	input[0]->val[0][0] = 1;
	input[1]->val[0][0] = 2;
	input[2]->val[0][0] = 3;
	input[3]->val[0][0] = 4;
	input[4]->val[0][0] = 5;
	input[5]->val[0][0] = 6;
	input[6]->val[0][0] = 7;
	input[7]->val[0][0] = 8;
	output[0]->val[0][0] = 1;
	output[1]->val[0][0] = 4;
	output[2]->val[0][0] = 9;
	output[3]->val[0][0] = 16;
	output[4]->val[0][0] = 25;
	output[5]->val[0][0] = 36;
	output[6]->val[0][0] = 49;
	output[7]->val[0][0] = 64;

	// for(int i=0;i<8;i++){
	// 	input[i]->val[0][0] = (input[i]->val[0][0]-1)/7;
	// 	output[i]->val[0][0] = (output[i]->val[0][0]-1)/63;
	// }

	NeuralNet *network = buildNetwork(1, "MSE");
	addLayer(network, 4, "sigmoid");
	addLayer(network, 8, "sigmoid");
	addLayer(network, 4, "sigmoid");
	addLayer(network, 1, "linear");

	for(int epoch=0;epoch<10000;epoch++){
		for(int sample=0;sample<8;sample++){
			trainNetwork(network, input[sample], output[sample], lr);
			printf("%.4f | %.4f | %.4f | %.4f\n", input[sample]->val[0][0], output[sample]->val[0][0], network->tail_layer->output->val[0][0], network->loss);
		}
		printf("---------------------------------------------------\n");																								
	}

	// for(int epoch=0;epoch<10000;epoch++){
	// 	printf("EPOCH : %d\n", epoch);
	// 	for(int sample=0;sample<1;sample++){
	// 		trainNetwork(network, input[sample], output[sample], lr);
	// 		for(Layer *layer=network->layers; layer; layer=layer->next_layer){
	// 			printLayer(layer);
	// 		}
	// 	}
	// }

	return 0;
}