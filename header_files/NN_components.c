#include "NN_components.h"

#define sigmoid(x)  (1/(1+exp(-x)))

Matrix* activate(Layer *layer){
	if(1){
		Matrix *m = copyMatrix(layer->weighted_sum);
		for(int i=0;i<m->row;i++){
			for(int j=0;j<m->col;j++){
				double x = m->val[i][j];
				m->val[i][j] = sigmoid(m->val[i][j]);
			}
		}
		return m;
	}
	else{
		exit(1);
	}
}

Matrix* activationDerivative(Layer *layer){
	if(!strcmp(layer->activation, "sigmoid")){
		Matrix *m = copyMatrix(layer->weighted_sum);
		for(int i=0;i<m->row;i++){
			for(int j=0;j<m->col;j++){
				double x = m->val[i][j];
				m->val[i][j] = sigmoid(m->val[i][j])*(1- sigmoid(m->val[i][j]));
			}
		}
		return m;
	}
	else{
		exit(1);
	}
}

Matrix* loss(NeuralNet *network, Matrix *output){
	if(!strcmp(network->loss_function, "MSE")){
		Matrix *m = subtract(network->tail_layer->output, output);
		m = elementWiseMultiply(m,m);
		return m;
	}
	else{
		exit(1);
	}
}

Matrix* lossDerivative(NeuralNet *network, Matrix *output){
	if(!strcmp(network->loss_function, "MSE")){
		Matrix *m = scale(subtract(output, network->tail_layer->output), 2);
		return m;
	}
	else{
		exit(1);
	}
}
