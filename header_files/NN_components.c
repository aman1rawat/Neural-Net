#include "NN_components.h"

#define sigmoid(x)  (1/(1+exp(-x)))

Matrix* activate(Layer *layer){
	if(strcmp(layer->activation, "linear")==0){
		return copyMatrix(layer->weighted_sum);
	}
	if(strcmp(layer->activation, "sigmoid")==0){
		Matrix *m = copyMatrix(layer->weighted_sum);
		for(int i=0;i<m->row;i++){
			for(int j=0;j<m->col;j++){
				double x = m->val[i][j];
				m->val[i][j] = sigmoid(m->val[i][j]);
			}
		}
		return m;
	}
	if(strcmp(layer->activation, "relu")==0){
		Matrix *m = copyMatrix(layer->weighted_sum);
		for(int i=0;i<m->row;i++){
			for(int j=0;j<m->col;j++){
				double x = m->val[i][j];
				m->val[i][j] = (m->val[i][j]<0)?0:m->val[i][j];
			}
		}
		return m;
	}
	else{
		exit(1);
	}
}

Matrix* activationDerivative(Layer *layer){
	if(strcmp(layer->activation, "linear")==0){
		Matrix *m = copyMatrix(layer->weighted_sum);
		fillMatrix(m, 1);
		return m;
	}
	if(strcmp(layer->activation, "sigmoid")==0){
		Matrix *m = copyMatrix(layer->weighted_sum);
		for(int i=0;i<m->row;i++){
			for(int j=0;j<m->col;j++){
				double x = m->val[i][j];
				m->val[i][j] = sigmoid(m->val[i][j])*(1- sigmoid(m->val[i][j]));
			}
		}
		return m;
	}
	if(strcmp(layer->activation, "relu")==0){
		Matrix *m = copyMatrix(layer->weighted_sum);
		for(int i=0;i<m->row;i++){
			for(int j=0;j<m->col;j++){
				double x = m->val[i][j];
				m->val[i][j] = (m->val[i][j]>0)?1:0;
			}
		}
		return m;
	}
	else{
		exit(1);
	}
}

double calculate_loss(NeuralNet *network, Matrix *output){
	if(strcmp(network->loss_function, "MSE")==0){
		Matrix *m = subtract(network->tail_layer->output, output);
		m = elementWiseMultiply(m,m);
		double calculated_loss = 0;
		for(int i=0;i<m->row;i++){
			for(int j=0;j<m->col;j++){
				calculated_loss += (m->val[i][j]); 
			}
		}
		calculated_loss/=(m->row*m->col);
		freeMatrix(m);
		return calculated_loss;
	}
	else{
		printf("ERROR\n");
		exit(1);
	}
}

Matrix* lossDerivative(NeuralNet *network, Matrix *output){
	if(strcmp(network->loss_function, "MSE")==0){
		Matrix *m = subtract(network->tail_layer->output, output);
		return m;
	}
	else{
		exit(1);
	}
}

void clipGradients(Matrix *gradient, double threshold){
    for(int i=0;i<gradient->row;i++){
        for(int j=0;j<gradient->col;j++){
            if(gradient->val[i][j]>threshold) gradient->val[i][j] = threshold;
            if(gradient->val[i][j]<(-threshold)) gradient->val[i][j] = -threshold;
        }
    }
}
