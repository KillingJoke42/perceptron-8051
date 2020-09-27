#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "nn.h"

typedef float (*activation_function_ptr_t)(float a, bool derivative);

// Null activation function
static float activation_function_none(float a, bool derivative)
{
	return 0;
}
static float activation_function_linear(float a, bool derivative)
{
	if (derivative)
		return 1;
	return a;
}
// Rectified Linear Unit (ReLU) activation function
static float activation_function_relu(float a, bool derivative)
{
	if (a >= 0)
		return (derivative ? 1 : a);
	return 0;
}

// Leaky Rectified Linear Unit (Leaky ReLU) activation function
static float activation_function_leaky_relu(float a, bool derivative)
{
	if (a > 0)
		return (derivative ? 1 : a);
	return (derivative ? 0.01 : a * 0.01f);
}
// Sigmoid activation function (aka Logistic, aka Soft Step)
static float activation_function_sigmoid(float a, bool derivative)
{
	if (derivative) {
		float f = activation_function_sigmoid(a, false);
		return(f * (1.0f - f));
	}
	return 1.0f / (1.0f + expf(-a));
}
// Tanh activation function
static float activation_function_tanh(float a, bool derivative)
{
	if (derivative)
		return 1.0 - activation_function_tanh(a, false) * activation_function_tanh(a, false);
	return (2.0 / (1.0 + expf(-2.0 * a))) - 1.0;
}
static activation_function_ptr_t activation_functions[] = {
	activation_function_none,
	activation_function_linear,
	activation_function_relu,
	activation_function_leaky_relu,
	activation_function_sigmoid,
	activation_function_tanh,
};

// Returns floating point random number between 0 and 1
static float frand(void)
{
	return rand() / (float)RAND_MAX;
}

// Computes the error given a cost function
static float error(float a, float b)
{
	return 0.5f * (a - b) * (a - b);
}

// Computes derivative of the error through the derivative of the cost function
static float error_derivative(float a, float b)
{
	return a - b;
}

static void forward_propagation(nn_t *nn)
{
	float sum;
	int i, j, layer;

	// Calculate neuron values in each layer
	for (layer = 1; layer < nn->num_layers; layer++) {
		for (i = 0; i < nn->widths[layer]; i++) {
			sum = 0;
			for (j = 0; j < nn->widths[layer - 1]; j++)
				sum += nn->neurons[layer - 1][j] * nn->weights[layer][i][j];
			nn->neurons[layer][i] = activation_functions[nn->activations[layer]](sum + nn->biases[layer], false);
		}
	}
}

nn_t *nn_init(void)
{
	nn_t *nn;

	nn = (nn_t *)malloc(sizeof(nn_t));
	if (NULL == nn)
		return NULL;
	nn->num_layers = 0;
	nn->widths = NULL;
	nn->weights = NULL;
	nn->neurons = NULL;
	nn->biases = NULL;
	nn->activations = NULL;
	return nn;
}

void nn_free(nn_t *nn)
{
	int layer, i;

	// There are no weights associated with the input layer, so we skip layer 0 and start at layer 1.
	for (layer = 1; layer < nn->num_layers; layer++) {
		for (i = 0; i < nn->widths[layer]; i++)
			free(nn->weights[layer][i]);
		free(nn->weights[layer]);
	}
	// There are no neurons in the input layer, as the input array itself is used to store these values.
	for (layer = 1; layer < nn->num_layers; layer++)
		free(nn->neurons[layer]);
	free(nn->weights);
	free(nn->neurons);
	free(nn->biases);
	free(nn->activations);
	free(nn->widths);
	free(nn);
}

int nn_add_layer(nn_t *nn, int width, int activation, float bias)
{
	nn->num_layers++;
	nn->widths = (int *)realloc(nn->widths, nn->num_layers * sizeof(*nn->widths));
	if (NULL == nn->widths)
		return 1;
	nn->widths[nn->num_layers - 1] = width;
	nn->activations = (int *)realloc(nn->activations, nn->num_layers * sizeof(*nn->activations));
	if (NULL == nn->activations)
		return 1;
	nn->activations[nn->num_layers -1 ] = activation;
	nn->biases = (float *)realloc(nn->biases, nn->num_layers * sizeof(*nn->biases));
	if (NULL == nn->biases)
		return 1;
	nn->biases[nn->num_layers - 1] = bias;
	nn->neurons=(float **)realloc(nn->neurons, nn->num_layers * sizeof(float *));
	if (NULL == nn->neurons)
		return 1;
	if (nn->num_layers > 1) {
		nn->neurons[nn->num_layers - 1] = (float *)malloc(nn->widths[nn->num_layers - 1] * sizeof(float));
		if (NULL == nn->neurons[nn->num_layers - 1])
			return 1;
	}
	nn->weights = (float ***)realloc(nn->weights, (nn->num_layers) * sizeof(float **));
	if (NULL == nn->weights)
		return 1;
	if (nn->num_layers > 1) {
		nn->weights[nn->num_layers - 1] = (float **)malloc((nn->widths[nn->num_layers - 1]) * sizeof(float *));
		if (NULL == nn->weights[nn->num_layers - 1])
			return 1;
		for (int neuron = 0; neuron < nn->widths[nn->num_layers - 1]; neuron++) {
			nn->weights[nn->num_layers - 1][neuron] = (float *)malloc((nn->widths[nn->num_layers - 2]) * sizeof(float));
			if (NULL == nn->weights[nn->num_layers - 1][neuron])
				return 1;
			// Randomize the weights in this layer
			for (int i = 0; i < nn->widths[nn->num_layers - 2]; i++)
				nn->weights[nn->num_layers - 1][neuron][i] = frand() - 0.5f;
		}
	}
	return 0;
}

// Saves a neural net model file
int nn_save(nn_t *nn, char *path)
{
	int layer, i, j;
	FILE *file;

	file = fopen(path, "w");
	if (NULL == file)
		return 1;
	fprintf(file, "%d\n", nn->num_layers);
	for (i = 0; i < nn->num_layers; i++)
		fprintf(file, "%d %d %f\n", nn->widths[i], nn->activations[i], nn->biases[i]);
	for (layer = 1; layer < nn->num_layers; layer++)
		for (i = 0; i < nn->widths[layer]; i++)
			for (j = 0; j < nn->widths[layer-1]; j++)
				fprintf(file, "%f\n", nn->weights[layer][i][j]);
	fclose(file);
	return 0;
}

// Loads a neural net model file
nn_t *nn_load(char *path)
{
	FILE *file;
	nn_t *nn;
	int width = 0;
	int activation = ACTIVATION_FUNCTION_TYPE_NONE;
	float bias = 0;
	int layer, i, j;
	int num_layers;

	file = fopen(path, "r");
	if (NULL == file)
		return NULL;
	nn = nn_init();
	fscanf(file, "%d\n", &num_layers);
	for (i = 0; i < num_layers; i++) {
		fscanf(file, "%d %d %f\n", &width, &activation, &bias);
		if (nn_add_layer(nn, width, activation, bias) != 0) {
			fclose(file);
			return NULL;
		}
	}
	// Read in the weights
	for (layer = 1; layer < nn->num_layers; layer++)
		for (i = 0; i < nn->widths[layer]; i++)
			for (j = 0; j < nn->widths[layer - 1]; j++)
				fscanf(file, "%f\n", &nn->weights[layer][i][j]);
	fclose(file);
	return nn;
}

// Trains a nn with a given input and target output at a specified learning rate
// Returns the total error between the target and the output of the neural network
float nn_train(nn_t *nn, float *inputs, float *targets, float rate)
{
	float de, da, sum;
	int i, j, layer;

	nn->neurons[0] = inputs;
	forward_propagation(nn);
	// Perform back propagation
	for (i = 0; i < nn->widths[nn->num_layers - 2]; i++) {
		sum=0;
		// Backpropagation Reference: Deep Learning Vol. 1, From Basics to Practice
		// Calculate total error at the output layer
		for (j = 0; j < nn->widths[nn->num_layers - 1]; j++) {
			// Derivative of the error
			de = error_derivative(nn->neurons[nn->num_layers - 1][j], targets[j]);
			// Derivative of the activation function
			da = activation_functions[nn->activations[nn->num_layers - 1]](nn->neurons[nn->num_layers - 1][j], true);
			sum += de * da * nn->weights[nn->num_layers - 1][j][i];
			// Correct the weights between this layer and the next layer
			nn->weights[nn->num_layers - 1][j][i] -= rate * de * da * nn->neurons[nn->num_layers - 2][i];
		}
		// Correct weights between previous layer and this one
		for (layer = nn->num_layers - 2; layer > 0; layer--)
			for (j = 0; j < nn->widths[layer - 1]; j++)
				nn->weights[layer][i][j] -= rate * (sum + nn->biases[layer]) * activation_functions[nn->activations[layer]](nn->neurons[layer][i], true) * nn->neurons[layer - 1][j];
	}
	// Calculate total error
	sum = 0;
	for (i = 0; i < nn->widths[nn->num_layers - 1]; i++)
		sum += error(targets[i], nn->neurons[nn->num_layers - 1][i]);
	return sum;
}

// Returns an output prediction given an input
float *nn_predict(nn_t *nn, float *inputs)
{
	nn->neurons[0] = inputs;
	forward_propagation(nn);
	// Return a pointer to the output layer
	return nn->neurons[nn->num_layers - 1];
}