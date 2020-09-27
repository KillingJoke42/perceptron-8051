  #include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "nn.h"
#include "data_build.h"

int main(void)
{
	// Tunable hyperparameters
	int num_inputs = 256;
	int num_outputs = 10;
	float learning_rate = 0.01f;
	float annealing = 1.0f;
	int epochs = 200;
	// End of tunable parameters
	data_t *data;
	nn_t *nn;
	int i, j;

	// Set the random seed
	srand(time(0));
	// Load sample data into a data structure in memory
	data = load_data("train.csv", num_inputs, num_outputs);
	// Initialize a neural network model
	nn = nn_init();
	// Construct the neural network, layer by layer
	nn_add_layer(nn, num_inputs, ACTIVATION_FUNCTION_TYPE_NONE, 0);
	nn_add_layer(nn, 50, ACTIVATION_FUNCTION_TYPE_RELU, 0);
	nn_add_layer(nn, num_outputs, ACTIVATION_FUNCTION_TYPE_SIGMOID, 0);
	// It is critical to shuffle training data to properly train the model
	shuffle(data);
	printf("error\tlearning_rate\n");
	for (i = 0; i < epochs; i++) {
		float error = 0.0f;
		for (j = 0; j < data->num_rows; j++) {
			float *input = data->input[j];
			float *target = data->target[j];
			error += nn_train(nn, input, target, learning_rate);
		}
		printf("%.5f\t%.5f\n", error / data->num_rows, learning_rate);
		learning_rate *= annealing;
	}
	// Save the neural network architecture and weights to a file so that it can be used later
	nn_save(nn, "model.txt");
	data_free(data);
	nn_free(nn);
	return 0;
}