#ifndef NN_H
#define NN_H

typedef enum activation_function_type {
	ACTIVATION_FUNCTION_TYPE_NONE = 0,
	ACTIVATION_FUNCTION_TYPE_LINEAR,
	ACTIVATION_FUNCTION_TYPE_RELU,
	ACTIVATION_FUNCTION_TYPE_LEAKY_RELU,
	ACTIVATION_FUNCTION_TYPE_SIGMOID,
	ACTIVATION_FUNCTION_TYPE_TANH,
} activation_function_type_t;

typedef struct nn {
	int num_layers;		// Number of layers
	int *widths;		// Number of neurons by layer
	int *activations;	// Activation function used for each layer
	float *biases;		// Biases by layer
	float **neurons;	// Neurons by layer
	float ***weights;	// Weights by layer
} nn_t;

nn_t *nn_init(void);
void nn_free(nn_t *nn);
int nn_add_layer(nn_t *nn, int width, int activation, float bias);
int nn_save(nn_t *nn, char *path);
nn_t *nn_load(char *path);
float nn_train(nn_t *nn, float *inputs, float *targets, float rate);
float *nn_predict(nn_t *nn, float *inputs);

#endif /* NN_H */