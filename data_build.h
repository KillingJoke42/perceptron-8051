typedef struct {
	int num_rows;		// Number of rows of data
	int num_inputs;		// Number of inputs to neural network
	int num_outputs;	// Number of outputs from neural network
	float **input;		// 2D array of inputs
	float **target;		// 2D array of targets
} data_t;

data_t *data_init(int num_rows, int num_inputs, int num_outputs);
void parse(data_t *data, char *line, int row);
int num_lines(FILE *file);
data_t *load_data(char *path, int num_inputs, int num_outputs);
void data_free(data_t *data);
void shuffle(data_t *data);
