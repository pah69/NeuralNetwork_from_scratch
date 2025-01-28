#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define INPUT_SIZE 784
#define HIDDEN_1_SIZE 128
#define HIDDEN_2_SIZE 128
#define HIDDEN_3_SIZE 128
#define OUTPUT_SIZE 10
#define IMAGE_SIZE 28 * 28
#define LEARNING_RATE 0.01
#define EPOCHS 1
#define BATCH_SIZE 32

// Activation functions
float relu(float x) { return x > 0 ? x : 0; }
float relu_derivative(float x) { return x > 0 ? 1 : 0; }

void softmax(float *input, size_t length) {
    float max_val = input[0], sum = 0.0;
    for (size_t i = 1; i < length; i++)
        if (input[i] > max_val) max_val = input[i];
    for (size_t i = 0; i < length; i++) {
        input[i] = exp(input[i] - max_val);
        sum += input[i];
    }
    for (size_t i = 0; i < length; i++) input[i] /= sum;
}

// Function to load weights from text files
void load_weights(const char *filename, float *weights, size_t size) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error: Unable to open file %s\n", filename);
        exit(EXIT_FAILURE);
    }
    for (size_t i = 0; i < size; i++) {
        fscanf(file, "%f,", &weights[i]);
    }
    fclose(file);
}

// Function to load MNIST data (images and labels)
void load_mnist_data(const char *image_file, const char *label_file,
                     float *data, unsigned char *labels, size_t num_samples) {
    FILE *image_fp = fopen(image_file, "rb");
    FILE *label_fp = fopen(label_file, "rb");
    if (!image_fp || !label_fp) {
        printf("Error: Unable to open MNIST files\n");
        exit(EXIT_FAILURE);
    }

    // Skip the headers
    fseek(image_fp, 16, SEEK_SET);
    fseek(label_fp, 8, SEEK_SET);

    for (size_t i = 0; i < num_samples; i++) {
        for (size_t j = 0; j < IMAGE_SIZE; j++) {
            unsigned char pixel;
            fread(&pixel, sizeof(unsigned char), 1, image_fp);
            data[i * IMAGE_SIZE + j] = pixel / 255.0; // Normalize to [0, 1]
        }
        fread(&labels[i], sizeof(unsigned char), 1, label_fp);
    }

    fclose(image_fp);
    fclose(label_fp);
}

// Forward propagation
void forward(float *input, float *weights, size_t input_size, size_t output_size, float *output, int apply_relu) {
    for (size_t i = 0; i < output_size; i++) {
        output[i] = 0;
        for (size_t j = 0; j < input_size; j++) {
            output[i] += input[j] * weights[i * input_size + j];
        }
        if (apply_relu) output[i] = relu(output[i]);
    }
}

// Backward propagation
void backward(float *input, float *d_output, size_t input_size, size_t output_size, float *weights, float *grad_weights) {
    for (size_t i = 0; i < output_size; i++) {
        for (size_t j = 0; j < input_size; j++) {
            grad_weights[i * input_size + j] += d_output[i] * input[j];
        }
    }
}

// Compute accuracy
float compute_accuracy(float *data, unsigned char *labels, float *W1, float *W2, float *W3, float *W4, int num_samples) {
    int correct = 0;
    float hidden1[HIDDEN_1_SIZE];
    float hidden2[HIDDEN_2_SIZE];
    float hidden3[HIDDEN_2_SIZE];
    float output[OUTPUT_SIZE];

    for (int i = 0; i < num_samples; i++) {
        forward(data + i * IMAGE_SIZE, W1, INPUT_SIZE, HIDDEN_1_SIZE, hidden1, 1);
        forward(hidden1, W2, HIDDEN_1_SIZE, HIDDEN_2_SIZE, hidden2, 1);
        forward(hidden2, W3, HIDDEN_2_SIZE, HIDDEN_2_SIZE, hidden3, 1);
        forward(hidden3, W4, HIDDEN_2_SIZE, OUTPUT_SIZE, output, 0);
        softmax(output, OUTPUT_SIZE);

        // Get predicted label
        int predicted_label = 0;
        for (int j = 1; j < OUTPUT_SIZE; j++) {
            if (output[j] > output[predicted_label]) predicted_label = j;
        }

        if (predicted_label == labels[i]) correct++;
    }

    return (float)correct / num_samples * 100.0;
}
// Modified inference function for 3 hidden layers
void inference(float* input, float* W1, float* W2, float* W3, float* W4,
               float* hidden1, float* hidden2, float* hidden3, float* output) {
    size_t input_size = INPUT_SIZE;
    size_t hidden_1_size = HIDDEN_1_SIZE;
    size_t hidden_2_size = HIDDEN_2_SIZE;
    size_t hidden_3_size = HIDDEN_3_SIZE;
    size_t output_size = OUTPUT_SIZE;

    // Forward pass through Layer 1
    for (size_t i = 0; i < hidden_1_size; i++) {
        hidden1[i] = 0.0f;
        for (size_t j = 0; j < input_size; j++) {
            hidden1[i] += input[j] * W1[i * input_size + j];
        }
        hidden1[i] = relu(hidden1[i]);
    }

    // Forward pass through Layer 2
    for (size_t i = 0; i < hidden_2_size; i++) {
        hidden2[i] = 0.0f;
        for (size_t j = 0; j < hidden_1_size; j++) {
            hidden2[i] += hidden1[j] * W2[i * hidden_1_size + j];
        }
        hidden2[i] = relu(hidden2[i]);
    }

    // Forward pass through Layer 3
    for (size_t i = 0; i < hidden_3_size; i++) {
        hidden3[i] = 0.0f;
        for (size_t j = 0; j < hidden_2_size; j++) {
            hidden3[i] += hidden2[j] * W3[i * hidden_2_size + j];
        }
        hidden3[i] = relu(hidden3[i]);
    }

    // Forward pass through Output Layer
    for (size_t i = 0; i < output_size; i++) {
        output[i] = 0.0f;
        for (size_t j = 0; j < hidden_3_size; j++) {
            output[i] += hidden3[j] * W4[i * hidden_3_size + j];
        }
    }

    // Apply softmax to output layer
    softmax(output, output_size);
}

// Train function
void train(float* train_data, unsigned char* train_labels, size_t num_samples,
           float* W1, float* W2, float* W3, float* W4, size_t epochs, size_t batch_size) {
    size_t input_size = INPUT_SIZE;
    size_t hidden_1_size = HIDDEN_1_SIZE;
    size_t hidden_2_size = HIDDEN_2_SIZE;
    size_t hidden_3_size = HIDDEN_3_SIZE;
    size_t output_size = OUTPUT_SIZE;

    // Allocate memory for layer activations and gradients
    float* hidden1 = malloc(hidden_1_size * sizeof(float));
    float* hidden2 = malloc(hidden_2_size * sizeof(float));
    float* hidden3 = malloc(hidden_3_size * sizeof(float));
    float* output = malloc(output_size * sizeof(float));
    float* output_gradient = malloc(output_size * sizeof(float));
    float* hidden3_gradient = malloc(hidden_3_size * sizeof(float));
    float* hidden2_gradient = malloc(hidden_2_size * sizeof(float));
    float* hidden1_gradient = malloc(hidden_1_size * sizeof(float));

    for (size_t epoch = 0; epoch < epochs; epoch++) {
        size_t correct = 0;

        for (size_t batch_start = 0; batch_start < num_samples; batch_start += batch_size) {
            size_t batch_end = fmin(batch_start + batch_size, num_samples);

            // Batch gradient accumulators
            float* W1_grad = calloc(input_size * hidden_1_size, sizeof(float));
            float* W2_grad = calloc(hidden_1_size * hidden_2_size, sizeof(float));
            float* W3_grad = calloc(hidden_2_size * hidden_3_size, sizeof(float));
            float* W4_grad = calloc(hidden_3_size * output_size, sizeof(float));

            for (size_t idx = batch_start; idx < batch_end; idx++) {
                // Forward pass
                inference(&train_data[idx * input_size], W1, W2, W3, W4, hidden1, hidden2, hidden3, output);

                // Compute loss and gradients for the output layer
                unsigned char true_label = train_labels[idx];
                for (size_t i = 0; i < output_size; i++) {
                    output_gradient[i] = (i == true_label ? 1.0f : 0.0f) - output[i];
                }

                // Backpropagation through Output Layer
                for (size_t i = 0; i < hidden_3_size; i++) {
                    hidden3_gradient[i] = 0.0f;
                    for (size_t j = 0; j < output_size; j++) {
                        hidden3_gradient[i] += output_gradient[j] * W4[j * hidden_3_size + i];
                        W4_grad[j * hidden_3_size + i] += output_gradient[j] * hidden3[i];
                    }
                }

                // Backpropagation through Hidden Layer 3
                for (size_t i = 0; i < hidden_2_size; i++) {
                    hidden2_gradient[i] = 0.0f;
                    for (size_t j = 0; j < hidden_3_size; j++) {
                        hidden2_gradient[i] += hidden3_gradient[j] * W3[j * hidden_2_size + i];
                        W3_grad[j * hidden_2_size + i] += hidden3_gradient[j] * hidden2[i];
                    }
                }

                // Backpropagation through Hidden Layer 2
                for (size_t i = 0; i < hidden_1_size; i++) {
                    hidden1_gradient[i] = 0.0f;
                    for (size_t j = 0; j < hidden_2_size; j++) {
                        hidden1_gradient[i] += hidden2_gradient[j] * W2[j * hidden_1_size + i];
                        W2_grad[j * hidden_1_size + i] += hidden2_gradient[j] * hidden1[i];
                    }
                }

                // Backpropagation through Hidden Layer 1
                for (size_t i = 0; i < input_size; i++) {
                    for (size_t j = 0; j < hidden_1_size; j++) {
                        W1_grad[j * input_size + i] += hidden1_gradient[j] * train_data[idx * input_size + i];
                    }
                }

                // Calculate accuracy
                size_t prediction = 0;
                for (size_t i = 1; i < output_size; i++) {
                    if (output[i] > output[prediction]) {
                        prediction = i;
                    }
                }
                if (prediction == true_label) {
                    correct++;
                }
            }

            // Update weights
            for (size_t i = 0; i < input_size * hidden_1_size; i++) {
                W1[i] += LEARNING_RATE * W1_grad[i];
            }
            for (size_t i = 0; i < hidden_1_size * hidden_2_size; i++) {
                W2[i] += LEARNING_RATE * W2_grad[i];
            }
            for (size_t i = 0; i < hidden_2_size * hidden_3_size; i++) {
                W3[i] += LEARNING_RATE * W3_grad[i];
            }
            for (size_t i = 0; i < hidden_3_size * output_size; i++) {
                W4[i] += LEARNING_RATE * W4_grad[i];
            }

            free(W1_grad);
            free(W2_grad);
            free(W3_grad);
            free(W4_grad);
        }

        // Print epoch accuracy
        printf("Epoch %zu Accuracy: %.2f%%\n", epoch + 1, 100.0 * correct / num_samples);
    }

    // Free allocated memory
    free(hidden1);
    free(hidden2);
    free(hidden3);
    free(output);
    free(output_gradient);
    free(hidden3_gradient);
    free(hidden2_gradient);
    free(hidden1_gradient);
}


int main() {
    size_t num_train_samples = 60000;
    size_t num_test_samples = 10000;

    // Allocate memory for train and test datasets
    float *train_data = malloc(num_train_samples * IMAGE_SIZE * sizeof(float));
    unsigned char *train_labels = malloc(num_train_samples * sizeof(unsigned char));
    float *test_data = malloc(num_test_samples * IMAGE_SIZE * sizeof(float));
    unsigned char *test_labels = malloc(num_test_samples * sizeof(unsigned char));

    // Load train and test MNIST data
    load_mnist_data("train-images.idx3-ubyte", "train-labels.idx1-ubyte", train_data, train_labels, num_train_samples);
    load_mnist_data("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", test_data, test_labels, num_test_samples);

    // Initialize weights
    float *W1 = malloc(INPUT_SIZE * HIDDEN_1_SIZE * sizeof(float));
    float *W2 = malloc(HIDDEN_1_SIZE * HIDDEN_2_SIZE * sizeof(float));
    float *W3 = malloc(HIDDEN_2_SIZE * HIDDEN_2_SIZE * sizeof(float));
    float *W4 = malloc(HIDDEN_2_SIZE * OUTPUT_SIZE * sizeof(float));

    // Load pre-trained weights or initialize them
    load_weights("mnist_model_W1.txt", W1, INPUT_SIZE * HIDDEN_1_SIZE);
    load_weights("mnist_model_W2.txt", W2, HIDDEN_1_SIZE * HIDDEN_2_SIZE);
    load_weights("mnist_model_W3.txt", W3, HIDDEN_2_SIZE * HIDDEN_2_SIZE);
    load_weights("mnist_model_W4.txt", W4, HIDDEN_2_SIZE * OUTPUT_SIZE);

    // Train the model
    printf("Training...\n");
    train(train_data, train_labels, num_train_samples, W1, W2, W3, W4, EPOCHS, BATCH_SIZE);

    // Evaluate on test dataset
    printf("Evaluating on test data...\n");
    float test_accuracy = compute_accuracy(test_data, test_labels, W1, W2, W3, W4, num_test_samples);
    printf("Test Accuracy: %.2f%%\n", test_accuracy);

    // Free allocated memory
    free(train_data);
    free(train_labels);
    free(test_data);
    free(test_labels);
    free(W1);
    free(W2);
    free(W3);
    free(W4);

    return 0;
}