#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

// Define image and label sizes for MNIST

#define IMAGE_SIZE 784    // 28x28 pixels per image
#define TRAINING_SET_SIZE 60000
#define INPUT_SIZE 784
#define TESTING_SET_SIZE 10000
#define HIDDEN_1_SIZE 128
#define HIDDEN_2_SIZE 128
#define HIDDEN_3_SIZE 128
#define HIDDEN_LAYER_SIZE 128
#define OUTPUT_SIZE 10
#define EPOCHS 10
#define BATCH_SIZE 64


// Function to perform ReLU activation
float relu(float x) {
    return x > 0 ? x : 0;
}

// Function to compute softmax
void softmax(float* input, size_t length) {
    float sum = 0.0;
    for (size_t i = 0; i < length; i++) {
        input[i] = exp(input[i]);
        sum += input[i];
    }
    for (size_t i = 0; i < length; i++) {
        input[i] /= sum;
    }
}

// Function to load CSV file and store in arrays
void load_csv(const char* file_path, uint8_t* images, uint8_t* labels, size_t* num_samples) {
    FILE* csv_file = fopen(file_path, "r");
    if (csv_file == NULL) {
        printf("Error opening file %s!\n", file_path);
        exit(1);
    }

    size_t sample_idx = 0;
    char line[8192];  // Adjust line size based on expected row size

    // Read each line from the CSV
    while (fgets(line, sizeof(line), csv_file)) {
        char* token = strtok(line, ",");
        if (token != NULL) {
            labels[sample_idx] = (uint8_t) atoi(token); // Read the label

            // Read the pixels (784 pixels per image)
            for (size_t pixel_idx = 0; pixel_idx < IMAGE_SIZE; pixel_idx++) {
                token = strtok(NULL, ",");
                if (token != NULL) {
                    images[sample_idx * IMAGE_SIZE + pixel_idx] = (uint8_t)atoi(token);  // Read pixel values
                }
            }
            sample_idx++;
        }
    }

    *num_samples = sample_idx;  // Set the number of samples read
    fclose(csv_file);
}

// Function for forward propagation through the network
void forward_pass(uint8_t* input, float* W1, float* W2, float* W3, float* W4, size_t input_size, size_t hidden_size, size_t output_size, float* output) {
    float hidden1[HIDDEN_LAYER_SIZE];
    float hidden2[HIDDEN_LAYER_SIZE];
    float hidden3[HIDDEN_LAYER_SIZE];

    // First layer: input -> hidden layer 1
    for (size_t i = 0; i < hidden_size; i++) {
        hidden1[i] = 0;
        for (size_t j = 0; j < input_size; j++) {
            hidden1[i] += input[j] * W1[i * input_size + j];
        }
        hidden1[i] = relu(hidden1[i]);
    }

    // Second layer: hidden1 -> hidden layer 2
    for (size_t i = 0; i < hidden_size; i++) {
        hidden2[i] = 0;
        for (size_t j = 0; j < hidden_size; j++) {
            hidden2[i] += hidden1[j] * W2[i * hidden_size + j];
        }
        hidden2[i] = relu(hidden2[i]);
    }

    // Third layer: hidden2 -> hidden layer 3
    for (size_t i = 0; i < hidden_size; i++) {
        hidden3[i] = 0;
        for (size_t j = 0; j < hidden_size; j++) {
            hidden3[i] += hidden2[j] * W3[i * hidden_size + j];
        }
        hidden3[i] = relu(hidden3[i]);
    }

    // Output layer: hidden3 -> output layer
    for (size_t i = 0; i < output_size; i++) {
        output[i] = 0;
        for (size_t j = 0; j < hidden_size; j++) {
            output[i] += hidden3[j] * W4[i * hidden_size + j];
        }
    }

    // Apply softmax on output
    softmax(output, output_size);
}

// Function to load weights from text file
void load_weights(const char* filename, float* weights, size_t size) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening file %s!\n", filename);
        exit(1);
    }
    for (size_t i = 0; i < size; i++) {
        fscanf(file, "%f,", &weights[i]);
    }
    fclose(file);
}

// Function to compute accuracy
float compute_accuracy(uint8_t* test_images, uint8_t* test_labels, float* W1, float* W2, float* W3, float* W4, size_t num_samples, size_t input_size, size_t hidden_size, size_t output_size) {
    size_t correct_predictions = 0;
    float output[output_size];

    for (size_t i = 0; i < num_samples; i++) {
        forward_pass(&test_images[i * IMAGE_SIZE], W1, W2, W3, W4, input_size, hidden_size, output_size, output);

        // Get predicted class (index of max value in output)
        size_t predicted_class = 0;
        for (size_t j = 1; j < output_size; j++) {
            if (output[j] > output[predicted_class]) {
                predicted_class = j;
            }
        }

        if (predicted_class == test_labels[i]) {
            correct_predictions++;
        }
    }

    return (float)correct_predictions / num_samples * 100;
}

int main() {
    // Allocate memory for input, labels, and weights
    uint8_t* train_images = (uint8_t*)malloc(TRAINING_SET_SIZE * IMAGE_SIZE * sizeof(uint8_t));
    uint8_t* train_labels = (uint8_t*)malloc(TRAINING_SET_SIZE * sizeof(uint8_t));
    uint8_t* test_images = (uint8_t*)malloc(TESTING_SET_SIZE * IMAGE_SIZE * sizeof(uint8_t));
    uint8_t* test_labels = (uint8_t*)malloc(TESTING_SET_SIZE * sizeof(uint8_t));

    size_t num_train_samples = 0;
    size_t num_test_samples = 0;

    // Load the training and testing data from the CSV files
    load_csv("mnist_train.csv", train_images, train_labels, &num_train_samples);
    load_csv("mnist_test.csv", test_images, test_labels, &num_test_samples);

    // Allocate memory for weights
    float* W1 = (float*)malloc(IMAGE_SIZE * HIDDEN_LAYER_SIZE * sizeof(float));
    float* W2 = (float*)malloc(HIDDEN_LAYER_SIZE * HIDDEN_LAYER_SIZE * sizeof(float));
    float* W3 = (float*)malloc(HIDDEN_LAYER_SIZE * HIDDEN_LAYER_SIZE * sizeof(float));
    float* W4 = (float*)malloc(HIDDEN_LAYER_SIZE * OUTPUT_SIZE * sizeof(float));

    // Load pre-trained weights or initialize them
    load_weights("mnist_model_W1.txt", W1, INPUT_SIZE * HIDDEN_1_SIZE);
    load_weights("mnist_model_W2.txt", W2, HIDDEN_1_SIZE * HIDDEN_2_SIZE);
    load_weights("mnist_model_W3.txt", W3, HIDDEN_2_SIZE * HIDDEN_2_SIZE);
    load_weights("mnist_model_W4.txt", W4, HIDDEN_2_SIZE * OUTPUT_SIZE);

    // Compute the accuracy
    float accuracy = compute_accuracy(test_images, test_labels, W1, W2, W3, W4, num_test_samples, IMAGE_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_SIZE);
    printf("Test accuracy: %.2f%%\n", accuracy);

    // Free allocated memory
    free(train_images);
    free(train_labels);
    free(test_images);
    free(test_labels);
    free(W1);
    free(W2);
    free(W3);
    free(W4);

    return 0;
}
