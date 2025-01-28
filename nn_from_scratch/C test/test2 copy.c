#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#define IMAGE_ROWS 28
#define IMAGE_COLS 28
#define IMAGE_SIZE (IMAGE_ROWS * IMAGE_COLS)
#define NUM_CLASSES 10

// ReLU activation function
float relu(float x) {
    return x > 0 ? x : 0;
}

// Softmax function
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

// Load MNIST images
int load_mnist_images(const char* filename, float** images, size_t* num_images) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Failed to open image file: %s\n", filename);
        return -1;
    }

    uint32_t magic_number, count, rows, cols;
    fread(&magic_number, sizeof(uint32_t), 1, file);
    fread(&count, sizeof(uint32_t), 1, file);
    fread(&rows, sizeof(uint32_t), 1, file);
    fread(&cols, sizeof(uint32_t), 1, file);

    magic_number = __builtin_bswap32(magic_number);  // Convert to little-endian
    count = __builtin_bswap32(count);

    *num_images = count;
    *images = malloc(count * IMAGE_SIZE * sizeof(float));

    uint8_t* buffer = malloc(IMAGE_SIZE * sizeof(uint8_t));
    for (size_t i = 0; i < count; i++) {
        fread(buffer, sizeof(uint8_t), IMAGE_SIZE, file);
        for (size_t j = 0; j < IMAGE_SIZE; j++) {
            (*images)[i * IMAGE_SIZE + j] = buffer[j] / 255.0f;  // Normalize
        }
    }

    free(buffer);
    fclose(file);
    return 0;
}

// Load MNIST labels
int load_mnist_labels(const char* filename, uint8_t** labels, size_t* num_labels) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Failed to open label file: %s\n", filename);
        return -1;
    }

    uint32_t magic_number, count;
    fread(&magic_number, sizeof(uint32_t), 1, file);
    fread(&count, sizeof(uint32_t), 1, file);

    magic_number = __builtin_bswap32(magic_number);  // Convert to little-endian
    count = __builtin_bswap32(count);

    *num_labels = count;
    *labels = malloc(count * sizeof(uint8_t));

    fread(*labels, sizeof(uint8_t), count, file);
    fclose(file);
    return 0;
}

// Forward propagation
void forward_propagation(float* input, float* weights, size_t input_size, size_t output_size, float* output) {
    for (size_t i = 0; i < output_size; i++) {
        output[i] = 0;
        for (size_t j = 0; j < input_size; j++) {
            output[i] += input[j] * weights[i * input_size + j];
        }
        output[i] = relu(output[i]);  // ReLU for hidden layers
    }
}

// Main function to test the model
int main() {
    size_t input_size = IMAGE_SIZE;
    size_t hidden_1_size = 128;
    size_t hidden_2_size = 128;
    size_t hidden_3_size = 128;
    size_t output_size = NUM_CLASSES;

    float* W1 = malloc(input_size * hidden_1_size * sizeof(float));
    float* W2 = malloc(hidden_1_size * hidden_2_size * sizeof(float));
    float* W3 = malloc(hidden_2_size * hidden_3_size * sizeof(float));
    float* W4 = malloc(hidden_3_size * output_size * sizeof(float));

    load_weights("weights_W1.txt", W1, input_size * hidden_1_size);
    load_weights("weights_W2.txt", W2, hidden_1_size * hidden_2_size);
    load_weights("weights_W3.txt", W3, hidden_2_size * hidden_3_size);
    load_weights("weights_W4.txt", W4, hidden_3_size * output_size);

    float *images;
    uint8_t *labels;
    size_t num_images, num_labels;

    load_mnist_images("t10k-images.idx3-ubyte", &images, &num_images);
    load_mnist_labels("t10k-labels.idx1-ubyte", &labels, &num_labels);

    size_t correct_predictions = 0;

    for (size_t i = 0; i < num_images; i++) {
        float hidden1[128], hidden2[128], hidden3[128], output[10];

        forward_propagation(&images[i * IMAGE_SIZE], W1, input_size, hidden_1_size, hidden1);
        forward_propagation(hidden1, W2, hidden_1_size, hidden_2_size, hidden2);
        forward_propagation(hidden2, W3, hidden_2_size, hidden_3_size, hidden3);
        forward_propagation(hidden3, W4, hidden_3_size, output_size, output);

        // Softmax for the output layer
        softmax(output, output_size);

        // Find the predicted label
        size_t predicted_label = 0;
        for (size_t j = 1; j < output_size; j++) {
            if (output[j] > output[predicted_label]) {
                predicted_label = j;
            }
        }

        if (predicted_label == labels[i]) {
            correct_predictions++;
        }
    }

    printf("Accuracy: %.2f%%\n", 100.0 * correct_predictions / num_images);

    free(W1);
    free(W2);
    free(W3);
    free(W4);
    free(images);
    free(labels);

    return 0;
}
