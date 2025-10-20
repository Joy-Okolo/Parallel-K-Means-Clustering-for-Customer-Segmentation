// Including necessary header files for standard input/output, memory allocation, math functions, etc.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <string.h>

// Defining a constant for the maximum number of iterations for the k-means algorithm
#define MAX_ITER 100

// Defining a structure to represent a data point with two features: annual income and spending score
typedef struct {
    float annual_income, spending_score;
} Point;

// Function to load data points from a CSV file
void load_data(Point **points, int *n, const char *file_path) {
    // Open the specified file in read mode
    FILE *file = fopen(file_path, "r");
    if (!file) {
        // Print an error message and exit if the file cannot be opened
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    // Declare a buffer to store lines read from the file
    char line[256];
    fgets(line, sizeof(line), file); // Skip the header line of the CSV file

    *n = 0; // Initialize the number of points to 0
    while (fgets(line, sizeof(line), file)) { // Count the number of data points
        (*n)++;
    }

    // Rewind the file pointer to the beginning of the file
    rewind(file);
    fgets(line, sizeof(line), file); // Skip the header line again

    // Allocate memory for the array of points
    *points = (Point *)malloc((*n) * sizeof(Point));
    int index = 0; // Initialize the index for the points array

    // Read the data points from the file and store them in the array
    while (fgets(line, sizeof(line), file)) {
        sscanf(line, "%*d,%*[^,],%*d,%f,%f", &(*points)[index].annual_income, &(*points)[index].spending_score);
        index++;
    }

    // Close the file after reading the data
    fclose(file);
}

// Function to calculate the Euclidean distance between two points
float euclidean_distance(Point p1, Point p2) {
    return sqrt(pow(p1.annual_income - p2.annual_income, 2) +
                pow(p1.spending_score - p2.spending_score, 2));
}

// K-Means clustering algorithm implementation
void kmeans(Point *points, int n, int k, int *clusters, Point *centroids) {
    // Iterate through a fixed number of iterations or until convergence
    for (int iter = 0; iter < MAX_ITER; iter++) {
        // Assign each point to the nearest cluster
        for (int i = 0; i < n; i++) {
            float min_dist = FLT_MAX; // Initialize minimum distance as the maximum float value
            int cluster = 0; // Initialize the cluster index
            for (int j = 0; j < k; j++) {
                // Calculate the distance from the point to the current centroid
                float dist = euclidean_distance(points[i], centroids[j]);
                if (dist < min_dist) {
                    // Update the nearest cluster and distance if the current centroid is closer
                    min_dist = dist;
                    cluster = j;
                }
            }
            // Assign the point to the nearest cluster
            clusters[i] = cluster;
        }

        // Allocate memory for counting points in each cluster and summing their coordinates
        int *count = calloc(k, sizeof(int));
        Point *sum = calloc(k, sizeof(Point));

        // Update the sum of points and count for each cluster
        for (int i = 0; i < n; i++) {
            count[clusters[i]]++;
            sum[clusters[i]].annual_income += points[i].annual_income;
            sum[clusters[i]].spending_score += points[i].spending_score;
        }

        // Update the centroids by calculating the average position of points in each cluster
        for (int j = 0; j < k; j++) {
            if (count[j] > 0) {
                centroids[j].annual_income = sum[j].annual_income / count[j];
                centroids[j].spending_score = sum[j].spending_score / count[j];
            }
        }

        // Free the memory allocated for counts and sums
        free(count);
        free(sum);
    }
}

// Function to print statistics and labels for each cluster
void print_cluster_stats(Point *points, int n, int k, int *clusters, Point *centroids) {
    printf("\nCluster Characteristics:\n");
    for (int j = 0; j < k; j++) {
        float total_income = 0, total_score = 0; // Initialize total income and score for the cluster
        int count = 0; // Initialize the count of points in the cluster
        for (int i = 0; i < n; i++) {
            if (clusters[i] == j) {
                // Add the income and score of each point in the cluster
                total_income += points[i].annual_income;
                total_score += points[i].spending_score;
                count++;
            }
        }

        // Calculate the average income and score for the cluster
        float avg_income = total_income / count;
        float avg_score = total_score / count;

        // Print the cluster characteristics
        printf("Cluster %d -> Avg Income: %.2f, Avg Score: %.2f, Count: %d\n",
               j, avg_income, avg_score, count);

        // Assign and print labels based on cluster characteristics
        if (avg_income < 40 && avg_score > 60) {
            printf("  Label: Budget-Conscious Shoppers\n");
        } else if (avg_income > 70 && avg_score < 40) {
            printf("  Label: Luxury Shoppers\n");
        } else {
            printf("  Label: Average Shoppers\n");
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        // Print usage instructions if the number of arguments is incorrect
        printf("Usage: %s <num_clusters> <file_path>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int k = atoi(argv[1]); // Parse the number of clusters from the command-line arguments
    const char *file_path = argv[2]; // Parse the file path from the command-line arguments

    Point *points; // Declare a pointer for storing data points
    int n; // Declare a variable for storing the number of data points

    // Load data points from the specified file
    load_data(&points, &n, file_path);

    // Allocate memory for cluster assignments and centroids
    int *clusters = malloc(n * sizeof(int));
    Point *centroids = malloc(k * sizeof(Point));

    srand(time(NULL)); // Seed the random number generator with the current time
    for (int i = 0; i < k; i++) {
        // Randomly initialize centroids by selecting random points from the dataset
        centroids[i] = points[rand() % n];
    }

    // Measure the execution time of the k-means algorithm
    clock_t start = clock();
    kmeans(points, n, k, clusters, centroids);
    clock_t end = clock();

    // Calculate and print the execution time
    double exec_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("\nCluster Assignments:\n");

    // Print the cluster assignments for each point
    for (int i = 0; i < n; i++) {
        printf("Point (Income: %.2f, Score: %.2f) -> Cluster %d\n",
               points[i].annual_income, points[i].spending_score, clusters[i]);
    }

    printf("\nExecution Time: %.3f seconds\n", exec_time);

    // Print cluster statistics and labels
    print_cluster_stats(points, n, k, clusters, centroids);

    // Free the memory allocated for points, clusters, and centroids
    free(points);
    free(clusters);
    free(centroids);

    return 0; // Exit the program successfully
}

