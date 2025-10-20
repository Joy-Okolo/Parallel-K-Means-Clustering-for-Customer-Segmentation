// Including necessary libraries for input/output operations, memory management, mathematical calculations, and OpenMP
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <omp.h>
#include <time.h>

// Defining the maximum number of iterations for the K-Means algorithm
#define MAX_ITER 100

// Defining a structure to represent a data point with two features: annual income and spending score
typedef struct {
    float annual_income, spending_score; // Features: annual income and spending score
} Point;

// Function prototypes for various tasks in the K-Means algorithm
void load_data(Point **points, int *n, const char *file_path); // Function to load data from a CSV file
void normalize_features(Point *points, int n); // Function to normalize features to a range of [0, 1]
float euclidean_distance(Point p1, Point p2); // Function to calculate Euclidean distance between two points
void sequential_kmeans(Point *points, int n, int k, int *clusters, Point *centroids); // Function for sequential K-Means
void parallel_kmeans(Point *points, int n, int k, int *clusters, Point *centroids); // Function for parallel K-Means using OpenMP
void calculate_cluster_characteristics(Point *points, int n, int k, int *clusters); // Function to calculate cluster characteristics

int main(int argc, char *argv[]) {
    // Check if the correct number of arguments is provided
    if (argc != 4) {
        printf("Usage: %s <num_clusters> <file_path> <num_threads>\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Parsing command-line arguments for number of clusters, file path, and number of threads
    int k = atoi(argv[1]);
    const char *file_path = argv[2];
    int num_threads = atoi(argv[3]);

    // Validating the number of threads
    if (num_threads <= 0) {
        printf("Number of threads must be a positive integer.\n");
        return EXIT_FAILURE;
    }

    // Setting the number of threads for OpenMP
    omp_set_num_threads(num_threads);
    printf("Using %d threads for computation.\n", num_threads);

    Point *points = NULL; // Pointer to store data points
    int n = 0; // Variable to store the number of data points

    // Loading data points from the file and normalizing features
    load_data(&points, &n, file_path);
    normalize_features(points, n);

    // Allocating memory for cluster assignments and centroids
    int *clusters = malloc(n * sizeof(int));
    Point *centroids = malloc(k * sizeof(Point));

    // K-Means++ initialization of centroids
    centroids[0] = points[rand() % n]; // Randomly selecting the first centroid
    for (int i = 1; i < k; i++) {
        float *distances = malloc(n * sizeof(float)); // Array to store distances of points to their nearest centroid
        float total_distance = 0.0; // Variable to accumulate total distances

        // Calculating distances to the nearest centroid
        for (int j = 0; j < n; j++) {
            float min_dist = FLT_MAX; // Initialize the minimum distance to a large value
            for (int l = 0; l < i; l++) {
                float dist = euclidean_distance(points[j], centroids[l]); // Compute distance to centroid
                if (dist < min_dist) {
                    min_dist = dist; // Update the minimum distance
                }
            }
            distances[j] = min_dist; // Store the minimum distance
            total_distance += min_dist; // Add to the total distance
        }

        // Selecting the next centroid probabilistically
        float threshold = ((float)rand() / RAND_MAX) * total_distance; // Generate a random threshold
        float cumulative_distance = 0.0; // Variable to accumulate distances
        for (int j = 0; j < n; j++) {
            cumulative_distance += distances[j]; // Accumulate distances
            if (cumulative_distance >= threshold) {
                centroids[i] = points[j]; // Select the current point as the next centroid
                break;
            }
        }

        free(distances); // Free the allocated memory for distances
    }

    // Printing the initialized centroids
    printf("Initial Centroids:\n");
    for (int j = 0; j < k; j++) {
        printf("Centroid %d -> Income: %.2f, Score: %.2f\n", j, centroids[j].annual_income, centroids[j].spending_score);
    }

    // Measuring the execution time for sequential K-Means
    double seq_start = omp_get_wtime(); // Start the timer for sequential execution
    sequential_kmeans(points, n, k, clusters, centroids); // Perform sequential K-Means
    double seq_end = omp_get_wtime(); // End the timer
    printf("\nSequential Execution Time: %f seconds\n", seq_end - seq_start);

    // Reinitializing centroids for parallel execution
    for (int i = 0; i < k; i++) {
        centroids[i] = points[rand() % n]; // Randomly selecting new centroids
    }

    // Measuring the execution time for parallel K-Means
    double par_start = omp_get_wtime(); // Start the timer for parallel execution
    parallel_kmeans(points, n, k, clusters, centroids); // Perform parallel K-Means
    double par_end = omp_get_wtime(); // End the timer

    // Printing the execution time and speedup for parallel K-Means
    printf("\nParallel Execution Time: %f seconds\n", par_end - par_start);
    printf("Speedup: %f\n", (seq_end - seq_start) / (par_end - par_start));

    // Printing the final centroids after parallel execution
    printf("Final Centroids:\n");
    for (int j = 0; j < k; j++) {
        printf("Centroid %d -> Income: %.2f, Score: %.2f\n", j, centroids[j].annual_income, centroids[j].spending_score);
    }

    // Calculating and printing cluster characteristics
    calculate_cluster_characteristics(points, n, k, clusters);

    // Printing the execution summary
    printf("\nExecution Summary:\n");
    printf("Total Points: %d\n", n);
    printf("Number of Clusters: %d\n", k);
    printf("Sequential Execution Time: %f seconds\n", seq_end - seq_start);
    printf("Parallel Execution Time: %f seconds\n", par_end - par_start);
    printf("Speedup Achieved: %f\n", (seq_end - seq_start) / (par_end - par_start));

    // Freeing allocated memory for points, clusters, and centroids
    free(points);
    free(clusters);
    free(centroids);

    return 0; // End of program
}

// Function to load data from a CSV file
void load_data(Point **points, int *n, const char *file_path) {
    FILE *file = fopen(file_path, "r"); // Open the file for reading
    if (!file) {
        perror("Error opening file"); // Print an error message if the file cannot be opened
        exit(EXIT_FAILURE); // Exit the program with failure status
    }
    char line[256]; // Buffer to store each line from the file
    fgets(line, sizeof(line), file); // Skip the header line
    *n = 0; // Initialize the count of data points
    while (fgets(line, sizeof(line), file)) {
        (*n)++; // Increment the count for each line
    }
    rewind(file); // Reset the file pointer to the beginning
    fgets(line, sizeof(line), file); // Skip the header line again
    *points = (Point *)malloc((*n) * sizeof(Point)); // Allocate memory for the points
    int index = 0; // Index to keep track of the current point
    while (fgets(line, sizeof(line), file)) {
        sscanf(line, "%*d,%*[^,],%*d,%f,%f", &(*points)[index].annual_income, &(*points)[index].spending_score); // Parse income and score
        index++; // Move to the next point
    }
    fclose(file); // Close the file
}

// Function to normalize features to a range of [0, 1]
void normalize_features(Point *points, int n) {
    float max_income = -FLT_MAX, min_income = FLT_MAX; // Initialize min and max for income
    float max_score = -FLT_MAX, min_score = FLT_MAX; // Initialize min and max for score

    // Find the min and max values for income and score
    for (int i = 0; i < n; i++) {
        if (points[i].annual_income > max_income) max_income = points[i].annual_income;
        if (points[i].annual_income < min_income) min_income = points[i].annual_income;
        if (points[i].spending_score > max_score) max_score = points[i].spending_score;
        if (points[i].spending_score < min_score) min_score = points[i].spending_score;
    }

    // Normalize each point's income and score to [0, 1]
    for (int i = 0; i < n; i++) {
        points[i].annual_income = (points[i].annual_income - min_income) / (max_income - min_income);
        points[i].spending_score = (points[i].spending_score - min_score) / (max_score - min_score);
    }
}

// Function to calculate the Euclidean distance between two points
float euclidean_distance(Point p1, Point p2) {
    return sqrt(pow(p1.annual_income - p2.annual_income, 2) +
                pow(p1.spending_score - p2.spending_score, 2)); // Calculate and return the distance
}

// Function for sequential K-Means
void sequential_kmeans(Point *points, int n, int k, int *clusters, Point *centroids) {
    for (int iter = 0; iter < MAX_ITER; iter++) { // Loop for a fixed number of iterations
        for (int i = 0; i < n; i++) { // Assign each point to the nearest cluster
            float min_dist = FLT_MAX; // Initialize minimum distance
            int cluster = 0; // Variable to store the nearest cluster
            for (int j = 0; j < k; j++) {
                float dist = euclidean_distance(points[i], centroids[j]); // Calculate distance to centroid
                if (dist < min_dist) {
                    min_dist = dist; // Update minimum distance
                    cluster = j; // Update nearest cluster
                }
            }
            clusters[i] = cluster; // Assign the point to the nearest cluster
        }

        int *count = calloc(k, sizeof(int)); // Allocate memory for counting points in each cluster
        Point *sum = calloc(k, sizeof(Point)); // Allocate memory for summing point features in each cluster

        for (int i = 0; i < n; i++) { // Sum the features of points in each cluster
            count[clusters[i]]++;
            sum[clusters[i]].annual_income += points[i].annual_income;
            sum[clusters[i]].spending_score += points[i].spending_score;
        }

        for (int j = 0; j < k; j++) { // Update the centroids based on the mean of the points in each cluster
            if (count[j] > 0) {
                centroids[j].annual_income = sum[j].annual_income / count[j];
                centroids[j].spending_score = sum[j].spending_score / count[j];
            }
        }

        free(count); // Free the allocated memory for count
        free(sum); // Free the allocated memory for sum
    }
}

// Function for parallel K-Means using OpenMP
void parallel_kmeans(Point *points, int n, int k, int *clusters, Point *centroids) {
    for (int iter = 0; iter < MAX_ITER; iter++) { // Loop for a fixed number of iterations
        #pragma omp parallel for // Parallelize the loop for assigning points to clusters
        for (int i = 0; i < n; i++) {
            float min_dist = FLT_MAX; // Initialize minimum distance
            int cluster = 0; // Variable to store the nearest cluster
            for (int j = 0; j < k; j++) {
                float dist = euclidean_distance(points[i], centroids[j]); // Calculate distance to centroid
                if (dist < min_dist) {
                    min_dist = dist; // Update minimum distance
                    cluster = j; // Update nearest cluster
                }
            }
            clusters[i] = cluster; // Assign the point to the nearest cluster
        }

        Point *local_sum = calloc(k, sizeof(Point)); // Allocate memory for summing features locally
        int *local_count = calloc(k, sizeof(int)); // Allocate memory for counting points locally

        #pragma omp parallel for // Parallelize the loop for summing point features
        for (int i = 0; i < n; i++) {
            int cluster = clusters[i]; // Get the cluster of the point
            #pragma omp atomic
            local_count[cluster]++; // Increment the count atomically
            #pragma omp atomic
            local_sum[cluster].annual_income += points[i].annual_income; // Add income atomically
            #pragma omp atomic
            local_sum[cluster].spending_score += points[i].spending_score; // Add score atomically
        }

        #pragma omp critical // Ensure only one thread updates the global centroids at a time
        {
            for (int j = 0; j < k; j++) {
                centroids[j].annual_income += local_sum[j].annual_income; // Update centroid income
                centroids[j].spending_score += local_sum[j].spending_score; // Update centroid score
                centroids[j].annual_income /= local_count[j]; // Calculate mean income
                centroids[j].spending_score /= local_count[j]; // Calculate mean score
            }
        }

        free(local_sum); // Free the allocated memory for local sums
        free(local_count); // Free the allocated memory for local counts
    }
}

// Function to calculate and print cluster characteristics
void calculate_cluster_characteristics(Point *points, int n, int k, int *clusters) {
    printf("\nCluster Characteristics:\n"); // Print header
    for (int j = 0; j < k; j++) { // Loop through each cluster
        float total_income = 0, total_score = 0; // Initialize totals
        int count = 0; // Initialize count
        for (int i = 0; i < n; i++) { // Sum the features of points in the cluster
            if (clusters[i] == j) {
                total_income += points[i].annual_income;
                total_score += points[i].spending_score;
                count++;
            }
        }
        float avg_income = total_income / count; // Calculate average income
        float avg_score = total_score / count; // Calculate average score
        printf("Cluster %d -> Avg Income: %.2f, Avg Score: %.2f, Count: %d\n", j, avg_income, avg_score, count); // Print cluster details

        // Assign labels based on characteristics
        if (avg_income < 0.4 && avg_score > 0.6) {
            printf("  Label: Budget-Conscious Shoppers\n");
        } else if (avg_income > 0.7 || avg_score < 0.4) {
            printf("  Label: Luxury Shoppers\n");
        } else {
            printf("  Label: Average Shoppers\n");
        }
    }
}

