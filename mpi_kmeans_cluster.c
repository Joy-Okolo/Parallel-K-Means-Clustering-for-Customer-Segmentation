// Including necessary header files for standard input/output, memory allocation, math functions, MPI, etc.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <mpi.h>
#include <time.h>

// Define a constant for the maximum number of iterations
#define MAX_ITER 100

// Define a structure to represent a data point with two features
typedef struct {
    float annual_income, spending_score; // Features: annual income and spending score
} Point;

// Function prototypes for various operations in the K-Means algorithm
void load_data(Point **points, int *n, const char *file_path); // Load data from a file
void normalize_features(Point *points, int n); // Normalize features to [0, 1] range
float euclidean_distance(Point p1, Point p2); // Compute Euclidean distance between two points
void sequential_kmeans(Point *points, int n, int k, int *clusters, Point *centroids); // Sequential K-Means algorithm
void parallel_kmeans(Point *points, int n, int k, int *clusters, Point *centroids, int rank, int size); // Parallel K-Means algorithm using MPI
void calculate_cluster_characteristics(Point *points, int n, int k, int *clusters); // Calculate cluster characteristics

// Main function
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv); // Initialize MPI environment

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the process
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the total number of processes

    if (rank == 0) {
        printf("Running K-Means with %d processes.\n", size); // Print the number of processes
    }

    // Check if the required number of arguments are provided
    if (argc != 3) {
        if (rank == 0) {
            printf("Usage: %s <num_clusters> <file_path>\n", argv[0]);
        }
        MPI_Finalize(); // Finalize MPI if arguments are incorrect
        return EXIT_FAILURE;
    }

    int k = atoi(argv[1]); // Parse the number of clusters from the arguments
    const char *file_path = argv[2]; // Parse the file path from the arguments

    Point *points = NULL; // Pointer to hold data points
    int n = 0; // Number of data points

    if (rank == 0) {
        load_data(&points, &n, file_path); // Load data points from the file
        normalize_features(points, n); // Normalize the data features
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD); // Broadcast the number of points to all processes

    int *clusters = malloc(n * sizeof(int)); // Array to store cluster assignments
    Point *centroids = malloc(k * sizeof(Point)); // Array to store centroids

    double sequential_time = 0.0; // Variable to track sequential execution time

    if (rank == 0) {
        // K-means++ initialization for centroids
        centroids[0] = points[rand() % n]; // Select the first centroid randomly
        for (int i = 1; i < k; i++) {
            float *distances = malloc(n * sizeof(float)); // Array to hold distances to nearest centroids
            float total_distance = 0.0; // Sum of all distances

            // Calculate the distance to the nearest centroid for each point
            for (int j = 0; j < n; j++) {
                float min_dist = FLT_MAX; // Initialize minimum distance
                for (int l = 0; l < i; l++) {
                    float dist = euclidean_distance(points[j], centroids[l]); // Compute distance to centroid
                    if (dist < min_dist) {
                        min_dist = dist; // Update minimum distance
                    }
                }
                distances[j] = min_dist; // Store the minimum distance
                total_distance += min_dist; // Add to total distance
            }

            // Select the next centroid probabilistically based on distances
            float threshold = ((float)rand() / RAND_MAX) * total_distance;
            float cumulative_distance = 0.0;
            for (int j = 0; j < n; j++) {
                cumulative_distance += distances[j]; // Accumulate distances
                if (cumulative_distance >= threshold) {
                    centroids[i] = points[j]; // Select this point as the next centroid
                    break;
                }
            }

            free(distances); // Free the memory for distances
        }

        // Print initial centroids
        printf("Initial Centroids:\n");
        for (int j = 0; j < k; j++) {
            printf("Centroid %d -> Income: %.2f, Score: %.2f\n", j, centroids[j].annual_income, centroids[j].spending_score);
        }

        double seq_start = MPI_Wtime(); // Start the timer for sequential execution
        sequential_kmeans(points, n, k, clusters, centroids); // Run the sequential K-Means algorithm
        double seq_end = MPI_Wtime(); // End the timer
        sequential_time = seq_end - seq_start; // Calculate the execution time

        printf("\nSequential Execution Time: %f seconds\n", sequential_time);

        // Reinitialize centroids for parallel execution
        for (int i = 0; i < k; i++) {
            centroids[i] = points[rand() % n]; // Select random points as centroids
        }
    }

    MPI_Bcast(&sequential_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); // Broadcast sequential time
    MPI_Bcast(centroids, k * 2, MPI_FLOAT, 0, MPI_COMM_WORLD); // Broadcast centroids to all processes

    double par_start = MPI_Wtime(); // Start the timer for parallel execution
    parallel_kmeans(points, n, k, clusters, centroids, rank, size); // Run the parallel K-Means algorithm
    double par_end = MPI_Wtime(); // End the timer

    if (rank == 0) {
        double parallel_time = par_end - par_start; // Calculate parallel execution time
        double speedup = sequential_time / parallel_time; // Calculate speedup

        printf("Parallel Execution Time: %f seconds\n", parallel_time);
        printf("Speedup: %f\n", speedup);

        // Print final centroids
        printf("Final Centroids:\n");
        for (int j = 0; j < k; j++) {
            printf("Centroid %d -> Income: %.2f, Score: %.2f\n", j, centroids[j].annual_income, centroids[j].spending_score);
        }

        calculate_cluster_characteristics(points, n, k, clusters); // Analyze cluster characteristics

        // Print summary
        printf("\nExecution Summary:\n");
        printf("Sequential Execution Time: %f seconds\n", sequential_time);
        printf("Parallel Execution Time: %f seconds\n", parallel_time);
        printf("Speedup: %f\n", speedup);
    }

    free(points); // Free the allocated memory for points
    free(clusters); // Free the allocated memory for clusters
    free(centroids); // Free the allocated memory for centroids

    MPI_Finalize(); // Finalize MPI environment
    return 0; // End of program
}

// Function to load data points from a CSV file
void load_data(Point **points, int *n, const char *file_path) {
    FILE *file = fopen(file_path, "r"); // Open the file
    if (!file) {
        perror("Error opening file"); // Print error if file cannot be opened
        exit(EXIT_FAILURE);
    }
    char line[256]; // Buffer to read lines
    fgets(line, sizeof(line), file); // Skip the header line
    *n = 0; // Initialize point count
    while (fgets(line, sizeof(line), file)) {
        (*n)++; // Count number of points
    }
    rewind(file); // Reset file pointer
    fgets(line, sizeof(line), file); // Skip the header again
    *points = (Point *)malloc((*n) * sizeof(Point)); // Allocate memory for points
    int index = 0;
    while (fgets(line, sizeof(line), file)) {
        sscanf(line, "%*d,%*[^,],%*d,%f,%f", &(*points)[index].annual_income, &(*points)[index].spending_score); // Parse point data
        index++;
    }
    fclose(file); // Close the file
}

// Normalize features to a range [0, 1]
void normalize_features(Point *points, int n) {
    float max_income = -FLT_MAX, min_income = FLT_MAX;
    float max_score = -FLT_MAX, min_score = FLT_MAX;

    for (int i = 0; i < n; i++) {
        if (points[i].annual_income > max_income) max_income = points[i].annual_income;
        if (points[i].annual_income < min_income) min_income = points[i].annual_income;
        if (points[i].spending_score > max_score) max_score = points[i].spending_score;
        if (points[i].spending_score < min_score) min_score = points[i].spending_score;
    }

    for (int i = 0; i < n; i++) {
        points[i].annual_income = (points[i].annual_income - min_income) / (max_income - min_income);
        points[i].spending_score = (points[i].spending_score - min_score) / (max_score - min_score);
    }
}

// Compute Euclidean distance between two points
float euclidean_distance(Point p1, Point p2) {
    return sqrt(pow(p1.annual_income - p2.annual_income, 2) +
                pow(p1.spending_score - p2.spending_score, 2));
}

// Perform sequential K-Means
void sequential_kmeans(Point *points, int n, int k, int *clusters, Point *centroids) {
    for (int iter = 0; iter < MAX_ITER; iter++) {
        for (int i = 0; i < n; i++) {
            float min_dist = FLT_MAX;
            int cluster = 0;
            for (int j = 0; j < k; j++) {
                float dist = euclidean_distance(points[i], centroids[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    cluster = j;
                }
            }
            clusters[i] = cluster;
        }

        int *count = calloc(k, sizeof(int));
        Point *sum = calloc(k, sizeof(Point));

        for (int i = 0; i < n; i++) {
            count[clusters[i]]++;
            sum[clusters[i]].annual_income += points[i].annual_income;
            sum[clusters[i]].spending_score += points[i].spending_score;
        }

        for (int j = 0; j < k; j++) {
            if (count[j] > 0) {
                centroids[j].annual_income = sum[j].annual_income / count[j];
                centroids[j].spending_score = sum[j].spending_score / count[j];
            }
        }

        free(count);
        free(sum);
    }
}

// Perform parallel K-Means using MPI
void parallel_kmeans(Point *points, int n, int k, int *clusters, Point *centroids, int rank, int size) {
    int local_n = n / size + (rank < n % size); // Calculate local data size per process
    Point *local_points = malloc(local_n * sizeof(Point)); // Allocate memory for local data
    int *local_clusters = malloc(local_n * sizeof(int)); // Local cluster assignments

    int *sendcounts = malloc(size * sizeof(int)); // Number of elements to send to each process
    int *displs = malloc(size * sizeof(int)); // Displacements for each process
    for (int i = 0, offset = 0; i < size; i++) {
        sendcounts[i] = n / size + (i < n % size ? 1 : 0);
        displs[i] = offset;
        offset += sendcounts[i];
    }
    MPI_Scatterv(points, sendcounts, displs, MPI_FLOAT, local_points, sendcounts[rank] * 2, MPI_FLOAT, 0, MPI_COMM_WORLD);

    printf("Processor %d is handling %d data points\n", rank, sendcounts[rank]);
    MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processes

    for (int iter = 0; iter < MAX_ITER; iter++) {
        for (int i = 0; i < local_n; i++) {
            float min_dist = FLT_MAX;
            int cluster = 0;
            for (int j = 0; j < k; j++) {
                float dist = euclidean_distance(local_points[i], centroids[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    cluster = j;
                }
            }
            local_clusters[i] = cluster;
        }

        Point *local_sum = calloc(k, sizeof(Point));
        int *local_count = calloc(k, sizeof(int));

        for (int i = 0; i < local_n; i++) {
            local_sum[local_clusters[i]].annual_income += local_points[i].annual_income;
            local_sum[local_clusters[i]].spending_score += local_points[i].spending_score;
            local_count[local_clusters[i]]++;
        }

        Point *global_sum = calloc(k, sizeof(Point));
        int *global_count = calloc(k, sizeof(int));

        MPI_Reduce(local_sum, global_sum, k * 2, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD); // Reduce local sums
        MPI_Reduce(local_count, global_count, k, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD); // Reduce local counts

        if (rank == 0) {
            for (int j = 0; j < k; j++) {
                if (global_count[j] > 0) {
                    centroids[j].annual_income = global_sum[j].annual_income / global_count[j];
                    centroids[j].spending_score = global_sum[j].spending_score / global_count[j];
                }
            }
        }
        MPI_Bcast(centroids, k * 2, MPI_FLOAT, 0, MPI_COMM_WORLD); // Broadcast updated centroids

        free(local_sum);
        free(local_count);
        free(global_sum);
        free(global_count);
    }

    MPI_Gatherv(local_clusters, local_n, MPI_INT, clusters, sendcounts, displs, MPI_INT, 0, MPI_COMM_WORLD); // Gather cluster assignments

    free(local_points);
    free(local_clusters);
    free(sendcounts);
    free(displs);
}

// Compute and print cluster characteristics
void calculate_cluster_characteristics(Point *points, int n, int k, int *clusters) {
    printf("\nCluster Characteristics:\n");
    for (int j = 0; j < k; j++) {
        float total_income = 0, total_score = 0;
        int count = 0;
        for (int i = 0; i < n; i++) {
            if (clusters[i] == j) {
                total_income += points[i].annual_income;
                total_score += points[i].spending_score;
                count++;
            }
        }
        float avg_income = total_income / count;
        float avg_score = total_score / count;
        printf("Cluster %d -> Avg Income: %.2f, Avg Score: %.2f, Count: %d\n", j, avg_income, avg_score, count);

        // Label clusters based on characteristics
        if (avg_income < 0.4 && avg_score > 0.6) {
            printf("  Label: Budget-Conscious Shoppers\n");
        } else if (avg_income > 0.7 || avg_score >0.4) {
            printf("  Label: Luxury Shoppers\n");
        } else {
            printf("  Label: Average Shoppers\n");
        }
    }
}

