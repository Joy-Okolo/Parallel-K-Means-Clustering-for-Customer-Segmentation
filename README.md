# Parallel-K-Means-Clustering-for-Customer-Segmentation
The goal of this project is to accelerate K-Means clustering for customer segmentation using parallel programming.
Given a mall customer dataset with annual income and spending score, the task was to group customers into meaningful clusters for insights such as targeted marketing or recommendation systems.

Traditional (sequential) K-Means performs clustering iteratively on every data point and is computationally expensive for large datasets. This project investigates how OpenMP (shared-memory parallelism) and MPI (distributed-memory parallelism) can improve performance by leveraging multiple cores and nodes.

Core Implementations
Sequential Baseline

Language: C

Implements classical K-Means:

Random centroid initialization

Iterative assignment and centroid update

Convergence check based on centroid stability

Complexity: O(n × k × i)

Limitation: Linear execution time; cannot scale beyond one core.

OpenMP Parallel Implementation

Parallelized assignment and update loops with #pragma omp parallel for.

Used atomic and critical sections for safe centroid updates.

Optimized for: Shared-memory systems.

Findings:

Speedup observed up to 6 threads, then diminishing returns due to:

Synchronization overhead

False sharing

Hardware thread limits

Best performance achieved with 2–6 threads.

Future Work: Experiment with dynamic scheduling and hybrid OpenMP + MPI for large datasets.

MPI Parallel Implementation

Distributed workload across multiple processes/nodes.

Key MPI functions:
MPI_Bcast, MPI_Scatterv, MPI_Gatherv, MPI_Reduce

Tested on Rushmore Cluster (2–16 processes, up to 4 machines).

Results:

Performance limited by communication overhead and network latency.

Speedup decreased beyond 4 processes due to synchronization and imbalance.

Hybrid model (MPI + OpenMP) recommended for scalability.

Results Summary
Configuration	Implementation	Observations
Sequential	Baseline	Linear growth in execution time with dataset size.
OpenMP (2–6 threads)	Shared-memory	Achieved noticeable speedup; best efficiency up to 6 threads.
OpenMP (8–16 threads)	Shared-memory	Performance degraded due to overhead and cache contention.
MPI (2–16 processes)	Distributed-memory	Communication overhead dominated; limited scalability.
MPI Multi-node (Rushmore)	Distributed cluster	Network latency reduced efficiency; hybrid approach proposed.
Key Learnings

K-Means is highly data-parallel and benefits from careful workload distribution.

OpenMP offers efficient intra-node speedup for medium datasets.

MPI enables large-scale distribution but is sensitive to communication cost.

Profiling, data partitioning, and hybrid models can improve scalability.

How to Compile & Run
# Sequential
gcc -o sequential_kmeans_cluster sequential_kmeans_cluster.c -lm
./sequential_kmeans_cluster <num_clusters> <file_path>

# OpenMP
gcc -o openmp_kmeans_cluster openmp_kmeans_cluster.c -fopenmp -lm
./openmp_kmeans_cluster <num_clusters> <file_path>

# MPI
mpicc -o mpi_kmeans_cluster mpi_kmeans_cluster.c -lm
mpirun -np <num_processes> ./mpi_kmeans_cluster <num_clusters> <file_path>

Conclusion

This project demonstrates how parallel programming transforms a computationally expensive algorithm like K-Means into a scalable and faster solution. While OpenMP achieved efficient parallelism on single-node systems, MPI showed scalability potential for large distributed systems. Future improvements include hybrid MPI + OpenMP approaches, dynamic load balancing, and adaptive scheduling.
