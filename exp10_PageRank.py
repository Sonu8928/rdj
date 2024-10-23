import numpy as np

# Normalize the matrix (make it a probability matrix - all columns sum to 1)
def normalizeAdjacencyMatrix(A):
    n = len(A)  # n = number of rows/columns in A
    for j in range(len(A[0])):
        sumOfCol = 0
        for i in range(len(A)):
            sumOfCol += A[i][j]
        if sumOfCol == 0:  # Adjust for dangling nodes (columns of zeros)
            for val in range(n):
                A[val][j] = 1/n
        else:
            for val in range(n):
                A[val][j] = (A[val][j] / sumOfCol)
    return A

# Implement damping matrix using the formula
# M = dA + (1-d)(1/n)Q, where Q is an array of 1's and d is the damping factor
def dampingMatrix(A, dampingFactor=0.85):
    n = len(A)  # n = number of rows/columns in A
    Q = [[1/n] * n for _ in range(n)]
    arrA = np.array(A)
    arrQ = np.array(Q)
    arrM = np.add(dampingFactor * arrA, (1 - dampingFactor) * arrQ)  # Create damping matrix
    return arrM

# Use power iteration to find the steady state vector
def findSteadyState(M, num_iterations=100, tolerance=1e-6):
    n = M.shape[0]
    # Initialize the rank vector with equal probability
    rank_vector = np.array([1/n] * n)
    for _ in range(num_iterations):
        new_rank_vector = M @ rank_vector
        # Check for convergence
        if np.linalg.norm(new_rank_vector - rank_vector) < tolerance:
            break
        rank_vector = new_rank_vector
    return rank_vector / np.sum(rank_vector)  # Normalize to sum to 1

def pageRank(A, dampingFactor=0.85, num_iterations=100, tolerance=1e-6):
    # Normalize the adjacency matrix
    A = normalizeAdjacencyMatrix(A)
    # Apply the damping factor
    M = dampingMatrix(A, dampingFactor)
    # Use power iteration to find the steady state vector
    steadyStateVector = findSteadyState(M, num_iterations, tolerance)
    return steadyStateVector

# TEST CASES
print("\nPage Rank Examples")

# Example 1 (4-node graph)
matrix1 = [
    [0, 1, 0, 0],
    [0, 0, 0, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0],
 
]
print("Example 1) matrix 1 = ")
print(np.array(matrix1))
print("Steady state vector: ")
print(pageRank(matrix1))

# Example 2 (8-node graph)
matrix2 = [
    [0, 0, 1, 0, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 0]
]
print("\nExample 2) matrix 2 = ")
print(np.array(matrix2))
print("Steady state vector: ")
print(pageRank(matrix2))
