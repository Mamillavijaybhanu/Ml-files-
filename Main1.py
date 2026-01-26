
#1.First question function to count vowels and consonants
def count_vowels_consonants(input_string):
    v_count = 0
    c_count = 0
    for character in input_string:
        if character.isalpha():
            if character.lower() in "aeiou":
                v_count += 1
            else:
                c_count += 1
    return v_count, c_count

# 2.Second question Function to multiply two matrices

def multiply_matrices(matrix_a, matrix_b):
    rows_1 = len(matrix_a)
    columns_1 = len(matrix_a[0])
    rows_2 = len(matrix_b)
    columns_2 = len(matrix_b[0])
    if columns_1 != rows_2:
        return None
    result_matrix = [[0 for _ in range(columns_2)] for _ in range(rows_1)]
    for i in range(rows_1):
        for j in range(columns_2):
            for k in range(columns_1):
                result_matrix[i][j] += matrix_a[i][k] * matrix_b[k][j]
    return result_matrix

# 3.Third question function to count common elements between two lists

def count_common_elements(list_one, list_two):
    common_elements = set(list_one) & set(list_two)
    return len(common_elements)

# 4.Fourth question function to compute transpose of a matrix

def transpose_matrix(matrix):
    r= len(matrix)
    c= len(matrix[0])

    transpose = [[matrix[j][i] for j in range(r)] for i in range(c)]
    return transpose


# 5.Fifth question function to generate random numbers and compute statistics

def calculate_statistics():
    import random
    import statistics
    numbers = [random.randint(100, 150) for _ in range(100)]

    mean_value = statistics.mean(numbers)
    median_value = statistics.median(numbers)
    mode_value = statistics.mode(numbers)

    return numbers, mean_value, median_value, mode_value


# Main Program
def main():
    # q1
    user_string = input("Enter a string: ")
    vowels, consonants = count_vowels_consonants(user_string)
    print("Vowels:", vowels)
    print("Consonants:", consonants)

    # q2
    rows_a = int(input("Enter rows of A: "))
    cols_a = int(input("Enter columns of A: "))
    matrix_a = []

    print("Enter elements of matrix A:")
    for _ in range(rows_a):
        matrix_a.append(list(map(int, input().split())))

    rows_b = int(input("Enter rows of B: "))
    cols_b = int(input("Enter columns of B: "))
    matrix_b = []

    print("Enter elements of matrix B:")
    for _ in range(rows_b):
        matrix_b.append(list(map(int, input().split())))

    product = multiply_matrices(matrix_a, matrix_b)
    if product is None:
        print("Matrix multiplication is not possible.")
    else:
        print("Product Matrix:")
        for row in product:
            print(row)

    # q3
    list_one = list(map(int, input("Enter first list: ").split()))
    list_two = list(map(int, input("Enter second list: ").split()))
    common_count = count_common_elements(list_one, list_two)
    print("Number of common elements:", common_count)

    # q4
    rows = int(input("Enter rows: "))
    columns = int(input("Enter columns: "))
    matrix = []

    print("Enter matrix elements:")
    for _ in range(rows):
        matrix.append(list(map(int, input().split())))

    transposed_matrix = transpose_matrix(matrix)
    print("Transpose Matrix:")
    for row in transposed_matrix:
        print(row)

    # q5
    numbers, mean_value, median_value, mode_value = calculate_statistics()
    print("Random Numbers:", numbers)
    print("Mean:", mean_value)
    print("Median:", median_value)
    print("Mode:", mode_value)


# Run main program
main()
