#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>

using namespace std;

void myFunction(int* myTable, int size);

void myFunction(int* myTable, int size) {
    printf("\n");
    printf("len of table = %d\n", size);
    int* buffor = new int[size];

    for (int i = 2048; i <= size; i *= 2) {
        int iterations = size / i;
        for (int j = 0; j < iterations; j++) {
            int start = 0 + j * i;
            int middle = i / 2 + start;
            int startStop = middle;
            int middleStop = i + start;
            for (int k = 0 + i * j; k < i * j + i; k++) {
                if (start < startStop) {
                    if (middle < middleStop) {
                        if (myTable[start] < myTable[middle])
                        {
                            buffor[k] = myTable[start++];
                        }
                        else
                        {
                            buffor[k] = myTable[middle++];
                        }
                    }
                    else
                    {
                        buffor[k] = myTable[start++];
                    }
                }
                else
                {
                    buffor[k] = myTable[middle++];
                }
            }
        }
        for (int l = 0; l < size; l++) { myTable[l] = buffor[l]; }
    }

    delete[] buffor;
}

int mergeCPU(int* table_in1, int* table_in2, int* table_in, unsigned long int len) {
    unsigned long int start = 0;
    unsigned long int start1 = 0;
    unsigned long int middle = 0;
    unsigned long int middle1 = len;
    unsigned long int size = middle1 * 2;
    for (unsigned long int j = 0; j < size; j++) {
        if (start < middle1 && middle < middle1) {
            if (table_in1[start] < table_in2[middle]) {
                table_in[j] = table_in1[start];
                start++;
            }
            else {
                table_in[j] = table_in2[middle];
                middle++;
            }
        }
        else {
            if (start < middle1) {
                table_in[j] = table_in1[start];
                start++;
            }
            else {
                table_in[j] = table_in2[middle];
                middle++;
            }

        }
    }

    return *table_in;
}

int mergeSortCPU(int* table_in, unsigned long int len) {
    unsigned long int table_length = len;
    unsigned long int half_length = table_length / 2;
    if (table_length <= 1) {
        return *table_in;
    }

    int* first_half = new int[half_length];
    int* second_half = new int[half_length];
    for (int i = 0; i < half_length; i++) {
        first_half[i] = table_in[i];
        second_half[i] = table_in[i + half_length];
    }

    *first_half = mergeSortCPU(first_half, half_length);
    *second_half = mergeSortCPU(second_half, half_length);

    *table_in = mergeCPU(first_half, second_half, table_in, half_length);

    delete[] first_half;
    delete[] second_half;

    return *table_in;
}

void generateNumbers(int* table1, int* table2, unsigned long int n) {
    int random_number = 0;
    for (int i = 0; i < n; i++) {
        random_number = rand() % 1000 + 1;
        table1[i] = random_number;
        table2[i] = random_number;
    }
}

void print_table(int* table, unsigned long int table_size) {
    printf("table = { ");
    for (unsigned long int i = 0; i < table_size - 1; i++)
        printf("%d, ", table[i]);
    printf("%d}\n", table[table_size - 1]);
}

int main() {
    const int MAX_THREADS = 1024;
    unsigned long int dynamic_size = 33554432;

    int* dynamic_table = new int[dynamic_size];
    int* dynamic_table_CPU = new int[dynamic_size];

    generateNumbers(dynamic_table, dynamic_table_CPU, dynamic_size);

    auto start_CPU = chrono::high_resolution_clock::now();
    mergeSortCPU(dynamic_table_CPU, dynamic_size);
    auto end_CPU = chrono::high_resolution_clock::now();
    chrono::duration<float> duration_CPU = end_CPU - start_CPU;

    std::cout << "CPU:: | Time = " << duration_CPU.count() << endl;

    int step = 1;
    printf("CPU: ");
    for (int i = 0; i < 16; i += 1) {
        printf("%d ", dynamic_table_CPU[i * step]);
    }
    printf("%d %d", dynamic_table_CPU[dynamic_size / 2], dynamic_table_CPU[dynamic_size - 1]);
    printf("\n");

    auto start_CPU2 = chrono::high_resolution_clock::now();
    myFunction(dynamic_table_CPU, dynamic_size);
    auto end_CPU2 = chrono::high_resolution_clock::now();
    chrono::duration<float> duration_CPU2 = end_CPU2 - start_CPU2;

    std::cout << "CPU2:: | Time = " << duration_CPU2.count() << endl;
    printf("\n");

    delete[] dynamic_table;
    delete[] dynamic_table_CPU;

    return 0;
}
