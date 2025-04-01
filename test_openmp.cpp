#include <iostream>

int main() {
    #ifdef _OPENMP
        std::cout << "OpenMP is supported! Version: " << _OPENMP << std::endl;
        
        #pragma omp parallel
        {
            #pragma omp single
            std::cout << "Number of available threads: " << omp_get_num_threads() << std::endl;
        }
    #else
        std::cout << "OpenMP is not supported!" << std::endl;
    #endif
    
    return 0;
} 