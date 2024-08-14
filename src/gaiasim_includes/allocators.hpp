#ifndef GAIA2_TEMPLATES
#define GAIA2_TEMPLATES
#include <string>

#ifdef __NVCC__
    #include <cuda_runtime_api.h>
    #include <cuda.h>
#elif __HIPCC__ || __HIPSTDPAR__
    #include <hip/hip_runtime.h>
#endif




template<typename T>
inline T *fast_allocate_vector(const long& elements) {
    long tmp = elements == 0 ? 1 : elements;
    return new T[tmp]{0};
};


template<typename T,typename I>
inline void fast_allocate_vector(T* &ptr, const I& elements) {
    long tmp = elements == 0 ? 1 : elements;
    #ifdef __NVCC__
        cudaMallocHost((void**)&ptr, sizeof(T)*tmp);
    #elif defined(__HIPSTDPAR__) || defined(__HIPCC__)
        hipDevice_t device = -1;
        hipGetDevice(&device);
        ptr = (T *) malloc(elements * sizeof(T));
        hipMemAdvise(ptr, elements * sizeof(T), hipMemAdviseSetCoarseGrain, device);
    #else
        ptr = new T[tmp];
    #endif
}



template<typename T, typename I>
inline void allocate_vector(T* &ptr, const I& elements, const std::string &vec_name, const int& rank) {
    if(elements){
        fast_allocate_vector<T>(ptr,elements);
        if (!ptr) {exit(err_malloc(vec_name.c_str(), rank));}
    }
};

template<typename T, typename I>
inline T *allocate_vector(const I& elements, const std::string &vec_name, const int& rank) {
    T* tmp{nullptr};
    if(elements){
        fast_allocate_vector<T>(tmp,elements);
        if (!tmp) {exit(err_malloc(vec_name.c_str(), rank));}
    }
    return tmp;
};

template<typename T>
inline T *allocate_vector(const long& elements, const std::string &vec_name, const int& rank) {
    T* tmp{nullptr};
    if(elements){
        fast_allocate_vector<T>(tmp,elements);
        if (!tmp) {exit(err_malloc(vec_name.c_str(), rank));}
    }
    return tmp;
};

template<typename T>
inline T *allocate_vector(const int& elements, const std::string &vec_name, const int& rank) {
    T* tmp{nullptr};
    if(elements){
        fast_allocate_vector<T>(tmp,elements);
        if (!tmp) {exit(err_malloc(vec_name.c_str(), rank));}
    }
    return tmp;
};


template<typename T>
inline void free_mem(T *const & ptr){
    #ifdef __NVCC__
        cudaFreeHost(ptr);
    #elif __HIPSTDPAR__ || __HIPCC__
        hipHostFree(ptr);
    #else
        delete[] ptr;
    #endif
}



#endif