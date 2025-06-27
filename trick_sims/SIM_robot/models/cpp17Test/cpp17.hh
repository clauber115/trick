#ifndef CPP17_HH
#define CPP17_HH

#include <type_traits>
#include <iostream>

// C++17 feature: inline variable
inline constexpr bool use_cpp17 = true;

// C++17 feature: if constexpr
template <typename T>
void print_type_info(const T& value, bool printout)
{
    if (printout) { std::cout<<"Running the type check..."<<std::endl; }

    if constexpr (std::is_integral_v<T>)
    {
        // is_integral_v is a C++17 alias for std::is_integral<T>::value
        static_assert(use_cpp17, "C++17 feature used");
    }
}

#endif
