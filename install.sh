CMAKE_BUILD_PARALLEL_LEVEL=$(cat /proc/cpuinfo | grep processor | wc -l) pip3 install ./python