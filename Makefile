CXX = g++
MPICXX = mpicxx
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra -Wpedantic

.PHONY: all serial parallel
all: serial parallel

serial: bin/jacobi_serial
parallel: bin/jacobi_parallel

bin:
	mkdir -p bin

bin/jacobi_serial: src/serial.cpp src/common.hpp | bin
	$(CXX) $(CXXFLAGS) src/serial.cpp -o $@

bin/jacobi_parallel: src/parallel.cpp src/common.hpp | bin
	$(MPICXX) $(CXXFLAGS) -fopenmp src/parallel.cpp -o $@
