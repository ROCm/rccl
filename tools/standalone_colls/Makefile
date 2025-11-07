
# Makefile for Ring AllReduce Benchmark

# Compiler
HIPCC = hipcc

# Target executable
TARGET = ring_allreduce_bench

# Source files
SRC = ring_allreduce_bench.hip

# Compiler flags
CXXFLAGS = -O3 -std=c++11

# ROCm architecture (adjust based on your GPU)
# Common values: gfx906 (MI50/MI60), gfx908 (MI100), gfx90a (MI210/MI250), gfx950 (MI355X)
ARCH = gfx950

# Additional flags
HIPFLAGS = --offload-arch=$(ARCH)

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(SRC)
	$(HIPCC) $(CXXFLAGS) $(HIPFLAGS) -o $(TARGET) $(SRC)

# Clean target
clean:
	rm -f $(TARGET) *.o

# Run with default parameters (threadblock mode, 1GB)
run: $(TARGET)
	./$(TARGET)

# Run threadblock mode
run-threadblock: $(TARGET)
	./$(TARGET) -m threadblock -b 28

# Run warp mode
run-warp: $(TARGET)
	./$(TARGET) -m warp -b 32

# Run with different sizes
run-1gb: $(TARGET)
	./$(TARGET) -s $$((1024*1024*1024))

run-2gb: $(TARGET)
	./$(TARGET) -s $$((2*1024*1024*1024))

run-4gb: $(TARGET)
	./$(TARGET) -s $$((4*1024*1024*1024))

# Help target
help:
	@echo "Ring AllReduce Benchmark Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all              - Build the benchmark (default)"
	@echo "  clean            - Remove built files"
	@echo "  run              - Run with default parameters"
	@echo "  run-threadblock  - Run in threadblock mode"
	@echo "  run-warp         - Run in warp mode"
	@echo "  run-1gb          - Run with 1GB data"
	@echo "  run-2gb          - Run with 2GB data"
	@echo "  run-4gb          - Run with 4GB data"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Variables:"
	@echo "  ARCH             - GPU architecture (default: gfx90a)"
	@echo ""
	@echo "Usage examples:"
	@echo "  make"
	@echo "  make run"
	@echo "  make ARCH=gfx908 all"
	@echo "  ./$(TARGET) -h"

.PHONY: all clean run run-threadblock run-warp run-1gb run-2gb run-4gb help


