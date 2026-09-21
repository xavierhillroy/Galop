# ---- Configuration ----
CXX      := g++
CXXFLAGS := -std=c++20 -Wall -Wextra -Wpedantic -O2 -fopenmp
DEPFLAGS := -MMD -MP

# Include dirs: headers now live in src/.
INCLUDES := -Isrc

# ---- Layout ----
SRC_DIR   := src
TOOLS_DIR := tools
TEST_DIR  := tests
BUILD_DIR := build

# Tell make where to find .cpp sources regardless of which dir they're in,
# so the %.o rule below doesn't need to know the source's directory.
vpath %.cpp $(SRC_DIR) $(TOOLS_DIR) $(TEST_DIR)

# ---- Files ----
# Listed by basename only; vpath resolves the directory.
CORE_SRCS := LGPEngine.cpp Interpreter.cpp Evaluator.cpp Dataset.cpp EvaluatorCpu.cpp
CORE_OBJS := $(addprefix $(BUILD_DIR)/,$(CORE_SRCS:.cpp=.o))

# Each binary has its own entry-point translation unit.
TEST_OBJS := $(CORE_OBJS) $(BUILD_DIR)/test_bed.o
RUN_OBJS  := $(CORE_OBJS) $(BUILD_DIR)/main.o
GEN_OBJS  := $(BUILD_DIR)/Dataset.o $(BUILD_DIR)/gen_datasets.o   # only needs Dataset
GENSW_OBJS := $(BUILD_DIR)/Dataset.o $(BUILD_DIR)/gen_sweep_datasets.o

# All objects we might build, for dependency includes and clean.
ALL_OBJS := $(CORE_OBJS) $(BUILD_DIR)/test_bed.o $(BUILD_DIR)/main.o $(BUILD_DIR)/gen_datasets.o
DEPS     := $(ALL_OBJS:.o=.d)

TEST_TARGET := lgp_test
RUN_TARGET  := lgp_run
GEN_TARGET  := gen_datasets

# ---- CUDA (GPU) ----
# Separate from the CPU build: nvcc compiles the .cu kernels + the host .cpp it
# needs. -arch=native targets whatever GPU is in this machine (sm_120 here,
# sm_90 on an H100 cluster node) -- portable across machines.
NVCC      := nvcc
NVCCFLAGS := -std=c++20 --expt-relaxed-constexpr -O2 -Isrc -arch=native -Xcompiler -fopenmp

# Differential correctness test: GPU kernels vs CPU oracle (tolerance compare).
CUDA_DIFF_SRCS := tests/cuda_diff.cu src/EvaluatorCuda.cu src/EvaluatorCpu.cpp \
                  src/Evaluator.cpp src/Interpreter.cpp src/Dataset.cpp src/LGPEngine.cpp
cuda_diff: $(CUDA_DIFF_SRCS)
	$(NVCC) $(NVCCFLAGS) $(CUDA_DIFF_SRCS) -o cuda_diff

# Whole-run throughput: full evolution, host loop + GPU/CPU evaluation.
CUDA_EVO_SRCS := tests/cuda_evo_bench.cu src/EvaluatorCuda.cu src/EvaluatorCpu.cpp \
                 src/Evaluator.cpp src/Interpreter.cpp src/Dataset.cpp src/LGPEngine.cpp
cuda_evo_bench: $(CUDA_EVO_SRCS)
	$(NVCC) $(NVCCFLAGS) $(CUDA_EVO_SRCS) -o cuda_evo_bench

# Memory-capacity probe (device-side buffers; driven by tools/oom_probe.py).
cuda_capacity: tests/cuda_capacity.cu src/EvaluatorCuda.cu src/Dataset.cpp
	$(NVCC) $(NVCCFLAGS) tests/cuda_capacity.cu src/EvaluatorCuda.cu src/Dataset.cpp -o cuda_capacity

.PHONY: gpu
gpu: cuda_diff cuda_evo_bench cuda_capacity

# ---- Targets ----
all: $(TEST_TARGET) $(RUN_TARGET) $(GEN_TARGET)

$(TEST_TARGET): $(TEST_OBJS)
	$(CXX) $(CXXFLAGS) $(TEST_OBJS) -o $(TEST_TARGET)

$(RUN_TARGET): $(RUN_OBJS)
	$(CXX) $(CXXFLAGS) $(RUN_OBJS) -o $(RUN_TARGET)

$(GEN_TARGET): $(GEN_OBJS)
	$(CXX) $(CXXFLAGS) $(GEN_OBJS) -o $(GEN_TARGET)

gen_sweep_datasets: $(GENSW_OBJS)
	$(CXX) $(CXXFLAGS) $(GENSW_OBJS) -o gen_sweep_datasets

# Compile any .cpp (found via vpath) into build/. The build dir is an
# order-only prerequisite (the | ) so its mtime doesn't force rebuilds.
$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(DEPFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -f $(ALL_OBJS) $(DEPS) $(TEST_TARGET) $(RUN_TARGET) $(GEN_TARGET) cuda_diff cuda_evo_bench cuda_capacity

-include $(DEPS)

.PHONY: all clean