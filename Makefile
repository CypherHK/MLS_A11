# ===== User Config =====
# Project root
PROJECT_ROOT := /Users/wangyucheng/Projects/MLS_A11

# TFLite Micro root 
TFLM_ROOT := $(PROJECT_ROOT)/tflite-micro

# Platform (matches  generated lib path)
GEN_DIR := $(TFLM_ROOT)/gen/osx_arm64_default_gcc
LIB     := $(GEN_DIR)/lib/libtensorflow-microlite.a

# 可执行程序名
BIN := mnist_micro

SRCS := model_inference.cc model_data.cc

# ====== Toolchain / Flags ======
CXX := clang++
CXXFLAGS := -O3 -std=c++17 -DNDEBUG -DTF_LITE_MICRO_DEBUG_LOG -DTF_LITE_STATIC_MEMORY


#EXTRA_INCS := -I$(PROJECT_ROOT)/op_resolver

# ===== Includes from TFLM tree =====
INCS := \
  -I$(TFLM_ROOT) \
  -I$(TFLM_ROOT)/tensorflow/lite \
  -I$(TFLM_ROOT)/tensorflow/lite/micro \
  -I$(TFLM_ROOT)/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include \
  -I$(TFLM_ROOT)/tensorflow/lite/micro/tools/make/downloads/gemmlowp \
  -I$(TFLM_ROOT)/tensorflow/lite/micro/tools/make/downloads/ruy \


LDFLAGS  := 
# 特殊情况
#LDFLAGS  += -fsanitize=address

.PHONY: all lib run clean show

all: $(BIN)


# 先确保库已生成
$(BIN): $(SRCS) | lib
	@echo "Linking with: $(LIB)"
	@if [ -z "$(LIB)" ]; then \
	  echo "ERROR: microlite static library not found under $(GEN_DIR)/lib"; \
	  echo "Run: make lib   (or fix GEN_DIR / library name)"; \
	  exit 2; \
	fi
	$(CXX) $(CXXFLAGS) $(INCS) $(SRCS) -o $@ $(LIB) $(LDFLAGS)

# 调用 TFLM 自带 make 脚本生成静态库
lib:
	@echo "Building TFLM static lib via tools/make ..."
	@cd $(TFLM_ROOT) && \
	  make -f tensorflow/lite/micro/tools/make/Makefile TARGET=osx microlite -j8


run: $(BIN)
	./$(BIN)

clean:
	rm -f $(BIN)

show:
	@echo "PROJECT_ROOT=$(PROJECT_ROOT)"
	@echo "TFLM_ROOT=$(TFLM_ROOT)"
	@echo "GEN_DIR=$(GEN_DIR)"
	@echo "LIB=$(LIB)"
