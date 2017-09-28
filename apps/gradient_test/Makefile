include ../support/Makefile.inc

.DEFAULT_GOAL := all

.PHONY:all
all: $(BIN)/convnet $(BIN)/blur;

CXXFLAGS += -g

$(BIN)/convnet: convnet.cpp $(LIB_HALIDE)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(IMAGE_IO_FLAGS) $(HALIDE_SYSTEM_LDFLAGS)

$(BIN)/blur: blur.cpp $(LIB_HALIDE)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(IMAGE_IO_FLAGS) $(HALIDE_SYSTEM_LDFLAGS)

.PHONY:clean
clean:
	rm -rf $(BIN)
