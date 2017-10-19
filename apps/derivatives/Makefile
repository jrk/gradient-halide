include ../support/Makefile.inc

.DEFAULT_GOAL := all

.PHONY:all
all: run

CXXFLAGS += -g

run: $(BIN)/main
	@-mkdir -p $(BIN)
	$(BIN)/main

$(BIN)/main: main.cpp $(BIN)/hdrnet_slice.a
	@-mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -I$(BIN) -Wall -O3 $^ -o $@ $(IMAGE_IO_FLAGS) $(LDFLAGS)

$(BIN)/hdrnet_slice.a: $(BIN)/hdrnet_slice_exec
	@-mkdir -p $(BIN)
	$^ -g hdrnet_slice -o $(BIN) -f hdrnet_slice target=$(HL_TARGET)

$(BIN)/hdrnet_slice_exec: hdrnet_slice.cpp $(GENERATOR_DEPS) $(LIB_HALIDE)
	@-mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -g -fno-rtti $(filter-out %.h,$^) -o $@ $(LDFLAGS) $(HALIDE_SYSTEM_LDFLAGS)

$(BIN)/convnet: convnet.cpp $(LIB_HALIDE)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(IMAGE_IO_FLAGS) $(HALIDE_SYSTEM_LDFLAGS)

$(BIN)/blur: blur.cpp $(LIB_HALIDE)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(IMAGE_IO_FLAGS) $(HALIDE_SYSTEM_LDFLAGS)

$(BIN)/demosaicking: demosaicking.cpp $(LIB_HALIDE)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(IMAGE_IO_FLAGS) $(HALIDE_SYSTEM_LDFLAGS)

.PHONY:clean
clean:
	rm -rf $(BIN)
