include ../support/Makefile.inc

BIN ?= bin

all: $(BIN)/process

$(BIN)/bilateral_layer.generator: bilateral_layer_generator.cpp $(GENERATOR_DEPS)
	@-mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -g -fno-rtti $(filter-out %.h,$^) -o $@ $(LDFLAGS)

$(BIN)/bilateral_layer.a: $(BIN)/bilateral_layer.generator
	@-mkdir -p $(BIN)
	$^ -g bilateral_layer -o $(BIN) -f bilateral_layer target=$(HL_TARGET) auto_schedule=false

$(BIN)/process: process.cpp $(BIN)/bilateral_layer.a $(BIN)/bilateral_layer.a
	@-mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -I$(BIN) -Wall -O3 $^ -o $@ $(LDFLAGS) -L../../lib

run: $(BIN)/process
	@-mkdir -p $(BIN)
	$(BIN)/process

clean:
	rm -rf $(BIN)
