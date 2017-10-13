include ../support/Makefile.inc

BIN ?= bin

all: $(BIN)/process

$(BIN)/diff_conv_layer_exec: diff_conv_layer_generator.cpp $(GENERATOR_DEPS)
	@-mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -g -fno-rtti $(filter-out %.h,$^) -o $@ $(LDFLAGS)

$(BIN)/diff_conv_layer.a: $(BIN)/diff_conv_layer_exec
	@-mkdir -p $(BIN)
	$^ -g diff_conv_layer -o $(BIN) -f diff_conv_layer target=$(HL_TARGET) auto_schedule=false

$(BIN)/diff_conv_layer_auto_schedule.a: $(BIN)/diff_conv_layer_exec
	@-mkdir -p $(BIN)
	$^ -g diff_conv_layer -o $(BIN) -f diff_conv_layer_auto_schedule target=$(HL_TARGET)-no_runtime auto_schedule=true

$(BIN)/process: process.cpp $(BIN)/diff_conv_layer.a $(BIN)/diff_conv_layer_auto_schedule.a
	@-mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -I$(BIN) -Wall -O3 $^ -o $@ $(LDFLAGS)

run: $(BIN)/process
	@-mkdir -p $(BIN)
	$(BIN)/process

clean:
	rm -rf $(BIN)
