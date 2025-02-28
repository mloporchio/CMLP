CXX=g++
CXX_FLAGS= -std=c++11 -O2 -I ./armadillo-9.200.7/include -Xpreprocessor -fopenmp
LD_FLAGS= -framework Accelerate -lomp

.PHONY: cleanall

%.o: %.cpp
	$(CXX) $(CXX_FLAGS) -c $^

monks_curve: Layer.o MLP.o Error.o Utils.o monks_curve.o
	$(CXX) $(LD_FLAGS) $^ -o monks_curve

monks_test: Layer.o MLP.o Error.o Utils.o monks_test.o
	$(CXX) $(LD_FLAGS) $^ -o monks_test

monks_val: Layer.o MLP.o Error.o Utils.o Validation.o monks_val.o
	$(CXX) $(LD_FLAGS) $^ -o monks_val

cup_test: Layer.o MLP.o Error.o Utils.o Validation.o cup_test.o
	$(CXX) $(LD_FLAGS) $^ -o cup_test

cup_curve: Layer.o MLP.o Error.o Utils.o cup_curve.o
	$(CXX) $(LD_FLAGS) $^ -o cup_curve

cup_rand: Layer.o MLP.o Error.o Utils.o Validation.o cup_rand.o
	$(CXX) $(LD_FLAGS) $^ -o cup_rand

cup_val: Layer.o MLP.o Error.o Utils.o Validation.o cup_val.o
	$(CXX) $(LD_FLAGS) $^ -o cup_val

cup_xv: Layer.o MLP.o Error.o Utils.o Validation.o cup_xv.o
	$(CXX) $(LD_FLAGS) $^ -o cup_xv

all: monks_curve monks_test monks_val cup_curve cup_test cup_rand cup_val

cleanall:
	-rm -f *.o monks_curve monks_test monks_val cup_curve cup_test cup_rand cup_val
