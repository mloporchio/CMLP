CXX=g++
CXX_FLAGS= -std=c++11 -O2 -I ./armadillo-9.200.7/include -Xpreprocessor -fopenmp
LD_FLAGS= -framework Accelerate -lomp

.PHONY: cleanall

%.o: %.cpp
	$(CXX) $(CXX_FLAGS) -c $^

clf_test: Layer.o MLP.o Error.o Utils.o clf_test.o
	$(CXX) $(LD_FLAGS) $^ -o clf_test

clf_search: Layer.o MLP.o Error.o Utils.o Validation.o clf_search.o
	$(CXX) $(LD_FLAGS) $^ -o clf_search

reg_test: Layer.o MLP.o Error.o Utils.o Validation.o reg_test.o
	$(CXX) $(LD_FLAGS) $^ -o reg_test

cv_test: Layer.o MLP.o Error.o Utils.o Validation.o cv_test.o
	$(CXX) $(LD_FLAGS) $^ -o cv_test

all: clf_test clf_search reg_test cv_test

cleanall:
	-rm -f *.o clf_test clf_search reg_test cv_test
