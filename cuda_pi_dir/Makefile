EXEC=calculate_pi
NVCC=nvcc

$(EXEC): calculate_pi.cu timer.hpp
	$(NVCC) $(CXXFLAGS) -o $(EXEC) calculate_pi.cu

clean:
	rm -f $(EXEC) 

