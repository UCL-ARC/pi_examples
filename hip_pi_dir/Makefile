EXEC=calculate_pi

$(EXEC): calculate_pi.hip timer.hpp
	hipcc $(CXXFLAGS) -o $(EXEC) calculate_pi.hip

clean:
	rm -f $(EXEC) 

