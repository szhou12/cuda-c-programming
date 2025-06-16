CU_APPS = simpleDeviceQuery simpleDivergence sumMatrix reduceInteger \
nestedHelloWorld nestedReduce nestedReduce2 nestedReduceNoSync
 
all: ${CU_APPS}
 
%: %.cu
	nvcc -g -G -I./include -I../tclap-1.4.0-rc1/include -I../common -diag-suppress 940 -diag-suppress 611 -diag-suppress 191 -Wno-deprecated-gpu-targets -o $@ $<
 
%: %.c
	gcc -pthread -g -o $@ $<
 
%: %.cpp
	g++ -pthread -I../common -I../tclap-1.4.0-rc1/include -fpermissive -g -o $@ $<
 
nested%: nested%.cu
	nvcc -g -G -I./include -I../common -rdc=true -lcudadevrt -o $@ $<
 
clean:
	rm -f ${CU_APPS}