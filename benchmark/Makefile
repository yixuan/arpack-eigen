all: benchmark

benchmark: main.cpp F77.o Cpp.o wrapper.o
	g++ -O2 -I../include main.cpp F77.o Cpp.o wrapper.o -o benchmark -larpack -llapack -lblas -lgfortran

F77.o: F77.cpp
	g++ -O2 -I../include -I. -c F77.cpp

wrapper.o: wrapper.f
	gfortran -O3 -c wrapper.f

Cpp.o: Cpp.cpp
	g++ -O2 -mavx -I../include -c Cpp.cpp

clean:
	rm benchmark *.o
