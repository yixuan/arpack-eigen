HEADERS = $(wildcard ../include/*.h)

all: QR.out SymEigs.out SymEigsShift.out

QR.out: $(HEADERS) QR.cpp
	g++ -O2 -I../include QR.cpp -o QR.out

SymEigs.out: $(HEADERS) SymEigs.cpp
	g++ -O2 -I../include SymEigs.cpp -o SymEigs.out

SymEigsShift.out: $(HEADERS) SymEigsShift.cpp
	g++ -O2 -I../include SymEigsShift.cpp -o SymEigsShift.out

clean:
	rm *.out
