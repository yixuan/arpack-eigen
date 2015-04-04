HEADERS = $(wildcard ../include/*.h)

all: QR SymEigs

QR: $(HEADERS) QR.cpp
	g++ -O2 -I../include QR.cpp -o QR

SymEigs: $(HEADERS) SymEigs.cpp
	g++ -O2 -I../include SymEigs.cpp -o SymEigs

clean:
	rm QR SymEigs
