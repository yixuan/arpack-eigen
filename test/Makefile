all: QR SymEigs

QR: QR.cpp
	g++ -O2 -I../include QR.cpp -o QR

SymEigs: SymEigs.cpp
	g++ -O2 -I../include SymEigs.cpp -o SymEigs

clean:
	rm QR SymEigs
