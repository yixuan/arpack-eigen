CXX = g++
CXXFLAGS = -std=c++11 -Wall -O2
CPPFLAGS = -I../include
LDFLAGS =
LIBS =

HEADERS = $(wildcard ../include/MatOp/*.h) $(wildcard ../include/*.h)


.PHONY: all test clean

all: QR.out SymEigs.out SymEigsShift.out GenEigs.out GenEigsRealShift.out

test:
	-./QR.out
	-./SymEigs.out
	-./SymEigsShift.out
	-./GenEigs.out
	-./GenEigsRealShift.out

%.out: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -o $@ $(LDFLAGS) $(LIBS)

clean:
	-rm *.out
