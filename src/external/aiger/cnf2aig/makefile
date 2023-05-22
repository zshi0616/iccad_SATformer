cnf2aig: cnf2aig.c makefile ../aiger/aiger.o
	gcc -Wall -O3 -DNDEBUG -o $@ cnf2aig.c ../aiger/aiger.o
clean:
	rm -f cnf2aig makefile
	rm -f log/*.err log/*.log
