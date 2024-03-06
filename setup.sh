cd src/external/PyMiniSolvers
make

cd ../aiger/aiger
# may need to change the complier in makefile to clang for macos
./configure.sh && make
cd ../cnf2aig
./configure && make

cd ../../..
