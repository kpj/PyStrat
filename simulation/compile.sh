cd "$(dirname "$0")"
rm -f libsimulate.so

g++ -Wall -shared -fPIC -o libsim.so simulate.cpp -std=c++0x -lboost_python3 $(python3-config --cflags --ldflags)
