all: nms.hpp
	g++ -I /home/patrick/Downloads/eigen-3.3.7/ dec.cpp vectorized_nms.cpp nms.cpp utils.cpp example.cpp `pkg-config opencv --cflags` `pkg-config opencv --libs` -std=c++14 -O2 -DNDEBUG -o example

decoder:
	g++ -I /home/patrick/Downloads/eigen-3.3.7/ nms.cpp utils.cpp decoder.cpp `pkg-config opencv --cflags` `pkg-config opencv --libs` -std=c++14 -O2 -DNDEBUG -o decoder
run:
	./example

clean:
	rm -rf example
	rm -rf decoder
