all: nms.hpp
	g++ nms.cpp utils.cpp example.cpp `pkg-config opencv --cflags` `pkg-config opencv --libs` -std=c++14 -o example

decoder:
	g++ decoder.cpp `pkg-config opencv --cflags` `pkg-config opencv --libs` -std=c++14 -o decoder


run:
	./example

clean:
	rm -rf example
	rm -rf decoder
