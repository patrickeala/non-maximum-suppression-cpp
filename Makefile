all: nms.hpp
	g++ -I /home/patrick/Downloads/eigen-3.3.7/ nms.cpp utils.cpp example.cpp `pkg-config opencv --cflags` `pkg-config opencv --libs` -std=c++14 -o example

decoder:
	g++ -I /home/patrick/Downloads/eigen-3.3.7/ decoder.cpp `pkg-config opencv --cflags` `pkg-config opencv --libs` -std=c++14 -o decoder

run:
	./example

clean:
	rm -rf example
	rm -rf decoder
