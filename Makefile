all: nms.hpp
	g++ -I /home/patrick/Downloads/eigen-3.3.7/ decoder.cpp vectorized_nms.cpp nms.cpp utils.cpp nms_tester.cpp `pkg-config opencv --cflags` `pkg-config opencv --libs` -std=c++14 -O2 -DNDEBUG -o nms_tester
	g++ -I /home/patrick/Downloads/eigen-3.3.7/ decoder.cpp vectorized_nms.cpp nms.cpp utils.cpp vec_tester.cpp `pkg-config opencv --cflags` `pkg-config opencv --libs` -std=c++14 -O2 -DNDEBUG -o vec_tester
decoder:
	g++ -I /home/patrick/Downloads/eigen-3.3.7/ decode_detections.cpp vec_tester.cpp -std=c++14 -O2 -DNDEBUG -o vec_tester

run:
	./example

clean:
	rm -rf nms_tester
	rm -rf decoder
	rm -rf vec_tester


