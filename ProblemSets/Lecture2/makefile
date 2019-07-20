#####################################
all: BS.exec
#####################################


# european option, B-S model
BS.exec: BS.cpp
	g++ BS.cpp -o BS.exec -fopenmp


## clean up
clean:
	rm -rf *.exec
