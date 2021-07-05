metric.dll: metric.cpp
	g++ -O3 --share -fopenmp metric.cpp -o metric.dll

clean:
	del metric.dll