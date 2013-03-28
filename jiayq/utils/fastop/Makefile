CC = g++
CCFLAGS = -fPIC -O3 -Wall -pedantic -ansi -msse -msse2 -ffast-math -msse3
LINKFLAGS = -shared -Wl
TARGET = libfastop.so
all:
	$(CC) -c $(CCFLAGS) *.cpp
	$(CC) $(LINKFLAGS) -o $(TARGET) *.o
clean:
	rm *.so
	rm *.o
