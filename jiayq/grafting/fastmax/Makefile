CC = g++
CCFLAGS = -fPIC -O3 -Wall -pedantic -ansi -msse -msse2
LINKFLAGS = -shared -Wl
all:
	$(CC) -c $(CCFLAGS) *.cpp
	$(CC) $(LINKFLAGS) -o libfastmax.so *.o
clean:
	rm *.so
	rm *.o
