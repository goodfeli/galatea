CC = gcc
CCFLAGS = -fPIC -O3 -Wall -pedantic -ansi -msse -msse2 -ffast-math -mfpmath=sse
LINKFLAGS = -shared -Wl
all:
	$(CC) -c $(CCFLAGS) *.c
	#$(CC) $(LINKFLAGS) -o libloss.so *.o amdlibm/lib/static/libamdlibm.a
	$(CC) $(LINKFLAGS) -o libloss.so *.o
clean:
	rm *.so
	rm *.o
