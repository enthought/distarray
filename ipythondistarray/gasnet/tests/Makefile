include /usr/local/gasnet/include/mpi-conduit/mpi-par.mak

test_init.o: test_init.c
	$(GASNET_CC) -DGASNET_PAR -I/Developer/SDKs/MacOSX10.5.sdk/System/Library/Frameworks/Python.framework/Versions/2.5/include/python2.5 $(GASNET_CPPFLAGS) $(GASNET_CFLAGS) -c test_init.c -o test_init.o

test_init.so: test_init.o
	$(GASNET_LD) -DGASNET_PAR $(GASNET_LDFLAGS) -dynamiclib -framework Python $(GASNET_LIBS) -o test_init.so test_init.o

clean:
	rm *.o;
	rm *.so