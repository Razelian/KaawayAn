CC ?= gcc
CFLAGS ?= -Wall -Wextra -O2 -I../components/kaawayan_proto/include -I../components/bme280/include

all: test

test_proto: test_proto.c ../components/kaawayan_proto/crc32.c ../components/kaawayan_proto/frame.c ../components/kaawayan_proto/ramp.c
	$(CC) $(CFLAGS) -o $@ $^

test_bme280: test_bme280.c ../components/bme280/bme280.c
	$(CC) $(CFLAGS) -DBME280_HOST_TEST -o $@ $^

test: test_proto test_bme280
	./test_proto
	./test_bme280

clean:
	rm -f test_proto test_proto.exe test_bme280 test_bme280.exe

.PHONY: all test clean
