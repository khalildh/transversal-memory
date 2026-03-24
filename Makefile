CC = cc
CFLAGS = -O3 -march=native -fPIC -shared
LDFLAGS = -lm

arc_tables.so: arc_tables.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f arc_tables.so

test: arc_tables.so
	uv run python test_arc_tables.py

.PHONY: clean test
