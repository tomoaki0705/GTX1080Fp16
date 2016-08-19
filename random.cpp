#include <cstdint>

const uint64_t seed = 0x1234567;
uint64_t state = seed;

unsigned RNG()
{
	state = (uint64_t)(unsigned)state* 4164903690U + (unsigned)(state >> 32);
	return (unsigned)state;
}

