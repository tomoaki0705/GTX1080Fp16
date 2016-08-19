#include <iostream>
//#include <vector_types.h>
#include <chrono>
//#include "types.h"
//#include <cuda_runtime.h>
#include <vector>
//#include <algorithm>
//#include <cmath>
//#include <immintrin.h>
//#include <iomanip>
#include "random.h"
#include "timeMeasure.h"

void convertArray(int imgW, int imgH, int cLoop)
{
	float* src;
	short* dst; 
	int s = imgW*imgH;

	src = (float*)malloc(s*sizeof(float));
	dst = (short*)malloc(s*sizeof(short));

	std::vector<duration> timeDuration;
	for(int iLoop = 0;iLoop < cLoop;iLoop++)
	{
		fillRandomNumber<float>(src, s);
		auto start = std::chrono::system_clock::now();
		for(int i = 0;i <= s-4;i+=4)
		{
			__m128  vF = _mm_load_ps(src + i);
			__m128i vH = _mm_cvtps_ph(vF, 0);
			_mm_storel_epi64((__m128i*)(dst + i), vH);
		}
		auto end   = std::chrono::system_clock::now();
		duration usec = std::chrono::duration_cast<timeResolution>(end - start).count();
		timeDuration.push_back(usec);
	}

	std::cout << extractDuration(timeDuration) << messageTime << s << "\tpixels (" << imgW << 'x' << imgH << ')' << std::endl;
}

int main()
{
	convertArray(640,  480,  100);
	convertArray(1920, 1080, 100);
	convertArray(3840, 2160, 100);
	convertArray(4000, 4000, 100);
	convertArray(8000, 8000, 100);

	return 0;
}
