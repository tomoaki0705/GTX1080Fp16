
unsigned RNG();

template<typename T> void fillRandomNumber(T* array, int cElement);

template<> void fillRandomNumber<float>(float* array, int cElement)
{
	for(unsigned int i = 0;i < cElement;i++)
	{
		short random = (short)(RNG() & 0x3fff);
		float exp = (float)(random - 0x2000) / (float)(0x1000);
		array[i] = (float)pow(2, exp);
	}
}

template<> void fillRandomNumber<short>(short* array, int cElement)
{
	for(unsigned int i = 0;i < cElement;i++)
	{
		short random = (short)(RNG() & 0x3fff);
		float exp = (float)(random - 0x2000) / (float)(0x1000);
		float var[4] = {(float)pow(2, exp), 0, 0, 0,};
		__m128 vF = _mm_load_ps(var);
		__m128i vH = _mm_cvtps_ph(vF, 0);
		short half[4];
		_mm_storel_epi64((__m128i*)half, vH);
		array[i] = half[0];
	}
}
