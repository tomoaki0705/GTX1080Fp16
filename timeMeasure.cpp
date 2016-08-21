#include "timeMeasure.h"
#include <chrono>
#include <algorithm>

duration extractDuration(std::vector<duration>& timeDuration)
{
#if 0
	// median value
    std::sort(timeDuration.begin(), timeDuration.end());
	return timeDuration[(timeDuration.size() & (~1)) >> 1];
#else
	// sum
    return std::accumulate(timeDuration.begin(), timeDuration.end(), 0);
#endif
}


