#include "timeMeasure.h"
#include <chrono>
#include <algorithm>

duration extractDuration(std::vector<duration>& timeDuration)
{
    return std::accumulate(timeDuration.begin(), timeDuration.end(), 0);
}


