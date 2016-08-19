#include <string>
#include <vector>

const std::string messageTime = " [us] elapsed\t";
#define timeResolution std::chrono::microseconds
typedef int duration;
duration extractDuration(std::vector<duration>& timeDuration);
