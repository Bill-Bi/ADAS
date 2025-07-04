#include "timer.h"
#include <iostream> 

// Constructor - initialize timer with current time
Timer::Timer() {
    start_time_point = getCurrentTime();
}

// Get current system time using high-resolution clock
Timer::time_point_t Timer::getCurrentTime() {
    return std::chrono::system_clock::now();
}

// Calculate the duration between two time points
// Returns duration in milliseconds
Timer::time_duration_t Timer::calcDiff(time_point_t begin, time_point_t end) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
}

// Get time passed since a specific time point
// Returns duration in milliseconds from past time to current time
Timer::time_duration_t Timer::calcTimePassed(time_point_t time_point) {
    return calcDiff(time_point, getCurrentTime());
}

// Delay execution for a specified duration
// Duration is specified in milliseconds
void Timer::delay(time_duration_t duration) {
    std::this_thread::sleep_for(std::chrono::milliseconds(duration));
}