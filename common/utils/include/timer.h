// ============================================================================
// High-Resolution Timer Utility Header
// ============================================================================
// Provides precise timing functionality for performance measurement and delays

#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <thread>

class Timer {

    public:
    // Type definitions for time measurements
    typedef std::chrono::system_clock::time_point time_point_t;
    typedef long long int time_duration_t;

    // Current time point when timer was created
    time_point_t start_time_point;

    // Constructor - initializes timer with current time
    Timer();
    
    // Get current system time
    static time_point_t getCurrentTime();

    // Calculate the duration between two time points
    // Returns duration in milliseconds
    static time_duration_t calcDiff(time_point_t begin, time_point_t end);

    // Get time passed since a specific time point
    // Returns duration in milliseconds from past time to current time
    static time_duration_t calcTimePassed(time_point_t time_point);

    // Delay execution for a specified duration
    // Duration is specified in milliseconds
    static void delay(time_duration_t duration);

};

#endif