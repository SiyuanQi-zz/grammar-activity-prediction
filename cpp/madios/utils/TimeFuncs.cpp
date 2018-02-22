#include "TimeFuncs.h"

#if defined _WIN32 || defined _WIN64
#include <windows.h>
double getTime()
{
    static double     queryPerfSecsPerCount = 0.0;
    LARGE_INTEGER     queryPerfCount;
    double            seconds;
    BOOL              success;

    if ( queryPerfSecsPerCount == 0.0 )
    {
        LARGE_INTEGER queryPerfCountsPerSec;

        // get ticks-per-second ratio, calc inverse ratio
        success = QueryPerformanceFrequency( &queryPerfCountsPerSec );
        if ( success && queryPerfCountsPerSec.QuadPart )
            queryPerfSecsPerCount = (double) 1.0 / (double) queryPerfCountsPerSec.QuadPart;
        else// failure (oh joy, we are running on Win9x)
            queryPerfSecsPerCount = -1.0;
    }

    if ( queryPerfSecsPerCount == -1.0 )
        seconds = 0.001 * (double) GetTickCount();// GetTickCount() is less accurate, but it is our only choice
    else
    {
        QueryPerformanceCounter( &queryPerfCount );
        seconds = queryPerfSecsPerCount * (double) queryPerfCount.QuadPart;
    }

    return seconds;
}

unsigned int getSeedFromTime()
{
    return GetTickCount();
}

#else
#include <sys/time.h>
double getTime()
{
    struct timeval tv;
    struct timezone tz;
    gettimeofday(&tv, &tz);

    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

unsigned int getSeedFromTime()
{
    struct timeval tv;
    struct timezone tz;
    gettimeofday(&tv, &tz);

    return tv.tv_usec;
}
#endif
