import atexit
from time import clock


def secondsToStr(t):
    return "%d:%02d:%02d.%03d" % \
        reduce(lambda ll,b : divmod(ll[0],b) + ll[1:],[(t*1000,),1000,60,60])


def log(s, elapsed=None):
    line = "=" * 40
    print(line)
    print("   %s %s %s" % (secondsToStr(clock()), '-', s))
    if elapsed:
        print("   Elapsed time: %s" % elapsed)
    print(line+'\n')


def endlog():
    end = clock()
    elapsed = end-start
    log("End Program", secondsToStr(elapsed))


def now():
    return secondsToStr(clock())


start = clock()
atexit.register(endlog)
log("Start Program")