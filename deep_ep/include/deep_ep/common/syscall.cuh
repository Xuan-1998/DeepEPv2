#pragma once

#include <sys/syscall.h>

#ifndef SYS_pidfd_open
#ifdef __NR_pidfd_open
#define SYS_pidfd_open __NR_pidfd_open
#elif defined(__x86_64__) || defined(__aarch64__)
#define SYS_pidfd_open 434
#else
#error "pidfd_open syscall number is unavailable on this architecture"
#endif
#endif

#ifndef SYS_pidfd_getfd
#ifdef __NR_pidfd_getfd
#define SYS_pidfd_getfd __NR_pidfd_getfd
#elif defined(__x86_64__) || defined(__aarch64__)
#define SYS_pidfd_getfd 438
#else
#error "pidfd_getfd syscall number is unavailable on this architecture"
#endif
#endif
