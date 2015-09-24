# Functions
cdef double rand_float()
cdef int rand_int(int n)
cdef int _weighted_random_int(double[:] weights)
cdef double _adhesion_ls(long l, long s, double gamma) except -1.0
cpdef set_globals(object network, object c=*, int g=*)
cpdef cluster(network, group_count=*, start_t=*, end_t=*, t_step=*, gamma=*)