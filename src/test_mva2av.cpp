#include <numeric>
#include <tuple>
#include <vector>
#include <limits>
#include <cassert>
#include <iostream>

#ifdef USE_MXX
#include "mxx/comm.hpp"
#include "mxx/collective.hpp"
#include "mxx/distribution.hpp"
#endif

#include <mpi.h>

void myMPIErrorHandler(MPI_Comm*, int* ...) {
  // throw exception, enables gdb stack trace analysis
  throw std::runtime_error("MPI Error");
}

using TestType = std::pair<double, float>;

MPI_Datatype get_mpi_datatype() {
  MPI_Datatype _type;

  int blocklen[2] = {1, 1};
  MPI_Aint displs[2] = {0,0};
  // get actual displacement (in case of padding in the structure)
  TestType p;
  MPI_Aint p_adr, t1_adr, t2_adr;
  MPI_Get_address(&p, &p_adr);
  MPI_Get_address(&p.first, &t1_adr);
  MPI_Get_address(&p.second, &t2_adr);
  displs[0] = t1_adr - p_adr;
  displs[1] = t2_adr - p_adr;

  MPI_Datatype types[2] = {MPI_DOUBLE, MPI_FLOAT};
  // in case elements are represented the opposite way around in
  // the pair (gcc does so), then swap them
  if (displs[0] > displs[1]) {
      std::swap(displs[0], displs[1]);
      std::swap(types[0], types[1]);
  }
  // create MPI_Datatype (resized to actual sizeof())
  MPI_Datatype struct_type;
  MPI_Type_create_struct(2, blocklen, displs, types, &struct_type);
  MPI_Type_create_resized(struct_type, 0, sizeof(p), &_type);
  MPI_Type_commit(&_type);
  MPI_Type_free(&struct_type);
  return _type;
}

#ifdef USE_MXX
int mxx_test(int argc, char** argv, 
             std::vector<size_t>& msgSizes,
             std::vector<size_t>& outSizes){
  MPI_Init(&argc, &argv);
  mxx::comm comm;
  // Set custom error handler (for debugging with working stack-trace on gdb)
  MPI_Errhandler handler;
  MPI_Errhandler_create(&myMPIErrorHandler, &handler);
  MPI_Errhandler_set(comm, handler);

  std::vector<TestType> msg(msgSizes.at(comm.rank()));
  std::vector<TestType> out(outSizes.at(comm.rank()));
  mxx::stable_distribute(msg.begin(), msg.end(), out.begin(), comm);
  return MPI_Finalize();
}
#endif


// rough interface for distribution of data elements over the processors of a
// communicator
class dist_base {
protected:
    unsigned int m_comm_size, m_comm_rank;
    size_t m_global_size;

    dist_base() : m_comm_size(0), m_comm_rank(0), m_global_size(0) {}

    dist_base(size_t global_size, int comm_size, int comm_rank) :
        m_comm_size(comm_size), m_comm_rank(comm_rank), m_global_size(global_size) {
    }

public:
    inline size_t global_size() const {
        return m_global_size;
    }

    inline int comm_size() const {
        return m_comm_size;
    }

    inline int comm_rank() const {
        return m_comm_rank;
    }

    /* need to be implemented by all deriving classes
    // simple size and prefix queries
    inline size_t local_size() const;
    inline size_t local_size(int rank) const;
    inline size_t global_size() const;
    inline size_t eprefix_size() const;
    inline size_t iprefix_size() const;
    inline size_t eprefix_size(int rank) const;
    inline size_t iprefix_size(int rank) const;

    // rank and gidx translations
    inline int    rank_of(size_t gidx) const;
    inline size_t lidx_of(size_t gidx) const;
    inline size_t gidx_of(int rank, size_t lidx) const;
    */
};

/*
 * TODO:
 * move simple bare queries to easier base class:
 * local_size(), global_size(), eprefix_size(), iprefix_size()
 */


class blk_dist_buf : public dist_base {
public:
    using dist_base::global_size;

    ~blk_dist_buf () {}

    /// collective allreduce for global size
    /*
    blk_dist_buf(const mxx::comm& comm, size_t local_size)
        : dist_base(comm, local_size),
          n(mxx::allreduce(local_size, comm)),
          div(n / m_comm_size), mod(n % m_comm_size),
          prefix(div*m_comm_rank + std::min<size_t>(mod, m_comm_rank)),
          div1mod((div+1)*mod)
    {
        assert(local_size == div + (m_comm_rank < mod ? 1 : 0));
    }
    */

    blk_dist_buf() = default;

    blk_dist_buf(size_t global_size, unsigned int comm_size, unsigned int comm_rank)
        : dist_base(global_size, comm_size, comm_rank),
          div(global_size / comm_size), mod(global_size % comm_size),
          m_local_size(div + (comm_rank < mod ? 1 : 0)),
          m_prefix(div*m_comm_rank + std::min<size_t>(mod, m_comm_rank)),
          div1mod((div+1)*mod) {
    }

    // default copy and move
    blk_dist_buf(blk_dist_buf&&) = default;
    blk_dist_buf(const blk_dist_buf&) = default;
    blk_dist_buf& operator=(const blk_dist_buf&) = default;
    blk_dist_buf& operator=(blk_dist_buf&&) = default;

    inline size_t local_size() const {
        return m_local_size;
    }

    inline size_t local_size(unsigned int rank) const {
        return div + (rank < mod ? 1 : 0);
    }

    inline size_t iprefix_size() const {
        return m_prefix + m_local_size;
    }

    inline size_t iprefix_size(unsigned int rank) const {
        return div*(rank+1) + std::min<size_t>(mod, rank + 1);
    }

    inline size_t eprefix_size() const {
        return m_prefix;
    }

    inline size_t eprefix_size(unsigned int rank) const {
        return div*rank + std::min<size_t>(mod, rank);
    }

    // which processor the element with the given global index belongs to
    inline int rank_of(size_t gidx) const {
        if (gidx < div1mod) {
            // a_i is within the first n % p processors
            return gidx/(div+1);
        } else {
            return mod + (gidx - div1mod)/div;
        }
    }

    inline size_t lidx_of(size_t gidx) const {
        return gidx - eprefix_size(rank_of(gidx));
    }

    inline size_t gidx_of(int rank, size_t lidx) const {
        return eprefix_size(rank) + lidx;
    }

private:
    /* data */
    // derived/buffered values (for faster computation of results)
    size_t div; // = n/p
    size_t mod; // = n%p
    // local size (number of local elements)
    size_t m_local_size;
    // the exclusive prefix (number of elements on previous processors)
    size_t m_prefix;
    /// number of elements on processors with one more element
    size_t div1mod; // = (n/p + 1)*(n % p)
};


using blk_dist = blk_dist_buf;

template <typename index_t = int>
std::vector<index_t> get_displacements(const std::vector<index_t>& counts)
{
    // copy and do an exclusive prefix sum
    std::vector<index_t> result(counts);
    // set the total sum to zero
    index_t sum = 0;
    index_t tmp;

    // calculate the exclusive prefix sum
    typename std::vector<index_t>::iterator begin = result.begin();
    while (begin != result.end())
    {
        tmp = sum;
        // assert that the sum will still fit into the index type (MPI default:
        // int)
        assert((std::size_t)sum + (std::size_t)*begin < (std::size_t) std::numeric_limits<index_t>::max());
        sum += *begin;
        *begin = tmp;
        ++begin;
    }
    return result;
}


int mva2av_test(int argc, char** argv,
                std::vector<size_t>& msgSizes,
                std::vector<size_t>& outSizes){
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Errhandler handler;
    MPI_Errhandler_create(&myMPIErrorHandler, &handler);
    MPI_Errhandler_set(comm, handler);

    // Get the number of processes
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<TestType> msg(msgSizes.at(rank));
    std::vector<TestType> out(outSizes.at(rank));

    // get local and global size
    int local_size = (int)msg.size();
    int total_size = 0;
    MPI_Allreduce(&local_size, &total_size, 1, MPI_INT, MPI_SUM, comm);

    // get prefix sum of size and total size
    int prefix;
    MPI_Exscan(&local_size, &prefix, 1, MPI_INT, MPI_SUM, comm);
    if (rank == 0)
        prefix = int();
    // calculate where to send elements
    // if there are any elements to send
    std::vector<int> send_counts(size, 0);
    if (local_size > 0) {
      blk_dist part(total_size, size, rank);
      int first_p = part.rank_of(prefix);
      int left_to_send = local_size;
      for (; left_to_send > 0 && first_p < size; ++first_p) {
          int nsend = std::min<int>(part.iprefix_size(first_p) - prefix, left_to_send);
          send_counts[first_p] = nsend;
          left_to_send -= nsend;
          prefix += nsend;
      }
    }

    std::vector<int> recv_counts(size, 0);
    MPI_Alltoall(const_cast<int*>(&send_counts[0]), 1, MPI_INT,
                 &recv_counts[0], 1, MPI_INT, comm);

    // get displacements
    std::vector<int> send_displs = get_displacements(send_counts);
    std::vector<int> recv_displs = get_displacements(recv_counts);
    if (rank == 0){
       std::cout << "Start All2All" << std::endl;
    }
    MPI_Datatype dt = get_mpi_datatype();
    MPI_Alltoallv(const_cast<TestType*>(&msg[0]), &send_counts[0], &send_displs[0], dt,
                  &out[0], &recv_counts[0], &recv_displs[0], dt, comm);

    if (rank == 0){
       std::cout << "Finish All2All" << std::endl;
    }
    return MPI_Finalize();
}
int main(int argc, char** argv) {

  std::vector<size_t> msgSizes{ 62810, 92785, 101355, 102780, 62810, 57100, 68520, 51390,
                                34260, 57100, 68520, 79940, 91360, 45680, 70660, 112059,
                                102780, 45680, 79940, 74230, 91360, 78515, 58526, 45680,
                                68520, 97070, 124911, 74940, 97070, 114200, 74230, 62810 };
  std::vector<size_t> outSizes{ 76550, 76550, 76550, 76550, 76550, 76550, 76550, 76550,
                                76550, 76550, 76550, 76550, 76550, 76550, 76550, 76550,
                                76550, 76550, 76550, 76550, 76550, 76550, 76550, 76549,
                                76549, 76549, 76549, 76549, 76549, 76549, 76549, 76549 };

#ifdef USE_MXX
    return mxx_test(argc, argv, msgSize, outSizes);
#endif
    return mva2av_test(argc, argv, msgSizes, outSizes);

}

