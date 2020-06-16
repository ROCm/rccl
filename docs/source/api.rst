.. toctree::
   :maxdepth: 4
   :caption: Contents:

===
API
===

This section provides details of the library API

Communicator Functions
----------------------

.. doxygenfunction:: ncclGetUniqueId

.. doxygenfunction:: ncclCommInitRank

.. doxygenfunction:: ncclCommInitAll

.. doxygenfunction:: ncclCommDestroy

.. doxygenfunction:: ncclCommAbort

.. doxygenfunction:: ncclCommCount

.. doxygenfunction:: ncclCommCuDevice

.. doxygenfunction:: ncclCommUserRank

Collective Communication Operations
-----------------------------------

Collective communication operations must be called separately for each communicator in a communicator clique.

They return when operations have been enqueued on the hipstream.

Since they may perform inter-CPU synchronization, each call has to be done from a different thread or process, or need to use Group Semantics (see below).

.. doxygenfunction:: ncclReduce

.. doxygenfunction:: ncclBcast

.. doxygenfunction:: ncclBroadcast

.. doxygenfunction:: ncclAllReduce

.. doxygenfunction:: ncclReduceScatter

.. doxygenfunction:: ncclAllGather

.. doxygenfunction:: ncclSend

.. doxygenfunction:: ncclRecv

.. doxygenfunction:: ncclGather

.. doxygenfunction:: ncclScatter

.. doxygenfunction:: ncclAllToAll

Group Semantics
---------------
When managing multiple GPUs from a single thread, and since NCCL collective
calls may perform inter-CPU synchronization, we need to "group" calls for
different ranks/devices into a single call.

Grouping NCCL calls as being part of the same collective operation is done
using ncclGroupStart and ncclGroupEnd. ncclGroupStart will enqueue all
collective calls until the ncclGroupEnd call, which will wait for all calls
to be complete. Note that for collective communication, ncclGroupEnd only
guarantees that the operations are enqueued on the streams, not that
the operation is effectively done.

Both collective communication and ncclCommInitRank can be used in conjunction
of ncclGroupStart/ncclGroupEnd.

.. doxygenfunction:: ncclGroupStart

.. doxygenfunction:: ncclGroupEnd

Library Functions
-----------------

.. doxygenfunction:: ncclGetVersion

.. doxygenfunction:: ncclGetErrorString

Types
-----

There are few data structures that are internal to the library. The pointer types to these
structures are given below. The user would need to use these types to create handles and pass them
between different library functions.

.. doxygentypedef:: ncclComm_t

.. doxygenstruct:: ncclUniqueId



Enumerations
------------

This section provides all the enumerations used.

.. doxygenenum:: ncclResult_t

.. doxygenenum:: ncclRedOp_t

.. doxygenenum:: ncclDataType_t




