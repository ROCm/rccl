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

.. doxygenfunction:: ncclCommCount

.. doxygenfunction:: ncclCommCuDevice

.. doxygenfunction:: ncclCommUserRank

Collection Communication Operations
-----------------------------------
 * Collective communication operations must be called separately for each
 * communicator in a communicator clique.
 *
 * They return when operations have been enqueued on the hip stream.
 *
Since they may perform inter-CPU synchronization, each call has to be done from a different thread or process, or need to use Group Semantics (see below).


Library Functions
-----------------

.. doxygenfunction:: ncclGetVersion

.. doxygenfunction:: ncclGetErrorString

Types
-----

There are few data structures that are internal to the library. The pointer types to these
structures are given below. The user would need to use these types to create handles and pass them
between different library functions.

.. doxygentypedef :: ncclComm_t

.. doxygenstruct:: ncclUniqueId



Enumerations
------------

This section provides all the enumerations used.

.. doxygenenum:: ncclResult_t

.. doxygenenum:: ncclRedOp_t

.. doxygenenum:: ncclDataType_t




