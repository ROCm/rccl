.. toctree::
   :maxdepth: 4 
   :caption: Contents:

===
API
===

This section provides details of the library API

Types
-----

There are few data structures that are internal to the library. The pointer types to these
structures are given below. The user would need to use these types to create handles and pass them
between different library functions.

.. doxygentypedef:: rocfft_plan

.. doxygentypedef:: rocfft_plan_description

.. doxygentypedef:: rocfft_execution_info

Library Setup and Cleanup
-------------------------

The following functions deals with initialization and cleanup of the library.

.. doxygenfunction:: rocfft_setup

.. doxygenfunction:: rocfft_cleanup

Plan
----

The following functions are used to create and destroy plan objects.

.. doxygenfunction:: rocfft_plan_create

.. doxygenfunction:: rocfft_plan_destroy

The following functions are used to query for information after a plan is created.

.. doxygenfunction:: rocfft_plan_get_work_buffer_size

.. doxygenfunction:: rocfft_plan_get_print

Plan description
----------------

Most of the times, :cpp:func:`rocfft_plan_create` is all is needed to fully specify a transform.
And the description object can be skipped. But when a transform specification has more details
a description object need to be created and set up and the handle passed to the :cpp:func:`rocfft_plan_create`.
Functions referred below can be used to manage plan description in order to specify more transform details.
The plan description object can be safely deleted after call to the plan api :cpp:func:`rocfft_plan_create`.

.. doxygenfunction:: rocfft_plan_description_create

.. doxygenfunction:: rocfft_plan_description_destroy

.. comment  doxygenfunction:: rocfft_plan_description_set_scale_float

.. comment doxygenfunction:: rocfft_plan_description_set_scale_double

.. doxygenfunction:: rocfft_plan_description_set_data_layout

.. comment doxygenfunction:: rocfft_plan_description_set_devices

Execution
---------

The following details the execution function. After a plan has been created, it can be used
to compute a transform on specified data. Aspects of the execution can be controlled and any useful
information returned to the user.

.. doxygenfunction:: rocfft_execute

Execution info
--------------

The execution api :cpp:func:`rocfft_execute` takes a rocfft_execution_info parameter. This parameter needs
to be created and setup by the user and passed to the execution api. The execution info handle encapsulates
information such as execution mode, pointer to any work buffer etc. It can also hold information that are 
side effect of execution such as event objects. The following functions deal with managing execution info
object. Note that the *set* functions below need to be called before execution and *get* functions after
execution.

.. doxygenfunction:: rocfft_execution_info_create

.. doxygenfunction:: rocfft_execution_info_destroy

.. doxygenfunction:: rocfft_execution_info_set_work_buffer

.. comment doxygenfunction:: rocfft_execution_info_set_mode

.. doxygenfunction:: rocfft_execution_info_set_stream

.. comment doxygenfunction:: rocfft_execution_info_get_events


Enumerations
------------

This section provides all the enumerations used.

.. doxygenenum:: rocfft_status

.. doxygenenum:: rocfft_transform_type

.. doxygenenum:: rocfft_precision

.. doxygenenum:: rocfft_result_placement

.. doxygenenum:: rocfft_array_type

.. doxygenenum:: rocfft_execution_mode




