Interface Tensor
=====================

.. code-block:: ts

   interface Tensor {
       data: SupportedTypedArray;
       getData(): SupportedTypedArray;
       getElementType(): element;
       getShape(): number[];
       getSize(): number;

   }

The ``Tensor`` is a lightweight class that represents data used for
inference. There are different ways to create a tensor. You can find them
in :doc:`TensorConstructor <TensorConstructor>` section.

* **Defined in:**
  `addon.ts:265 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L265>`__


Properties
#####################


.. rubric:: data

.. container:: m-4

   .. code-block:: ts

      data: SupportedTypedArray

   This property provides access to the tensor's data.

   Its getter returns a subclass of TypedArray that corresponds to the
   tensor element type, e.g. ``Float32Array`` corresponds to ``float32``. The
   content of the ``TypedArray`` subclass is a copy of the tensor underlaying
   memory.

   Its setter fills the underlaying tensor memory by copying the binary data
   buffer from the ``TypedArray`` subclass. An exception will be thrown if the size
   or type of array does not match the tensor.

   -  **Defined in:**
      `addon.ts:89 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L89>`__


Methods
#####################


.. rubric:: getData

.. container:: m-4

   .. code-block:: ts

      getData(): SupportedTypedArray;

   It gets tensor data.

   * **Returns:** SupportedTypedArray

     A subclass of ``TypedArray`` corresponding to the tensor
     element type, e.g. ``Float32Array`` corresponds to float32.

   * **Defined in:**
     `addon.ts:356 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L356>`__

.. rubric:: getElementType

.. container:: m-4

   .. code-block:: ts

      getElementType(): element

   It gets the tensor element type.

   * **Returns:** :doc:`element <../enums/element>`

   * **Defined in:**
     `addon.ts:350 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L350>`__


.. rubric:: getShape

.. container:: m-4

   .. code-block:: ts

      getShape(): number[]

   It gets the tensor shape.

   * **Returns:** number[]

   * **Defined in:**
     `addon.ts:360 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L360>`__


.. rubric:: getSize

.. container:: m-4

   .. code-block:: ts

      getSize(): number[]

   It gets the tensor size as a total number of elements.

   * **Returns:** number[]

   * **Defined in:**
     `addon.ts:364 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L364>`__

