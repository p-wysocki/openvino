Interface InputTensorInfo
=========================

.. code-block:: ts

   interface InputTensorInfo {
       setElementType(elementType): InputTensorInfo;
       setLayout(layout): InputTensorInfo;
       setShape(shape): InputTensorInfo;
   }

* **Defined in:**
  `addon.ts:524 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L524>`__


Methods
#####################


.. rubric:: setElementType

.. container:: m-4

   .. code-block:: ts

      setElementType(elementType): InputTensorInfo

   * **Parameters:**

     - elementType: :doc:`elementTypeString <../types/elementTypeString>` | :doc:`element <../enums/element>`

   * **Returns:** :doc:`InputTensorInfo <InputTensorInfo>`

   * **Defined in:**
     `addon.ts:525 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L525>`__


.. rubric:: setLayout

.. container:: m-4

   .. code-block:: ts

      setLayout(layout): InputTensorInfo

   * **Parameters:**

     - layout: string

   * **Returns:** :doc:`InputTensorInfo <InputTensorInfo>`

   * **Defined in:**
     `addon.ts:526 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L526>`__


.. rubric:: setShape

.. container:: m-4

   .. code-block:: ts

      setShape(shape): InputTensorInfo

   * **Parameters:**

     - shape: number[]

   * **Returns:** :doc:`InputTensorInfo <InputTensorInfo>`

   * **Defined in:**
     `addon.ts:527 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L527>`__

