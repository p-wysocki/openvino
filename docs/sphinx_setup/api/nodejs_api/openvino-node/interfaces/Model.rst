Interface Model
===============

.. code-block:: ts

   interface Model {
       inputs: Output[];
       outputs: Output[];
       getFriendlyName(): string;
       getName(): string;
       getOutputShape(index): number[];
       getOutputSize(): number;
       input(): Output;
       input(name): Output;
       input(index): Output;
       isDynamic(): boolean;
       output(): Output;
       output(name): Output;
       output(index): Output;
       setFriendlyName(name): void;
   }

A user-defined model read by :ref:`Core.readModel <readModel>`.

* **Defined in:**
  `addon.ts:191 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L191>`__


Properties
#####################


.. rubric:: inputs

.. container:: m-4

   .. code-block:: ts

      inputs: Output[]

   -  **Defined in:**
      `addon.ts:193 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L193>`__

.. rubric:: outputs


.. container:: m-4

   .. code-block:: ts

      outputs: Output[]

   -  **Defined in:**
      `addon.ts:192 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L192>`__


Methods
#####################


.. rubric:: getFriendlyName

.. container:: m-4

   .. code-block:: ts

      getFriendlyName(): string

   It gets the friendly name for a model. If a friendly name is not set
   via :ref:`Model.setFriendlyName <setFriendlyName>`, a unique model name is returned.

   * **Returns:** string

     A string with a friendly name of the model.

   * **Defined in:**
     `addon.ts:200 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L200>`__


.. rubric:: getName

.. container:: m-4

   .. code-block:: ts

      getName(): string

   It gets the unique name of the model.

   * **Returns:** string

     A string with the name of the model.

   * **Defined in:**
     `addon.ts:196 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L196>`__


.. rubric:: getOutputShape

.. container:: m-4

   .. code-block:: ts

      getOutputShape(): number[]

   It returns the shape of the element at the specified index.

   * **Returns:** number[]

   * **Defined in:**
     `addon.ts:201 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L201>`__


.. rubric:: getOutputSize

.. container:: m-4

   .. code-block:: ts

      getOutputSize(): number[]

   It returns the number of the model outputs.

   * **Returns:** number[]

   * **Defined in:**
     `addon.ts:198 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L198>`__


.. rubric:: input

.. container:: m-4

   .. code-block:: ts

      input(): Output

   It gets the input of the model. If a model has more than one input,
   this method throws an exception.

   * **Returns:**  :doc:`Output <Output>`

   * **Defined in:**
     `addon.ts:219 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L219>`__


   .. code-block:: ts

      input(name: string): Output

   It gets the input of the model identified by the tensor name.

   * **Parameters:**

     - ``Optional``

       .. code-block:: ts

          name: string

       The tensor name.

   * **Returns:**  :doc:`Output <Output>`

   * **Defined in:**
     `addon.ts:224 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L224>`__


   .. code-block:: ts

      input(index: number): Output

   It gets the input of the model identified by the index.

   * **Parameters:**

     - ``Optional``

       .. code-block:: ts

          index: number

       The index of the input.

   * **Returns:**  :doc:`Output <Output>`

   * **Defined in:**
     `addon.ts:229 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L229>`__


.. rubric:: isDynamic

.. container:: m-4

   .. code-block:: ts

      isDynamic(): boolean

   It returns true if any of the ops defined in the model contains a partial shape.

   * **Returns:**  boolean

   * **Defined in:**
     `addon.ts:234 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L234>`__


.. rubric:: output

.. container:: m-4

   .. code-block:: ts

      output(nameOrId?): Output

   * **Parameters:**

     - ``Optional``

       .. code-block:: ts

          nameOrId: string|number

   * **Returns:**  :doc:`Output <Output>`

   * **Defined in:**
     `addon.ts:194 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L194>`__


.. rubric:: setFriendlyName
   :name: setFriendlyName

.. container:: m-4

   .. code-block:: ts

      setFriendlyName(name): void

   * **Parameters:**

     - name: string

   * **Returns:** void

   * **Defined in:**
     `addon.ts:199 <https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/js/node/lib/addon.ts#L199>`__