// -*- coding: utf-8 -*-
// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

const { addon: ov } = require('../..');
const assert = require('assert');
const { describe, it } = require('node:test');
const { getModelPath } = require('./utils.js');

const testXml = getModelPath().xml;
const core = new ov.Core();
const properties = {
  "AUTO_BATCH_TIMEOUT": '1'
};
const compiledModel = core.compileModelSync(testXml, 'BATCH:CPU', properties);

describe('setProperty() / getProperty()', () => {

  describe('getProperty()', () => {
    it('returns the value of property from compiled model', () => {
      assert.strictEqual(compiledModel.getProperty('AUTO_BATCH_TIMEOUT'), '1');
    });
    it('throws an error when called without arguments', () => {
      assert.throws(
        () => compiledModel.getProperty(),
        /'getProperty' method called with incorrect parameters/
      );
    });
    it('throws an error when called with property name that does not exists', ()=>{
      assert.throws(
        () => compiledModel.getProperty('PROPERTY_THAT_DOES_NOT_EXIST')
      );
    });
  });

  describe('setProperty()', () => {
    it('sets a properties for compiled model', () => {
      properties["AUTO_BATCH_TIMEOUT"] = '1000';
      assert.doesNotThrow(() => compiledModel.setProperty(properties));
    });
  
    it('throws an error when called without an object argument', () => {
      assert.throws(
        () => compiledModel.setProperty(),
        /'setProperty' method called with incorrect parameters/
      );
    });
    it('throws an error when called with wrong argument', () => {
      assert.throws(
          () => compiledModel.setProperty(123),
          /'setProperty' method called with incorrect parameters/
      );
    });
  
    it('throws an error when called with multiple arguments', () => {
      assert.throws(
        () => compiledModel.setProperty({"PERFORMANCE_HINT": "THROUGHPUT"}, {"NUM_STREAMS": "AUTO"}),
        /'setProperty' method called with incorrect parameters/
      );
    });
  
    it('returns the set property of the compiled model', () => {
      properties["AUTO_BATCH_TIMEOUT"] = '123';
      compiledModel.setProperty(properties);
      assert.strictEqual(compiledModel.getProperty('AUTO_BATCH_TIMEOUT'), 123);
    });
  
    it('retains the last set property when set multiple times', () => {
      compiledModel.setProperty({"AUTO_BATCH_TIMEOUT": '321'});
      compiledModel.setProperty({'AUTO_BATCH_TIMEOUT': '132'});
      assert.strictEqual(compiledModel.getProperty('AUTO_BATCH_TIMEOUT'), 132);
    });
  
    it('allows to pass empty object', () => {
      assert.doesNotThrow(() => compiledModel.setProperty({}));
    });
  });
});