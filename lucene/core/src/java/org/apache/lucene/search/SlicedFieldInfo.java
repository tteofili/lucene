/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.lucene.search;

import java.util.Objects;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;

/**
 * FieldInfo wrapper that reports sliced vector dimensions.
 * 
 * <p>This wrapper provides dimension translation information to codec readers
 * while preserving all original field metadata through delegation.
 *
 * @lucene.experimental
 */
class SlicedFieldInfo {
  
  private final FieldInfo delegate;
  private final VectorSlice vectorSlice;
  private final int slicedDimension;
  
  /**
   * Creates a new SlicedFieldInfo.
   * 
   * @param delegate original field info
   * @param vectorSlice vector slice to apply
   */
  public SlicedFieldInfo(FieldInfo delegate, VectorSlice vectorSlice) {
    this.delegate = Objects.requireNonNull(delegate, "delegate");
    this.vectorSlice = Objects.requireNonNull(vectorSlice, "vectorSlice");
    this.slicedDimension = vectorSlice.getSliceLength(delegate.getVectorDimension());
  }
  
  /**
   * Returns the effective (sliced) vector dimension.
   */
  public int getVectorDimension() {
    return slicedDimension;
  }
  
  /**
   * Returns the original vector dimension before slicing.
   */
  public int getOriginalVectorDimension() {
    return delegate.getVectorDimension();
  }
  
  /**
   * Returns the vector slice.
   */
  public VectorSlice getVectorSlice() {
    return vectorSlice;
  }
  
  /**
   * Returns true if this field uses dimension slicing.
   */
  public boolean isSliced() {
    return !vectorSlice.isNoSlice();
  }
  
  /**
   * Returns the original FieldInfo.
   */
  public FieldInfo getDelegate() {
    return delegate;
  }
  
  /**
   * Returns the vector encoding from the delegate.
   */
  public VectorEncoding getVectorEncoding() {
    return delegate.getVectorEncoding();
  }
  
  /**
   * Returns the vector similarity function from the delegate.
   */
  public VectorSimilarityFunction getVectorSimilarityFunction() {
    return delegate.getVectorSimilarityFunction();
  }
  
  /**
   * Returns true if the field has vector values from the delegate.
   */
  public boolean hasVectorValues() {
    return delegate.hasVectorValues();
  }
  
  @Override
  public String toString() {
    return "SlicedFieldInfo{" +
        "delegate=" + delegate +
        ", vectorSlice=" + vectorSlice +
        '}';
  }
  
  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    SlicedFieldInfo that = (SlicedFieldInfo) o;
    return Objects.equals(delegate, that.delegate) &&
        Objects.equals(vectorSlice, that.vectorSlice);
  }
  
  @Override
  public int hashCode() {
    return Objects.hash(delegate, vectorSlice);
  }
}